#include "graph_search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace patience {
namespace {

using Candidate = std::pair<double, std::size_t>;

struct MinDistance {
    bool operator()(const Candidate& a, const Candidate& b) const noexcept {
        return a.first > b.first;
    }
};

std::vector<Candidate> sorted_topk(
    const std::priority_queue<Candidate>& heap, std::size_t k) {
    auto copy = heap;
    std::vector<Candidate> result;
    while (!copy.empty()) {
        result.push_back(copy.top());
        copy.pop();
    }
    std::sort(result.begin(), result.end());
    if (result.size() > k) {
        result.resize(k);
    }
    return result;
}

}  // namespace

const char* to_string(GraphKind kind) noexcept {
    switch (kind) {
        case GraphKind::HNSW: return "HNSW";
        case GraphKind::NSG: return "NSG";
        case GraphKind::Vamana: return "Vamana";
    }
    return "unknown";
}

const char* to_string(MetricKind metric) noexcept {
    switch (metric) {
        case MetricKind::L2: return "L2";
        case MetricKind::Cosine: return "cosine";
        case MetricKind::MIPS: return "MIPS";
    }
    return "unknown";
}

GraphSearcher::GraphSearcher(
    GraphKind graph,
    MetricKind metric,
    std::size_t dimension,
    std::vector<float> vectors,
    std::vector<std::vector<std::size_t>> adjacency,
    std::size_t entry_point)
    : graph_(graph),
      metric_(metric),
      dimension_(dimension),
      vectors_(std::move(vectors)),
      adjacency_(std::move(adjacency)),
      entry_point_(entry_point) {
    if (dimension_ == 0 || adjacency_.empty() ||
        vectors_.size() != adjacency_.size() * dimension_) {
        throw std::invalid_argument("vector storage and graph dimensions do not match");
    }
    if (entry_point_ >= adjacency_.size()) {
        throw std::invalid_argument("entry point is outside the graph");
    }
    for (const auto& neighbors : adjacency_) {
        for (const auto id : neighbors) {
            if (id >= adjacency_.size()) {
                throw std::invalid_argument("graph contains an invalid neighbor id");
            }
        }
    }
}

double GraphSearcher::distance(const float* query, std::size_t id) const {
    const float* point = vectors_.data() + id * dimension_;
    double dot = 0.0;
    double query_norm = 0.0;
    double point_norm = 0.0;
    double l2 = 0.0;
    for (std::size_t d = 0; d < dimension_; ++d) {
        const double q = query[d];
        const double p = point[d];
        const double delta = q - p;
        l2 += delta * delta;
        dot += q * p;
        query_norm += q * q;
        point_norm += p * p;
    }
    if (metric_ == MetricKind::L2) {
        return l2;
    }
    if (metric_ == MetricKind::MIPS) {
        return -dot;
    }
    const double denominator = std::sqrt(query_norm * point_norm);
    return denominator > 0.0 ? 1.0 - dot / denominator : 1.0;
}

GraphSearchResult GraphSearcher::search(
    const float* query, std::size_t k, std::size_t search_width) const {
    if (query == nullptr || k == 0 || search_width < k) {
        throw std::invalid_argument("query must be non-null and search_width must be >= k > 0");
    }

    std::priority_queue<Candidate, std::vector<Candidate>, MinDistance> frontier;
    std::priority_queue<Candidate> best;
    std::vector<unsigned char> visited(adjacency_.size(), 0);
    std::size_t cost = 0;
    std::unordered_set<std::size_t> previous_topk;

    const double entry_distance = distance(query, entry_point_);
    ++cost;
    frontier.emplace(entry_distance, entry_point_);
    best.emplace(entry_distance, entry_point_);
    visited[entry_point_] = 1;

    GraphSearchResult output;
    auto record = [&](bool force_changed) {
        const auto visible = sorted_topk(best, k);
        if (visible.size() < k) {
            return;
        }
        std::unordered_set<std::size_t> current;
        std::vector<std::size_t> labels;
        double energy = 0.0;
        for (const auto& item : visible) {
            current.insert(item.second);
            labels.push_back(item.second);
            // MIPS distances can be negative, so use a positive monotone energy.
            energy += metric_ == MetricKind::MIPS ? std::exp(item.first) : item.first;
        }
        const bool changed = force_changed || current != previous_topk;
        previous_topk = std::move(current);
        output.trace.points.push_back({cost, changed, energy / k, std::move(labels)});
    };

    while (!frontier.empty()) {
        const auto current = frontier.top();
        frontier.pop();
        if (best.size() >= search_width && current.first > best.top().first) {
            break;
        }
        for (const auto neighbor : adjacency_[current.second]) {
            if (visited[neighbor]) {
                continue;
            }
            visited[neighbor] = 1;
            const double d = distance(query, neighbor);
            ++cost;
            if (best.size() < search_width || d < best.top().first) {
                frontier.emplace(d, neighbor);
                best.emplace(d, neighbor);
                if (best.size() > search_width) {
                    best.pop();
                }
            }
            record(output.trace.points.empty());
        }
    }

    const auto final = sorted_topk(best, k);
    for (const auto& item : final) {
        output.labels.push_back(item.second);
    }
    output.trace.total_cost = cost;
    output.trace.final_labels = output.labels;
    return output;
}

}  // namespace patience
