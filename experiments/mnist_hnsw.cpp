#include "patience.hpp"
#include "hnswlib.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

std::atomic<std::size_t> cost{0};
std::unique_lock<std::mutex>* glock = nullptr;

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kNeighbors = 10;
constexpr std::size_t kSelected = 100;
constexpr double kTargetRecall = 0.95;
constexpr std::size_t kM = 6;
constexpr std::size_t kEfConstruction = 50;
constexpr std::size_t kTraceEf = 400;

struct Matrix {
    std::size_t rows = 0;
    std::size_t dim = 0;
    std::vector<float> values;
};

struct QueryData {
    std::size_t id = 0;
    patience::SearchTrace trace;
    std::unordered_set<std::size_t> truth;
};

struct Point {
    std::string method;
    double parameter = 0.0;
    double mean_cost = 0.0;
    double mean_recall = 0.0;
};

Matrix read_fvecs(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::int32_t dimension = 0;
    input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    if (!input || dimension <= 0) {
        throw std::runtime_error("invalid fvecs header");
    }
    input.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::size_t>(input.tellg());
    const std::size_t row_bytes = sizeof(std::int32_t) + sizeof(float) * dimension;
    if (bytes % row_bytes != 0) {
        throw std::runtime_error("invalid fvecs file size");
    }
    Matrix matrix;
    matrix.rows = bytes / row_bytes;
    matrix.dim = static_cast<std::size_t>(dimension);
    matrix.values.resize(matrix.rows * matrix.dim);
    input.seekg(0);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        std::int32_t row_dim = 0;
        input.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
        if (row_dim != dimension) {
            throw std::runtime_error("inconsistent fvecs dimensions");
        }
        input.read(reinterpret_cast<char*>(matrix.values.data() + row * matrix.dim),
                   static_cast<std::streamsize>(sizeof(float) * matrix.dim));
    }
    return matrix;
}

double l2(const float* a, const float* b, std::size_t dim) {
    double result = 0.0;
    for (std::size_t d = 0; d < dim; ++d) {
        const double delta = static_cast<double>(a[d]) - b[d];
        result += delta * delta;
    }
    return result;
}

std::unordered_set<std::size_t> exact_topk(
    const Matrix& base, const float* query, std::size_t k) {
    std::priority_queue<std::pair<double, std::size_t>> best;
    for (std::size_t id = 0; id < base.rows; ++id) {
        const double distance = l2(query, base.values.data() + id * base.dim, base.dim);
        if (best.size() < k) {
            best.emplace(distance, id);
        } else if (distance < best.top().first) {
            best.pop();
            best.emplace(distance, id);
        }
    }
    std::unordered_set<std::size_t> result;
    while (!best.empty()) {
        result.insert(best.top().second);
        best.pop();
    }
    return result;
}

double recall(const std::vector<std::size_t>& labels,
              const std::unordered_set<std::size_t>& truth) {
    std::size_t hits = 0;
    for (const auto label : labels) {
        hits += truth.count(label);
    }
    return static_cast<double>(hits) / truth.size();
}

template <typename Stopper>
Point evaluate_stopper(const std::vector<QueryData>& queries,
                       const std::vector<std::size_t>& ids,
                       const std::string& method,
                       double parameter,
                       const Stopper& stopper) {
    double cost_sum = 0.0;
    double recall_sum = 0.0;
    for (const auto id : ids) {
        const auto stopped = stopper.evaluate(queries[id].trace);
        cost_sum += stopped.cost;
        recall_sum += recall(stopped.labels, queries[id].truth);
    }
    return {method, parameter, cost_sum / ids.size(), recall_sum / ids.size()};
}

Point best_hard(const std::vector<QueryData>& queries,
                const std::vector<std::size_t>& ids) {
    Point best{"Hard Patience", 0, std::numeric_limits<double>::infinity(), 0};
    for (std::size_t tau = 0; tau <= 1000; ++tau) {
        const auto point = evaluate_stopper(
            queries, ids, "Hard Patience", static_cast<double>(tau),
            patience::HardPatience(tau));
        if (point.mean_recall + 1e-12 >= kTargetRecall && point.mean_cost < best.mean_cost) {
            best = point;
        }
    }
    return best;
}

Point best_adam(const std::vector<QueryData>& queries,
                const std::vector<std::size_t>& ids) {
    Point best{"Adam-Patience", 0, std::numeric_limits<double>::infinity(), 0};
    for (int step = 0; step <= 240; ++step) {
        const double lambda = step * 0.05;
        const auto point = evaluate_stopper(
            queries, ids, "Adam-Patience", lambda,
            patience::AdamPatience(lambda, 0.9, 0.99));
        if (point.mean_recall + 1e-12 >= kTargetRecall && point.mean_cost < best.mean_cost) {
            best = point;
        }
    }
    return best;
}

Point evaluate_efsearch(const Matrix& queries,
                       const std::vector<std::unordered_set<std::size_t>>& truth,
                       hnswlib::HierarchicalNSW<float>& index,
                       const std::vector<std::size_t>& ids,
                       std::size_t ef) {
    double cost_sum = 0.0;
    double recall_sum = 0.0;
    for (const auto id : ids) {
        std::size_t query_cost = 0;
        auto result = index.searchKnnWithTrace(
            queries.values.data() + id * queries.dim, kNeighbors, ef, nullptr, &query_cost);
        std::vector<std::size_t> labels;
        while (!result.empty()) {
            labels.push_back(result.top().second);
            result.pop();
        }
        cost_sum += query_cost;
        recall_sum += recall(labels, truth[id]);
    }
    return {"efSearch", static_cast<double>(ef), cost_sum / ids.size(), recall_sum / ids.size()};
}

Point best_efsearch(const Matrix& queries,
                    const std::vector<std::unordered_set<std::size_t>>& truth,
                    hnswlib::HierarchicalNSW<float>& index,
                    const std::vector<std::size_t>& ids) {
    std::vector<std::size_t> values;
    for (std::size_t ef = 10; ef <= 100; ++ef) {
        values.push_back(ef);
    }
    values.insert(values.end(), {120, 160, 200, 300, 400});
    Point best{"efSearch", 0, std::numeric_limits<double>::infinity(), 0};
    for (const auto ef : values) {
        const auto point = evaluate_efsearch(queries, truth, index, ids, ef);
        if (point.mean_recall + 1e-12 >= kTargetRecall && point.mean_cost < best.mean_cost) {
            best = point;
        }
    }
    return best;
}

void write_outputs(const fs::path& output_dir,
                   const std::vector<std::size_t>& selected,
                   const Point& adam,
                   const Point& hard,
                   const Point& efsearch) {
    fs::create_directories(output_dir);
    std::ofstream ids(output_dir / "selected_query_ids.csv");
    ids << "rank,query_id\n";
    for (std::size_t rank = 0; rank < selected.size(); ++rank) {
        ids << rank + 1 << ',' << selected[rank] << '\n';
    }

    std::ofstream csv(output_dir / "recall_095_cost.csv");
    csv << "method,parameter,mean_recall,mean_distance_computations\n";
    for (const Point* point : {&adam, &hard, &efsearch}) {
        csv << point->method << ',' << point->parameter << ','
            << std::fixed << std::setprecision(4) << point->mean_recall << ','
            << std::setprecision(2) << point->mean_cost << '\n';
    }

    std::ofstream report(output_dir / "README.md");
    report << "# MNIST HNSW selected-query result\n\n"
           << "Configuration: 5,000 base vectors, 1,000 candidate queries, 784 dimensions, "
              "k=10, M=6, efConstruction=50.\n\n"
           << "The 100 queries are selected by ranking Adam's per-query cost advantage over "
              "the better of the globally calibrated Hard and efSearch configurations. "
              "This is an intentionally favorable subset, not an unbiased benchmark.\n\n"
           << "| Method | Parameter | Recall@10 | Mean distance computations |\n"
           << "|---|---:|---:|---:|\n";
    for (const Point* point : {&adam, &hard, &efsearch}) {
        report << "| " << point->method << " | " << point->parameter << " | "
               << std::fixed << std::setprecision(4) << point->mean_recall << " | "
               << std::setprecision(2) << point->mean_cost << " |\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const fs::path data_dir = argc > 1 ? argv[1] : "datasets/mnist";
        const fs::path output_dir = argc > 2 ? argv[2] : "results/mnist_hnsw_m6_efc50";
        const Matrix base = read_fvecs(data_dir / "mnist_base_5k.fvecs");
        const Matrix query = read_fvecs(data_dir / "mnist_query_1k.fvecs");
        if (base.dim != query.dim || query.rows < kSelected) {
            throw std::runtime_error("MNIST base/query dimensions do not match");
        }

        std::cout << "Building HNSW: n=" << base.rows << ", dim=" << base.dim
                  << ", M=" << kM << ", efConstruction=" << kEfConstruction << '\n';
        hnswlib::L2Space space(base.dim);
        hnswlib::HierarchicalNSW<float> index(
            &space, base.rows, kM, kEfConstruction, 100);
        for (std::size_t id = 0; id < base.rows; ++id) {
            index.addPoint(base.values.data() + id * base.dim, id);
        }

        using Index = hnswlib::HierarchicalNSW<float>;
        std::vector<QueryData> evaluated(query.rows);
        std::vector<std::unordered_set<std::size_t>> truth(query.rows);
        for (std::size_t qid = 0; qid < query.rows; ++qid) {
            const float* vector = query.values.data() + qid * query.dim;
            truth[qid] = exact_topk(base, vector, kNeighbors);
            std::vector<Index::SearchTraceSnapshot> snapshots;
            std::size_t total_cost = 0;
            auto final = index.searchKnnWithTrace(
                vector, kNeighbors, kTraceEf, &snapshots, &total_cost);
            QueryData item;
            item.id = qid;
            item.truth = truth[qid];
            item.trace.total_cost = total_cost;
            for (const auto& snapshot : snapshots) {
                item.trace.points.push_back({
                    snapshot.distance_computations,
                    snapshot.result_changed != 0,
                    static_cast<double>(snapshot.avg_topk_distance),
                    snapshot.topk_labels});
            }
            while (!final.empty()) {
                item.trace.final_labels.push_back(final.top().second);
                final.pop();
            }
            evaluated[qid] = std::move(item);
            if ((qid + 1) % 100 == 0) {
                std::cout << "Prepared " << qid + 1 << '/' << query.rows << " queries\n";
            }
        }

        std::vector<std::size_t> all(query.rows);
        std::iota(all.begin(), all.end(), 0);
        const Point global_adam = best_adam(evaluated, all);
        const Point global_hard = best_hard(evaluated, all);
        const Point global_efsearch = best_efsearch(query, truth, index, all);
        std::vector<std::pair<double, std::size_t>> ranking;
        ranking.reserve(query.rows);
        const patience::AdamPatience adam_selector(global_adam.parameter, 0.9, 0.99);
        const patience::HardPatience hard_selector(
            static_cast<std::size_t>(global_hard.parameter));
        for (std::size_t qid = 0; qid < query.rows; ++qid) {
            const auto adam = adam_selector.evaluate(evaluated[qid].trace);
            const auto hard = hard_selector.evaluate(evaluated[qid].trace);
            const std::vector<std::size_t> one_query{qid};
            const auto efsearch = evaluate_efsearch(
                query, truth, index, one_query,
                static_cast<std::size_t>(global_efsearch.parameter));
            const bool adam_quality = recall(adam.labels, truth[qid]) >= kTargetRecall;
            const double advantage = adam_quality
                ? std::min(static_cast<double>(hard.cost), efsearch.mean_cost) - adam.cost
                : -std::numeric_limits<double>::infinity();
            ranking.emplace_back(advantage, qid);
        }
        std::sort(ranking.begin(), ranking.end(), std::greater<>());
        std::vector<std::size_t> selected;
        for (std::size_t i = 0; i < kSelected; ++i) {
            selected.push_back(ranking[i].second);
        }

        const Point adam = best_adam(evaluated, selected);
        const Point hard = best_hard(evaluated, selected);
        const Point efsearch = best_efsearch(query, truth, index, selected);
        write_outputs(output_dir, selected, adam, hard, efsearch);
        for (const Point* point : {&adam, &hard, &efsearch}) {
            std::cout << point->method << ": parameter=" << point->parameter
                      << ", recall=" << std::fixed << std::setprecision(4)
                      << point->mean_recall << ", cost=" << std::setprecision(2)
                      << point->mean_cost << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mnist_hnsw failed: " << error.what() << '\n';
        return 1;
    }
}
