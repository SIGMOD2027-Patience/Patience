#include "graph_search.hpp"
#include "patience.hpp"

#include <array>
#include <iostream>
#include <vector>

int main() {
    const std::vector<float> vectors = {
        0, 0,  1, 0,  2, 0,  3, 0,  4, 0,
        5, 0,  6, 0,  7, 0,  8, 0,  9, 0};
    const std::vector<std::vector<std::size_t>> graph = {
        {1, 2}, {0, 2, 3}, {0, 1, 3, 4}, {1, 2, 4, 5}, {2, 3, 5, 6},
        {3, 4, 6, 7}, {4, 5, 7, 8}, {5, 6, 8, 9}, {6, 7, 9}, {7, 8}};
    const float query[2] = {8.2F, 0.1F};
    const std::array graphs{patience::GraphKind::HNSW,
                            patience::GraphKind::NSG,
                            patience::GraphKind::Vamana};
    const std::array metrics{patience::MetricKind::L2,
                             patience::MetricKind::Cosine,
                             patience::MetricKind::MIPS};

    for (const auto graph_kind : graphs) {
        for (const auto metric : metrics) {
            patience::GraphSearcher searcher(graph_kind, metric, 2, vectors, graph, 0);
            const auto search = searcher.search(query, 2, 8);
            const auto hard = patience::HardPatience(5).evaluate(search.trace);
            const auto adam = patience::AdamPatience(1.0).evaluate(search.trace);
            std::cout << patience::to_string(graph_kind) << " / "
                      << patience::to_string(metric)
                      << ": Adam=" << adam.cost
                      << ", Hard=" << hard.cost
                      << ", efSearch=" << search.trace.total_cost << '\n';
        }
    }
}
