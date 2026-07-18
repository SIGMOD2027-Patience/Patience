#include "graph_search.hpp"
#include "patience.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    const std::vector<float> vectors = {
        1, 0,  1, 1,  0, 1,  -1, 1,  -1, 0,  -1, -1,  0, -1,  1, -1};
    const std::vector<std::vector<std::size_t>> adjacency = {
        {1, 7}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}, {6, 0}};
    const float query[2] = {0.8F, 0.2F};
    const std::array graphs{patience::GraphKind::HNSW,
                            patience::GraphKind::NSG,
                            patience::GraphKind::Vamana};
    const std::array metrics{patience::MetricKind::L2,
                             patience::MetricKind::Cosine,
                             patience::MetricKind::MIPS};

    std::size_t tested = 0;
    for (const auto graph : graphs) {
        for (const auto metric : metrics) {
            patience::GraphSearcher searcher(graph, metric, 2, vectors, adjacency, 4);
            const auto result = searcher.search(query, 2, 6);
            if (result.labels.size() != 2 || result.trace.total_cost == 0 ||
                result.trace.points.empty()) {
                throw std::runtime_error("graph/metric search did not produce a complete trace");
            }
            (void)patience::HardPatience(3).evaluate(result.trace);
            (void)patience::AdamPatience(1.0).evaluate(result.trace);
            std::cout << "passed " << patience::to_string(graph) << " / "
                      << patience::to_string(metric) << '\n';
            ++tested;
        }
    }
    if (tested != 9) {
        throw std::runtime_error("the complete 3x3 support matrix was not tested");
    }
    return 0;
}
