#include "patience.hpp"
#include "hnswlib.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <vector>

std::atomic<std::size_t> cost{0};
std::unique_lock<std::mutex>* glock = nullptr;

int main() {
    constexpr std::size_t dimension = 2;
    constexpr std::size_t count = 200;
    constexpr std::size_t k = 5;

    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> index(&space, count, 12, 100, 42);
    std::vector<float> data(count * dimension);
    for (std::size_t i = 0; i < count; ++i) {
        data[i * dimension] = static_cast<float>(i % 20);
        data[i * dimension + 1] = static_cast<float>(i / 20);
        index.addPoint(data.data() + i * dimension, i);
    }

    const float query[dimension] = {7.2F, 4.8F};
    using Index = hnswlib::HierarchicalNSW<float>;
    std::vector<Index::SearchTraceSnapshot> snapshots;
    std::size_t total_cost = 0;
    auto neighbors = index.searchKnnWithTrace(query, k, 100, &snapshots, &total_cost);

    patience::SearchTrace trace;
    trace.total_cost = total_cost;
    for (const auto& snapshot : snapshots) {
        trace.points.push_back({snapshot.distance_computations,
                                snapshot.result_changed != 0,
                                static_cast<double>(snapshot.avg_topk_distance),
                                snapshot.topk_labels});
    }
    while (!neighbors.empty()) {
        trace.final_labels.push_back(neighbors.top().second);
        neighbors.pop();
    }

    const auto hard = patience::HardPatience(30).evaluate(trace);
    const auto adam = patience::AdamPatience(2.0).evaluate(trace);
    std::cout << "natural cost: " << total_cost << '\n'
              << "hard patience cost: " << hard.cost << '\n'
              << "adam-patience cost: " << adam.cost << '\n';
    return 0;
}
