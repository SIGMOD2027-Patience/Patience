#pragma once

#include "patience.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace patience {

enum class GraphKind { HNSW, NSG, Vamana };
enum class MetricKind { L2, Cosine, MIPS };

const char* to_string(GraphKind kind) noexcept;
const char* to_string(MetricKind metric) noexcept;

struct GraphSearchResult {
    SearchTrace trace;
    std::vector<std::size_t> labels;
};

// Common base-layer traversal used by HNSW, NSG, and Vamana.  HNSW callers
// supply the level-0 graph and the entry point reached by upper-layer descent;
// NSG and Vamana callers supply their native graph and navigation point.
class GraphSearcher {
public:
    GraphSearcher(GraphKind graph,
                  MetricKind metric,
                  std::size_t dimension,
                  std::vector<float> vectors,
                  std::vector<std::vector<std::size_t>> adjacency,
                  std::size_t entry_point);

    GraphSearchResult search(const float* query, std::size_t k, std::size_t search_width) const;

    GraphKind graph_kind() const noexcept { return graph_; }
    MetricKind metric_kind() const noexcept { return metric_; }
    std::size_t size() const noexcept { return adjacency_.size(); }
    std::size_t dimension() const noexcept { return dimension_; }

private:
    double distance(const float* query, std::size_t id) const;

    GraphKind graph_;
    MetricKind metric_;
    std::size_t dimension_;
    std::vector<float> vectors_;
    std::vector<std::vector<std::size_t>> adjacency_;
    std::size_t entry_point_;
};

}  // namespace patience
