This repository contains the source code for the SIGMOD 2027 Round 3 submission **Adam-Patience: A Novel Paradigm Beyond Search Width for Efficient Approximate Vector Search on Graphs**.

# Adam-Patience

---

## Introduction

Adam-Patience is an adaptive early-termination paradigm for graph-based approximate vector search. Instead of controlling search solely through a fixed search-width parameter, it monitors the normalized progress of the visible top-k result and terminates when additional graph exploration provides insufficient improvement.

The project is written in C++17 and supports the following complete matrix:

| Graph index | L2 | Cosine | MIPS |
|---|:---:|:---:|:---:|
| HNSW | ✓ | ✓ | ✓ |
| NSG | ✓ | ✓ | ✓ |
| Vamana | ✓ | ✓ | ✓ |

Hard Patience and Adam-Patience share a graph-independent trace representation. HNSW supplies its level-0 graph and the entry point obtained from upper-layer descent; NSG and Vamana supply their native adjacency lists and navigation points. The common search implementation performs the actual graph traversal, records top-k changes and distance-computation cost, and applies either stopping policy.

## Usage

Build and test the project:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the bundled HNSW example:

```bash
./build/hnsw_patience_demo
```

Run all nine graph/metric combinations:

```bash
./build/all_graphs_metrics
```

## C++ API

```cpp
#include "graph_search.hpp"
#include "patience.hpp"

patience::GraphSearcher searcher(
    patience::GraphKind::NSG,
    patience::MetricKind::Cosine,
    dimension,
    vectors,
    adjacency,
    entry_point);

auto search = searcher.search(query, k, search_width);
auto hard = patience::HardPatience(100).evaluate(search.trace);
auto adam = patience::AdamPatience(3.0, 0.9, 0.99).evaluate(search.trace);
```

`AdamPatience(3.0)` uses the raw stopping threshold `10^-3`.

## Dataset Format

The generic graph layer accepts:

- a contiguous row-major `float` vector array;
- vector dimension;
- an adjacency list indexed by vector ID;
- the graph navigation entry point;
- `k` and the baseline search width.

For cosine search, vectors may be supplied directly because cosine normalization is computed by the metric implementation. MIPS minimizes negative inner product. L2 uses squared Euclidean distance.

## Project Structure

- `include/patience.hpp`, `src/patience.cpp`: Hard Patience and Adam-Patience.
- `include/graph_search.hpp`, `src/graph_search.cpp`: HNSW/NSG/Vamana traversal with L2, cosine, and MIPS.
- `third_party/hnswlib/`: bundled HNSW implementation.
- `examples/`: runnable HNSW and 3×3 support-matrix examples.
- `tests/`: C++ policy tests and all graph/metric combinations.
- `index.html`: repository web-page crawler directive supplied with the artifact.

## Requirements

- Linux
- CMake 3.14 or newer
- a C++17 compiler (GCC 9.5 or newer is supported)
