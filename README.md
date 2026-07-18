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

Run the reproducible MNIST experiment:

```bash
./build/mnist_hnsw datasets/mnist results/mnist_hnsw_m4_efc50_efs30
```

## MNIST HNSW Experiment

The repository includes 5,000 MNIST base vectors and 1,000 candidate query vectors in `fvecs` format. Each vector contains 784 floating-point pixel values. The experiment computes exact L2 top-10 ground truth, builds a lower-quality HNSW with `M=4` and `efConstruction=50`, and uses an `efSearch=30` trace cap for calibrating the stopping policies.

The program first calibrates global Adam-Patience, Hard Patience, and efSearch parameters on all 1,000 candidate queries. A query is eligible only when all three methods have per-query Recall@10 at least 0.90 and its costs satisfy `Adam < Hard < efSearch`. Eligible queries are ranked by `efSearch cost - Adam cost`, and the top 100 are retained. This is deliberately a favorable subset requested for the demonstration; it must not be interpreted as an unbiased MNIST benchmark.

For those fixed 100 query IDs, each method independently selects its lowest-cost parameter point with aggregate Recall@10 at least 0.90:

| Method | Parameter | Recall@10 | Mean distance computations |
|---|---:|---:|---:|
| Adam-Patience | Lambda=1.15 | 0.9020 | **125.12** |
| Hard Patience | tau=11 | 0.9040 | 128.24 |
| efSearch | ef=13 | 0.9040 | 129.44 |

Adam-Patience uses 2.43% fewer distance computations than Hard Patience; Hard Patience uses 0.93% fewer than efSearch. The exact query IDs and machine-readable results are stored in `results/mnist_hnsw_m4_efc50_efs30/`.

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

The bundled MNIST files use the standard `fvecs` row encoding: one signed 32-bit dimension value followed by that row's 784 `float32` components. Their SHA-256 checksums are:

```text
b4ca5da0217117b308ca3b219fc94016515e562b60477d102231a2f95cabec77  mnist_base_5k.fvecs
2f97ca6265139d1ea4c7c94c1969f810ecd91d3c56b0506e0e5b56b3abaf9eb1  mnist_query_1k.fvecs
```

## Project Structure

- `include/patience.hpp`, `src/patience.cpp`: Hard Patience and Adam-Patience.
- `include/graph_search.hpp`, `src/graph_search.cpp`: HNSW/NSG/Vamana traversal with L2, cosine, and MIPS.
- `third_party/hnswlib/`: bundled HNSW implementation.
- `examples/`: runnable HNSW and 3×3 support-matrix examples.
- `experiments/mnist_hnsw.cpp`: real MNIST HNSW build, query selection, and recall/cost comparison.
- `datasets/mnist/`: bundled MNIST base and query vectors.
- `results/mnist_hnsw_m4_efc50_efs30/`: selected query IDs and Recall@10≥0.90 results.
- `tests/`: C++ policy tests and all graph/metric combinations.
- `index.html`: repository web-page crawler directive supplied with the artifact.

## Requirements

- Linux
- CMake 3.14 or newer
- a C++17 compiler (GCC 9.5 or newer is supported)
