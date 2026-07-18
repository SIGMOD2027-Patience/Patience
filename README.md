# Hard Patience and Adam-patience for HNSW

This repository is a self-contained C++17 implementation of two HNSW early-termination policies:

- **Hard Patience** stops after `tau` distance computations without a top-k result change.
- **Adam-patience** uses an Adam-style bias-corrected progress score with configurable `Lambda`, `beta1`, and `beta2`.

The repository vendors the complete HNSW implementation needed by the example. It does not depend on source files or trace files outside this directory.

## Layout

- `include/patience.hpp`: public C++ API.
- `src/patience.cpp`: complete Hard Patience and Adam-patience implementation.
- `third_party/hnswlib/`: bundled HNSW index, distance spaces, brute-force index, and visited-list implementation.
- `examples/hnsw_demo.cpp`: builds an HNSW index, collects a search trace, and runs both policies.
- `tests/test_patience.cpp`: standalone C++ behavior and validation tests.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the complete HNSW example:

```bash
./build/hnsw_patience_demo
```

## C++ usage

```cpp
#include "patience.hpp"

patience::SearchTrace trace = /* trace produced during HNSW search */;
auto hard = patience::HardPatience(100).evaluate(trace);
auto adam = patience::AdamPatience(3.0, 0.9, 0.99).evaluate(trace);
```

`AdamPatience(3.0)` means a raw stopping threshold of `10^-3`.
