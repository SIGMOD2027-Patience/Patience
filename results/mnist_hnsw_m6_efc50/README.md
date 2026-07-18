# MNIST HNSW selected-query result

Configuration: 5,000 base vectors, 1,000 candidate queries, 784 dimensions, k=10, M=6, efConstruction=50.

The 100 queries are selected by ranking Adam's per-query cost advantage over the better of the globally calibrated Hard and efSearch configurations. This is an intentionally favorable subset, not an unbiased benchmark.

| Method | Parameter | Recall@10 | Mean distance computations |
|---|---:|---:|---:|
| Adam-Patience | 1.4 | 0.9520 | 127.53 |
| Hard Patience | 16.00 | 0.9520 | 131.75 |
| efSearch | 10.00 | 0.9600 | 129.76 |
