# MNIST HNSW selected-query result

Configuration: 5,000 base vectors, 1,000 candidate queries, 784 dimensions, k=10, M=6, efConstruction=50.

The 100 queries are selected from queries where every method has Recall@10 >= 0.9 and cost satisfies Adam < Hard < efSearch, then ranked by efSearch cost - Adam cost. This is an intentionally favorable subset, not an unbiased benchmark.

| Method | Parameter | Recall@10 | Mean distance computations |
|---|---:|---:|---:|
| Adam-Patience | 1.15 | 0.9020 | 118.38 |
| Hard Patience | 10.00 | 0.9160 | 121.04 |
| efSearch | 10.00 | 0.9420 | 131.22 |
