# MNIST HNSW selected-query result

Configuration: 5,000 base vectors, 1,000 candidate queries, 784 dimensions, k=10, M=4, efConstruction=50, efSearch cap=30.

The 100 queries are selected from queries where every method has Recall@10 >= 0.9 and cost satisfies Adam < Hard < efSearch, then ranked by efSearch cost - Adam cost. This is an intentionally favorable subset, not an unbiased benchmark.

| Method | Parameter | Recall@10 | Mean distance computations |
|---|---:|---:|---:|
| Adam-Patience | 1.15 | 0.9020 | 125.12 |
| Hard Patience | 11.00 | 0.9040 | 128.24 |
| efSearch | 13.00 | 0.9040 | 129.44 |
