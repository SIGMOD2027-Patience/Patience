# Bundled MNIST vectors

This directory contains a compact MNIST subset for the repository's reproducible HNSW experiment:

- `mnist_base_5k.fvecs`: 5,000 base vectors.
- `mnist_query_1k.fvecs`: 1,000 candidate query vectors.
- dimension: 784.
- element type: `float32`.
- row format: `int32 dimension`, followed by 784 values.

The experiment treats the image pixels as vectors and uses squared L2 distance. Exact top-10 neighbors are computed at runtime, so no precomputed ground-truth file is required.

MNIST was created by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges from NIST handwritten-digit data. Original dataset page: <https://yann.lecun.org/exdb/mnist/>. The redistributed subset is provided under the MNIST dataset terms (commonly documented as CC BY-SA 3.0); retain this attribution when redistributing it.

SHA-256:

```text
b4ca5da0217117b308ca3b219fc94016515e562b60477d102231a2f95cabec77  mnist_base_5k.fvecs
2f97ca6265139d1ea4c7c94c1969f810ecd91d3c56b0506e0e5b56b3abaf9eb1  mnist_query_1k.fvecs
```
