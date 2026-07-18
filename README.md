# Hard Patience and Adam-patience

This directory contains exactly two HNSW trace-replay early-termination methods:

- **Hard Patience**: stop after `tau` consecutive distance computations without a result change.
- **Adam-patience**: track positive log-distance improvement with Adam-style first-moment and infinity-norm accumulators, and stop when the bias-corrected score falls below `lambda = 10^-Lambda`.

## Files

- `run_adam_patience_replay.py`: implementations, trace loading, parameter sweeps, and Hard-vs-Adam report generation.
- `run_beta_sensitivity_matrix.py`: Adam-patience beta sensitivity experiment.
- `tests/test_patience.py`: focused behavior and boundary tests for both methods.

## Run

```bash
python run_adam_patience_replay.py --datasets deep1m
python -m unittest discover -s tests -v
```

The replay script reads existing traces from `/home/kai1/hnsw/Cmp_results` and writes generated reports under `results/`.
