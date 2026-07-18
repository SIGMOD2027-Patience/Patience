import unittest

import numpy as np

from run_adam_patience_replay import (
    QueryTrace,
    adam_patience_stops,
    hard_patience_stop,
    lambda_thresholds,
    sweep_adam_patience,
    sweep_hard_patience,
)


def make_trace(change_dists=(1, 4), change_recalls=(0.1, 0.8), total=10):
    change_dists = np.asarray(change_dists, dtype=np.int64)
    change_recalls = np.asarray(change_recalls, dtype=np.float64)
    return QueryTrace(
        qid=0,
        dist=change_dists.copy(),
        recall=change_recalls.copy(),
        theta=np.linspace(2.0, 1.0, len(change_dists)),
        total=total,
        final_recall=float(change_recalls[-1]),
        change_dists=change_dists,
        change_recalls=change_recalls,
    )


class HardPatienceTests(unittest.TestCase):
    def test_stops_in_first_gap_larger_than_tau(self):
        trace = make_trace(change_dists=(2, 8), change_recalls=(0.2, 0.9), total=12)
        self.assertEqual(hard_patience_stop(trace, 3), (5, 0.2))

    def test_reaches_total_when_patience_does_not_expire(self):
        trace = make_trace(change_dists=(2, 5, 8), change_recalls=(0.2, 0.5, 0.9), total=10)
        self.assertEqual(hard_patience_stop(trace, 3), (10, 0.9))

    def test_sweep_labels_only_hard_patience(self):
        result = sweep_hard_patience([make_trace()], np.array([1, 2]))
        self.assertEqual(set(result["method"]), {"hard_patience"})


class AdamPatienceTests(unittest.TestCase):
    def test_zero_threshold_never_stops_early(self):
        trace = make_trace(total=10)
        costs, recalls = adam_patience_stops(trace, np.array([0.0]), 0.9, 0.99)
        self.assertEqual(costs.tolist(), [10.0])
        self.assertEqual(recalls.tolist(), [0.8])

    def test_large_threshold_stops_after_score_is_initialized(self):
        trace = make_trace(change_dists=(1, 4), change_recalls=(0.1, 0.8), total=10)
        costs, recalls = adam_patience_stops(trace, np.array([2.0]), 0.9, 0.99)
        self.assertEqual(costs.tolist(), [4.0])
        self.assertEqual(recalls.tolist(), [0.8])

    def test_lambda_exponents_are_converted_to_thresholds(self):
        values = lambda_thresholds(np.array([np.inf, 3.0, 1.0]))
        np.testing.assert_allclose(values, [0.0, 1e-3, 1e-1])

    def test_sweep_labels_only_adam_patience(self):
        result = sweep_adam_patience([make_trace()], np.array([np.inf, 1.0]), 0.9, 0.99)
        self.assertEqual(set(result["method"]), {"adam_patience"})


if __name__ == "__main__":
    unittest.main()
