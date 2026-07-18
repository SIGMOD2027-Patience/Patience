#!/usr/bin/env python3
"""Hard Patience and Adam-patience implementations for saved HNSW traces.

The script uses the trace files already produced for the paper datasets.
It fixes beta1=0.9 and beta2=0.99, sweeps the log-threshold Lambda for
Adam-patience where lambda=10^{-Lambda}, and sweeps tau for hard patience
under the same trace-replay metric: mean recall@k and mean distance
computations per query.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/home/kai1/hnsw")
TRACE_DIR = BASE / "Cmp_results"
OUT_DIR = BASE / "Adam-patience" / "results"

BETA1 = 0.9
BETA2 = 0.99

DATASETS = {
    "deep1m": {
        "label": "DEEP1M",
        "M": 12,
        "efC": 120,
        "efS": 400,
        "k": 10,
        "trace": TRACE_DIR / "deep1m_hnsw_m12_efc120_efs400_q1000_k10_trace.csv",
    },
    "sift10M": {
        "label": "SIFT10M",
        "M": 16,
        "efC": 200,
        "efS": 800,
        "k": 10,
        "trace": TRACE_DIR / "sift10M_hnsw_m16_efc200_efs800_q1000_k10_trace.csv",
    },
    "gist": {
        "label": "GIST1M",
        "M": 16,
        "efC": 200,
        "efS": 800,
        "k": 10,
        "trace": TRACE_DIR / "gist_hnsw_m16_efc200_efs800_q1000_k10_trace.csv",
    },
    "tiny5m": {
        "label": "Tiny5M",
        "M": 16,
        "efC": 200,
        "efS": 800,
        "k": 10,
        "trace": TRACE_DIR / "tiny5m_hnsw_m16_efc200_efs800_q1000_k10_trace.csv",
    },
}

DEFAULT_LAMBDA_EXPONENTS = np.concatenate(([np.inf], np.arange(320.0, 0.0, -1.0)))


@dataclass
class QueryTrace:
    qid: int
    dist: np.ndarray
    recall: np.ndarray
    theta: np.ndarray
    total: int
    final_recall: float
    change_dists: np.ndarray
    change_recalls: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="deep1m,sift10M,gist,tiny5m",
        help="comma-separated dataset keys",
    )
    parser.add_argument(
        "--lambda-exponents",
        default=None,
        help="comma-separated Lambda values where raw lambda=10^{-Lambda}.",
    )
    parser.add_argument("--beta1", type=float, default=BETA1)
    parser.add_argument("--beta2", type=float, default=BETA2)
    return parser.parse_args()


def parse_lambda_exponents(exponent_text: str | None) -> np.ndarray:
    if exponent_text is None:
        return DEFAULT_LAMBDA_EXPONENTS.copy()
    values = []
    for token in exponent_text.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    values = sorted(set(values))
    if not values:
        raise ValueError("Lambda list is empty")
    return np.array(values, dtype=np.float64)


def lambda_thresholds(lambda_exponents: np.ndarray) -> np.ndarray:
    thresholds = np.zeros(len(lambda_exponents), dtype=np.float64)
    finite = np.isfinite(lambda_exponents)
    thresholds[finite] = np.power(10.0, -lambda_exponents[finite])
    return thresholds


def tau_grid(max_total: int) -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                np.arange(0, 200, 10),
                np.arange(200, 1000, 25),
                np.arange(1000, 4000, 100),
                np.arange(4000, int(max_total) + 1000, 500),
            ]
        )
    ).astype(int)


def load_queries(trace_path: Path) -> list[QueryTrace]:
    usecols = [
        "query_id",
        "distance_computations",
        "result_changed",
        "avg_topk_distance",
        "total_distance_computations",
        "recall",
    ]
    df = pd.read_csv(trace_path, usecols=usecols)
    df = df.sort_values(["query_id", "distance_computations"]).reset_index(drop=True)
    queries: list[QueryTrace] = []
    for qid, group in df.groupby("query_id", sort=False):
        group = group.sort_values("distance_computations")
        dist = group["distance_computations"].to_numpy(dtype=np.int64)
        recall = group["recall"].to_numpy(dtype=np.float64)
        avg = group["avg_topk_distance"].to_numpy(dtype=np.float64)
        finite_avg = avg[np.isfinite(avg)]
        if len(finite_avg) > 0 and float(np.min(finite_avg)) > 0.0:
            theta = np.log(np.maximum(avg, 1e-30))
        else:
            theta = avg.copy()
        total = int(group["total_distance_computations"].iloc[0])
        changed = group["result_changed"].to_numpy(dtype=np.int64) == 1
        queries.append(
            QueryTrace(
                qid=int(qid),
                dist=dist,
                recall=recall,
                theta=theta,
                total=total,
                final_recall=float(recall[-1]),
                change_dists=dist[changed],
                change_recalls=recall[changed],
            )
        )
    return queries


def hard_patience_stop(query: QueryTrace, tau: int) -> tuple[int, float]:
    """Return ``(cost, recall)`` when ``tau`` unchanged steps elapse.

    A result change resets the patience counter.  The natural trace end is a
    hard cap, so the returned cost never exceeds ``query.total``.
    """
    if tau < 0:
        raise ValueError("tau must be nonnegative")
    change_dists = query.change_dists
    change_recalls = query.change_recalls
    if len(change_dists) == 0:
        if len(query.dist) == 0:
            return min(tau, query.total), 0.0
        return min(int(query.dist[0]) + tau, query.total), float(query.recall[0])

    stop_after = int(change_dists[-1])
    recall = float(change_recalls[-1])
    for idx in range(len(change_dists)):
        is_last = idx == len(change_dists) - 1
        if is_last or int(change_dists[idx + 1] - change_dists[idx]) > tau:
            stop_after = int(change_dists[idx])
            recall = float(change_recalls[idx])
            break
    return min(stop_after + int(tau), query.total), recall


def sweep_hard_patience(queries: list[QueryTrace], taus: np.ndarray) -> pd.DataFrame:
    rows = []
    for tau in taus:
        vals = [hard_patience_stop(query, int(tau)) for query in queries]
        costs, recalls = zip(*vals)
        rows.append(
            {
                "method": "hard_patience",
                "param": "tau",
                "param_value": float(tau),
                "mean_cost": float(np.mean(costs)),
                "mean_recall": float(np.mean(recalls)),
            }
        )
    return pd.DataFrame(rows)


def score_after_zero_steps(
    r: int,
    n: int,
    m: float,
    u: float,
    beta1: float,
    beta2: float,
) -> float:
    if u <= 0.0:
        return math.inf
    denom = 1.0 - beta1 ** (n + r)
    if denom <= 0.0:
        return math.inf
    return (m / u) * ((beta1 / beta2) ** r) / denom


def first_zero_stop_step(
    gap: int,
    threshold: float,
    n: int,
    m: float,
    u: float,
    beta1: float,
    beta2: float,
) -> int | None:
    if gap <= 0 or threshold <= 0.0 or u <= 0.0:
        return None
    if score_after_zero_steps(gap, n, m, u, beta1, beta2) >= threshold:
        return None
    if score_after_zero_steps(1, n, m, u, beta1, beta2) < threshold:
        return 1
    lo, hi = 1, gap
    while lo < hi:
        mid = (lo + hi) // 2
        if score_after_zero_steps(mid, n, m, u, beta1, beta2) < threshold:
            hi = mid
        else:
            lo = mid + 1
    return lo


def adam_patience_stops(
    query: QueryTrace,
    lambdas: np.ndarray,
    beta1: float,
    beta2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay Adam-patience for several raw lambda thresholds at once."""
    if not 0.0 < beta1 < 1.0:
        raise ValueError("beta1 must be in (0, 1)")
    if not 0.0 < beta2 < 1.0:
        raise ValueError("beta2 must be in (0, 1)")
    lambdas = np.asarray(lambdas, dtype=np.float64)
    if lambdas.ndim != 1:
        raise ValueError("lambdas must be a one-dimensional array")
    if np.any(~np.isfinite(lambdas)) or np.any(lambdas < 0.0):
        raise ValueError("lambda thresholds must be finite and nonnegative")
    n_lambdas = len(lambdas)
    stop_cost = np.full(n_lambdas, query.total, dtype=np.float64)
    stop_recall = np.full(n_lambdas, query.final_recall, dtype=np.float64)
    stopped = np.zeros(n_lambdas, dtype=bool)

    if len(query.dist) == 0:
        stop_cost[:] = 0.0
        stop_recall[:] = 0.0
        return stop_cost, stop_recall

    prev_c = int(query.dist[0])
    theta_prev = float(query.theta[0])
    current_recall = float(query.recall[0])
    m = 0.0
    u = 0.0
    n = 0

    for pos in range(1, len(query.dist)):
        event_c = int(query.dist[pos])
        gap = max(0, event_c - prev_c - 1)
        if gap > 0 and not stopped.all():
            active = np.where(~stopped)[0]
            for idx in active:
                r = first_zero_stop_step(
                    gap=gap,
                    threshold=float(lambdas[idx]),
                    n=n,
                    m=m,
                    u=u,
                    beta1=beta1,
                    beta2=beta2,
                )
                if r is not None:
                    stop_cost[idx] = float(prev_c + r)
                    stop_recall[idx] = current_recall
                    stopped[idx] = True
        if gap > 0:
            n += gap
            m *= beta1 ** gap
            u *= beta2 ** gap

        theta = float(query.theta[pos])
        g = max(theta_prev - theta, 0.0)
        theta_prev = theta
        n += 1
        m = beta1 * m + (1.0 - beta1) * g
        u = max(beta2 * u, abs(g))
        current_recall = float(query.recall[pos])
        prev_c = event_c

        if u > 0.0 and not stopped.all():
            denom = 1.0 - beta1**n
            score = math.inf if denom <= 0.0 else m / (denom * u)
            newly_stopped = (~stopped) & (score < lambdas)
            if np.any(newly_stopped):
                stop_cost[newly_stopped] = float(event_c)
                stop_recall[newly_stopped] = current_recall
                stopped[newly_stopped] = True

    gap = max(0, query.total - prev_c)
    if gap > 0 and not stopped.all():
        active = np.where(~stopped)[0]
        for idx in active:
            r = first_zero_stop_step(
                gap=gap,
                threshold=float(lambdas[idx]),
                n=n,
                m=m,
                u=u,
                beta1=beta1,
                beta2=beta2,
            )
            if r is not None:
                stop_cost[idx] = float(prev_c + r)
                stop_recall[idx] = current_recall
                stopped[idx] = True

    return stop_cost, stop_recall


def sweep_adam_patience(
    queries: list[QueryTrace],
    lambda_exponents: np.ndarray,
    beta1: float,
    beta2: float,
) -> pd.DataFrame:
    thresholds = lambda_thresholds(lambda_exponents)
    all_costs = []
    all_recalls = []
    for query in queries:
        costs, recalls = adam_patience_stops(query, thresholds, beta1, beta2)
        all_costs.append(costs)
        all_recalls.append(recalls)
    cost_matrix = np.vstack(all_costs)
    recall_matrix = np.vstack(all_recalls)

    rows = []
    for idx, value in enumerate(lambda_exponents):
        rows.append(
            {
                "method": "adam_patience",
                "param": "Lambda",
                "param_value": float(value),
                "lambda_threshold": float(thresholds[idx]),
                "mean_cost": float(cost_matrix[:, idx].mean()),
                "mean_recall": float(recall_matrix[:, idx].mean()),
            }
        )
    return pd.DataFrame(rows)


def best_at_target(curve: pd.DataFrame, method: str, target: float) -> dict:
    df = curve[(curve["method"] == method) & (curve["mean_recall"] >= target - 1e-12)]
    if df.empty:
        return {
            "cost": np.nan,
            "recall": np.nan,
            "param_value": np.nan,
            "lambda_threshold": np.nan,
        }
    row = df.loc[df["mean_cost"].idxmin()]
    return {
        "cost": float(row["mean_cost"]),
        "recall": float(row["mean_recall"]),
        "param_value": float(row["param_value"]),
        "lambda_threshold": float(row["lambda_threshold"]) if "lambda_threshold" in row else np.nan,
    }


def summarize_dataset(dataset: str, config: dict, lambda_exponents: np.ndarray, beta1: float, beta2: float) -> dict:
    started = time.time()
    queries = load_queries(config["trace"])
    baseline_cost = float(np.mean([query.total for query in queries]))
    baseline_recall = float(np.mean([query.final_recall for query in queries]))
    max_total = max(query.total for query in queries)

    hard_df = sweep_hard_patience(queries, tau_grid(max_total))
    adam_df = sweep_adam_patience(queries, lambda_exponents, beta1, beta2)
    curve = pd.concat([hard_df, adam_df], ignore_index=True)

    curve.insert(0, "dataset", dataset)
    curve.insert(1, "label", config["label"])
    curve.insert(2, "graph", "hnsw")
    curve.insert(3, "M", config["M"])
    curve.insert(4, "efConstruction", config["efC"])
    curve.insert(5, "baseline_efSearch", config["efS"])
    curve.insert(6, "k", config["k"])
    curve.insert(7, "queries", len(queries))
    curve.insert(8, "beta1", beta1)
    curve.insert(9, "beta2", beta2)
    curve["baseline_mean_cost"] = baseline_cost
    curve["baseline_mean_recall"] = baseline_recall
    curve["recall_loss_vs_baseline"] = baseline_recall - curve["mean_recall"]
    curve["speedup_vs_baseline"] = baseline_cost / curve["mean_cost"]

    targets = [
        ("full", baseline_recall),
        ("0.995x", 0.995 * baseline_recall),
        ("0.99x", 0.99 * baseline_recall),
        ("0.98x", 0.98 * baseline_recall),
        ("0.95x", 0.95 * baseline_recall),
    ]
    iso_rows = []
    for target_name, target_recall in targets:
        hard = best_at_target(curve, "hard_patience", target_recall)
        adam = best_at_target(curve, "adam_patience", target_recall)
        delta = np.nan
        if np.isfinite(hard["cost"]) and np.isfinite(adam["cost"]):
            delta = 100.0 * (hard["cost"] - adam["cost"]) / hard["cost"]
        iso_rows.append(
            {
                "dataset": dataset,
                "label": config["label"],
                "target": target_name,
                "target_recall": target_recall,
                "hard_tau": hard["param_value"],
                "hard_cost": hard["cost"],
                "hard_recall": hard["recall"],
                "adam_Lambda": adam["param_value"],
                "adam_lambda_threshold": adam["lambda_threshold"],
                "adam_cost": adam["cost"],
                "adam_recall": adam["recall"],
                "adam_vs_hard_cost_delta_pct": delta,
            }
        )

    elapsed = time.time() - started
    return {
        "curve": curve,
        "iso": pd.DataFrame(iso_rows),
        "meta": {
            "dataset": dataset,
            "label": config["label"],
            "trace": str(config["trace"]),
            "queries": len(queries),
            "M": config["M"],
            "efConstruction": config["efC"],
            "baseline_efSearch": config["efS"],
            "k": config["k"],
            "beta1": beta1,
            "beta2": beta2,
            "baseline_mean_cost": baseline_cost,
            "baseline_mean_recall": baseline_recall,
            "Lambda_count": len(lambda_exponents),
            "tau_count": len(tau_grid(max_total)),
            "elapsed_seconds": elapsed,
        },
    }


def write_text_summary(path: Path, iso: pd.DataFrame, metadata: list[dict]) -> None:
    lines = []
    lines.append("Adam-patience replay summary")
    lines.append("=" * 80)
    if metadata:
        lines.append(f"Fixed constants: beta1={metadata[0]['beta1']}, beta2={metadata[0]['beta2']}")
    else:
        lines.append(f"Fixed constants: beta1={BETA1}, beta2={BETA2}")
    lines.append("Adam-patience user parameter: Lambda, with raw lambda=10^{-Lambda}")
    lines.append("Cost metric: mean distance computations per query")
    lines.append("Quality metric: mean recall@10")
    lines.append("")
    for meta in metadata:
        lines.append(
            f"{meta['label']}: M={meta['M']}, efConstruction={meta['efConstruction']}, "
            f"efSearch cap={meta['baseline_efSearch']}, queries={meta['queries']}, "
            f"baseline cost={meta['baseline_mean_cost']:.1f}, "
            f"baseline recall={meta['baseline_mean_recall']:.6f}"
        )
        df = iso[iso["dataset"] == meta["dataset"]]
        lines.append(
            f"{'target':<8}{'target_recall':>15}{'hard tau':>12}"
            f"{'hard cost':>14}{'Adam Lambda':>15}{'Adam cost':>14}"
            f"{'delta':>12}"
        )
        for row in df.itertuples(index=False):
            delta = "n/a" if not np.isfinite(row.adam_vs_hard_cost_delta_pct) else f"{row.adam_vs_hard_cost_delta_pct:+.2f}%"
            adam_lambda_text = "inf" if not np.isfinite(row.adam_Lambda) else f"{row.adam_Lambda:.4g}"
            lines.append(
                f"{row.target:<8}{row.target_recall:>15.6f}"
                f"{row.hard_tau:>12.4g}{row.hard_cost:>14.1f}"
                f"{adam_lambda_text:>15}{row.adam_cost:>14.1f}"
                f"{delta:>12}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_keys = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = [key for key in dataset_keys if key not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset keys: {unknown}")

    lambda_exponents = parse_lambda_exponents(args.lambda_exponents)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    curves = []
    isos = []
    metadata = []
    for dataset in dataset_keys:
        config = DATASETS[dataset]
        if not config["trace"].exists():
            raise FileNotFoundError(config["trace"])
        print(f"[{time.strftime('%H:%M:%S')}] replaying {config['label']} from {config['trace']}", flush=True)
        result = summarize_dataset(dataset, config, lambda_exponents, args.beta1, args.beta2)
        curves.append(result["curve"])
        isos.append(result["iso"])
        metadata.append(result["meta"])
        print(
            f"[{time.strftime('%H:%M:%S')}] done {config['label']} in "
            f"{result['meta']['elapsed_seconds']:.1f}s",
            flush=True,
        )

    all_curves = pd.concat(curves, ignore_index=True)
    all_iso = pd.concat(isos, ignore_index=True)

    curve_path = OUT_DIR / "adam_vs_hard_patience_curve.csv"
    iso_path = OUT_DIR / "adam_vs_hard_patience_iso_summary.csv"
    meta_path = OUT_DIR / "adam_vs_hard_patience_metadata.json"
    text_path = OUT_DIR / "adam_vs_hard_patience_summary.txt"

    all_curves.to_csv(curve_path, index=False)
    all_iso.to_csv(iso_path, index=False)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_text_summary(text_path, all_iso, metadata)

    print(f"wrote curve: {curve_path}")
    print(f"wrote iso summary: {iso_path}")
    print(f"wrote metadata: {meta_path}")
    print(f"wrote text summary: {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
