#!/usr/bin/env python3
"""Run a 3x3 Adam-patience beta sensitivity matrix on L2 HNSW traces."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/home/kai1/hnsw")
OUT_DIR = BASE / "Adam-patience" / "results" / "beta_sensitivity_matrix"
TRACE_DIR = BASE / "Cmp_results"

sys.path.insert(0, str(BASE / "Adam-patience"))
from run_adam_patience_replay import load_queries, lambda_thresholds  # noqa: E402


DATASETS = [
    ("DEEP1M", "deep1m_hnsw_m12_efc120_efs2000_q1000_k10"),
    ("SIFT10M", "sift10M_hnsw_m16_efc200_efs4000_q1000_k10"),
    ("GIST1M", "gist_hnsw_m16_efc200_efs3000_q1000_k10"),
    ("Tiny5M", "tiny5m_hnsw_m16_efc200_efs8000_q1000_k10"),
]
BETA1_VALUES = [0.85, 0.90, 0.95]
BETA2_VALUES = [0.90, 0.99, 0.999]
TARGETS = [round(x, 2) for x in np.arange(0.90, 1.00, 0.01)]
LAMBDA_EXPONENTS = np.concatenate(([np.inf], np.arange(800.0, 0.0, -1.0)))


def score_zero_steps(r, n, m, u, beta1, beta2):
    if u <= 0.0:
        return np.full_like(r, np.inf, dtype=np.float64)
    denom = 1.0 - np.power(beta1, n + r)
    out = np.full_like(r, np.inf, dtype=np.float64)
    valid = denom > 0.0
    out[valid] = (m / u) * np.power(beta1 / beta2, r[valid]) / denom[valid]
    return out


def first_zero_stop_steps_vectorized(gap, thresholds, n, m, u, beta1, beta2):
    if gap <= 0 or u <= 0.0:
        return np.full(len(thresholds), -1, dtype=np.int64)
    finite = thresholds > 0.0
    result = np.full(len(thresholds), -1, dtype=np.int64)
    if not np.any(finite):
        return result

    idx = np.where(finite)[0]
    th = thresholds[idx]
    end_r = np.full(len(idx), gap, dtype=np.int64)
    end_score = score_zero_steps(end_r.astype(np.float64), n, m, u, beta1, beta2)
    can_stop = end_score < th
    if not np.any(can_stop):
        return result

    idx = idx[can_stop]
    th = th[can_stop]
    lo = np.ones(len(idx), dtype=np.int64)
    hi = np.full(len(idx), gap, dtype=np.int64)
    while np.any(lo < hi):
        mid = (lo + hi) // 2
        score = score_zero_steps(mid.astype(np.float64), n, m, u, beta1, beta2)
        go_left = score < th
        hi = np.where(go_left, mid, hi)
        lo = np.where(go_left, lo, mid + 1)
    result[idx] = lo
    return result


def adam_patience_stops_fast(query, thresholds, beta1, beta2):
    n_lambdas = len(thresholds)
    stop_cost = np.full(n_lambdas, query.total, dtype=np.float64)
    stop_recall = np.full(n_lambdas, query.final_recall, dtype=np.float64)
    stopped = np.zeros(n_lambdas, dtype=bool)

    if len(query.dist) == 0:
        return np.zeros(n_lambdas), np.zeros(n_lambdas)

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
            steps = first_zero_stop_steps_vectorized(gap, thresholds[active], n, m, u, beta1, beta2)
            hit = steps > 0
            if np.any(hit):
                hit_idx = active[hit]
                stop_cost[hit_idx] = prev_c + steps[hit]
                stop_recall[hit_idx] = current_recall
                stopped[hit_idx] = True
        if gap > 0:
            n += gap
            m *= beta1**gap
            u *= beta2**gap

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
            newly = (~stopped) & (score < thresholds)
            if np.any(newly):
                stop_cost[newly] = event_c
                stop_recall[newly] = current_recall
                stopped[newly] = True

    gap = max(0, query.total - prev_c)
    if gap > 0 and not stopped.all():
        active = np.where(~stopped)[0]
        steps = first_zero_stop_steps_vectorized(gap, thresholds[active], n, m, u, beta1, beta2)
        hit = steps > 0
        if np.any(hit):
            hit_idx = active[hit]
            stop_cost[hit_idx] = prev_c + steps[hit]
            stop_recall[hit_idx] = current_recall

    return stop_cost, stop_recall


def sweep_dataset_beta(label, stem, beta1, beta2, thresholds):
    queries = load_queries(TRACE_DIR / f"{stem}_trace.csv")
    all_costs = []
    all_recalls = []
    for query in queries:
        costs, recalls = adam_patience_stops_fast(query, thresholds, beta1, beta2)
        all_costs.append(costs)
        all_recalls.append(recalls)
    cost_matrix = np.vstack(all_costs)
    recall_matrix = np.vstack(all_recalls)
    curve = pd.DataFrame(
        {
            "dataset": label,
            "stem": stem,
            "beta1": beta1,
            "beta2": beta2,
            "Lambda": LAMBDA_EXPONENTS,
            "lambda_threshold": thresholds,
            "mean_cost": cost_matrix.mean(axis=0),
            "mean_recall": recall_matrix.mean(axis=0),
        }
    )
    return curve


def run_task(task):
    label, stem, beta1, beta2 = task
    thresholds = lambda_thresholds(LAMBDA_EXPONENTS)
    print(f"{label}: beta1={beta1:g}, beta2={beta2:g}", flush=True)
    return sweep_dataset_beta(label, stem, beta1, beta2, thresholds)


def summarize_targets(curves):
    rows = []
    for (dataset, beta1, beta2), group in curves.groupby(["dataset", "beta1", "beta2"], sort=False):
        for target in TARGETS:
            feasible = group[group["mean_recall"] >= target - 1e-12]
            if feasible.empty:
                rows.append(
                    {
                        "dataset": dataset,
                        "target_recall": target,
                        "beta1": beta1,
                        "beta2": beta2,
                        "best_Lambda": np.nan,
                        "lambda_threshold": np.nan,
                        "mean_cost": np.nan,
                        "mean_recall": np.nan,
                        "reachable": False,
                    }
                )
                continue
            row = feasible.loc[feasible["mean_cost"].idxmin()]
            rows.append(
                {
                    "dataset": dataset,
                    "target_recall": target,
                    "beta1": beta1,
                    "beta2": beta2,
                    "best_Lambda": row["Lambda"],
                    "lambda_threshold": row["lambda_threshold"],
                    "mean_cost": row["mean_cost"],
                    "mean_recall": row["mean_recall"],
                    "reachable": True,
                }
            )
    return pd.DataFrame(rows)


def plot_target_lines(summary):
    for dataset, group in summary.groupby("dataset", sort=False):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for (beta1, beta2), part in group.groupby(["beta1", "beta2"], sort=True):
            part = part.sort_values("target_recall")
            label = rf"$\beta_1={beta1:g},\,\beta_2={beta2:g}$"
            ax.plot(part["target_recall"], part["mean_cost"], marker="o", linewidth=1.25, markersize=3.2, label=label)
        ax.set_title(f"{dataset}: Adam-patience beta sensitivity")
        ax.set_xlabel("Recall@10 target")
        ax.set_ylabel("Mean distance computations")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(ncol=3, fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{dataset.lower()}_target_cost_lines.pdf")
        fig.savefig(OUT_DIR / f"{dataset.lower()}_target_cost_lines.png", dpi=240)
        plt.close(fig)


def plot_heatmaps(summary):
    for target in TARGETS:
        pivot = (
            summary[summary["target_recall"] == target]
            .groupby(["beta1", "beta2"], as_index=False)["mean_cost"]
            .mean()
            .pivot(index="beta1", columns="beta2", values="mean_cost")
            .sort_index(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        im = ax.imshow(pivot.to_numpy(), cmap="viridis")
        ax.set_title(f"Mean cost across datasets at Recall@10={target:.2f}")
        ax.set_xlabel(r"$\beta_2$")
        ax.set_ylabel(r"$\beta_1$")
        ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{x:g}" for x in pivot.index])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                text = "n/a" if not np.isfinite(value) else f"{value:.0f}"
                ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, label="Mean distance computations")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"aggregate_heatmap_recall_{target:.2f}.pdf")
        fig.savefig(OUT_DIR / f"aggregate_heatmap_recall_{target:.2f}.png", dpi=240)
        plt.close(fig)


def write_markdown(summary):
    agg = (
        summary.groupby(["target_recall", "beta1", "beta2"], as_index=False)["mean_cost"]
        .mean()
        .sort_values(["target_recall", "mean_cost"])
    )
    best = agg.groupby("target_recall", as_index=False).first()
    default = agg[(agg["beta1"] == 0.90) & (agg["beta2"] == 0.99)].rename(columns={"mean_cost": "default_cost"})
    merged = best.merge(default[["target_recall", "default_cost"]], on="target_recall", how="left")
    merged["default_vs_best_pct"] = 100.0 * (merged["default_cost"] - merged["mean_cost"]) / merged["mean_cost"]

    lines = ["# Adam-patience beta sensitivity matrix", ""]
    lines.append("L2 HNSW high-envelope traces; Lambda is swept for each beta pair, then the lowest-cost point meeting each Recall@10 target is selected.")
    lines.append("")
    lines.append("## Best aggregate beta pair by target")
    lines.append("")
    lines.append("| Recall@10 | best beta1 | best beta2 | mean cost | default (0.9,0.99) cost | default gap |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in merged.itertuples(index=False):
        lines.append(
            f"| {row.target_recall:.2f} | {row.beta1:g} | {row.beta2:g} | "
            f"{row.mean_cost:.1f} | {row.default_cost:.1f} | {row.default_vs_best_pct:+.2f}% |"
        )
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append("- `beta_sensitivity_curves.csv`: full Lambda sweep curves.")
    lines.append("- `beta_sensitivity_target_costs.csv`: selected cost for each dataset, target, and beta pair.")
    lines.append("- `*_target_cost_lines.png/pdf`: per-dataset cost-vs-target lines.")
    lines.append("- `aggregate_heatmap_recall_*.png/pdf`: aggregate 3x3 heatmaps by target.")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        (label, stem, beta1, beta2)
        for label, stem in DATASETS
        for beta1 in BETA1_VALUES
        for beta2 in BETA2_VALUES
    ]
    workers = min(len(tasks), os.cpu_count() or 1)
    print(f"Running {len(tasks)} tasks with {workers} workers", flush=True)
    with Pool(processes=workers) as pool:
        curves = pool.map(run_task, tasks)
    curve_df = pd.concat(curves, ignore_index=True)
    summary = summarize_targets(curve_df)
    curve_df.to_csv(OUT_DIR / "beta_sensitivity_curves.csv", index=False)
    summary.to_csv(OUT_DIR / "beta_sensitivity_target_costs.csv", index=False)
    plot_target_lines(summary)
    plot_heatmaps(summary)
    write_markdown(summary)
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
