from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


BASELINE = Path(
    "research/results/api_decomposition_baseline_final50.csv"
)

DEFERRED = Path(
    "research/results/api_decomposition_deferred_50_v2.csv"
)

OUTPUT = Path(
    "research/results/experiment_5b_final_analysis.json"
)


METRICS = [
    "total_server_ms",
    "auth_ms",
    "usage_check_ms",
    "upload_read_ms",
    "image_decode_ms",
    "inference_ms",
    "postprocess_ms",
    "database_log_ms",
    "response_build_ms",
]


def percentile(values: pd.Series, q: float) -> float:
    return float(np.percentile(values.to_numpy(dtype=float), q))


def summarize(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}

    for metric in METRICS:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()

        result[metric] = {
            "n": int(values.size),
            "mean_ms": float(values.mean()),
            "median_ms": float(values.median()),
            "p95_ms": percentile(values, 95),
            "p99_ms": percentile(values, 99),
            "std_ms": float(values.std(ddof=1)),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
        }

    return result


def reduction_percent(
    baseline: float,
    deferred: float,
) -> float:
    if baseline == 0:
        return math.nan

    return ((baseline - deferred) / baseline) * 100.0


def main() -> None:
    if not BASELINE.exists():
        raise FileNotFoundError(
            f"Baseline dataset not found: {BASELINE}"
        )

    if not DEFERRED.exists():
        raise FileNotFoundError(
            f"Deferred dataset not found: {DEFERRED}"
        )

    baseline = pd.read_csv(BASELINE)
    deferred = pd.read_csv(DEFERRED)

    if len(baseline) != 50:
        raise ValueError(
            f"Expected 50 baseline requests, got {len(baseline)}"
        )

    if len(deferred) != 50:
        raise ValueError(
            f"Expected 50 deferred requests, got {len(deferred)}"
        )

    required = set(METRICS)

    for name, df in [
        ("baseline", baseline),
        ("deferred", deferred),
    ]:
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} dataset is missing columns: {sorted(missing)}"
            )

    baseline_stats = summarize(baseline)
    deferred_stats = summarize(deferred)

    reductions: dict[str, float] = {}

    for metric in METRICS:
        reductions[metric] = reduction_percent(
            baseline_stats[metric]["mean_ms"],
            deferred_stats[metric]["mean_ms"],
        )

    result = {
        "experiment": "5B",
        "description": (
            "Synchronous database logging versus deferred "
            "database logging"
        ),
        "sample_size": {
            "baseline": len(baseline),
            "deferred": len(deferred),
        },
        "datasets": {
            "baseline": str(BASELINE),
            "deferred": str(DEFERRED),
        },
        "baseline": baseline_stats,
        "deferred": deferred_stats,
        "mean_reduction_percent": reductions,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("LARA EXPERIMENT 5B — FINAL ANALYSIS")
    print("=" * 78)

    print(f"Baseline requests : {len(baseline)}")
    print(f"Deferred requests : {len(deferred)}")
    print()

    print(
        f"{'Metric':<24}"
        f"{'Baseline':>14}"
        f"{'Deferred':>14}"
        f"{'Reduction':>14}"
    )

    print("-" * 66)

    for metric in [
        "total_server_ms",
        "inference_ms",
        "auth_ms",
        "usage_check_ms",
        "database_log_ms",
    ]:
        b = baseline_stats[metric]["mean_ms"]
        d = deferred_stats[metric]["mean_ms"]
        r = reductions[metric]

        print(
            f"{metric:<24}"
            f"{b:>14.2f}"
            f"{d:>14.2f}"
            f"{r:>13.2f}%"
        )

    print()
    print("=" * 78)
    print("TOTAL SERVER LATENCY")
    print("=" * 78)

    for label, key in [
        ("Mean", "mean_ms"),
        ("Median", "median_ms"),
        ("P95", "p95_ms"),
        ("P99", "p99_ms"),
        ("Std. deviation", "std_ms"),
    ]:
        b = baseline_stats["total_server_ms"][key]
        d = deferred_stats["total_server_ms"][key]

        r = reduction_percent(b, d)

        print(
            f"{label:<18}"
            f"{b:>12.2f} ms"
            f"{d:>12.2f} ms"
            f"{r:>12.2f}%"
        )

    print()
    print(f"Analysis saved to: {OUTPUT}")


if __name__ == "__main__":
    main()