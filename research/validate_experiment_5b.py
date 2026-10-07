from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASELINE = Path("research/results/api_decomposition_baseline_final50.csv")
DEFERRED = Path("research/results/api_decomposition_deferred_50_v2.csv")
OUTPUT = Path("research/results/experiment_5b_statistical_validation.json")


def bootstrap_mean_difference(
    baseline: np.ndarray,
    deferred: np.ndarray,
    *,
    iterations: int = 20_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap the difference in means:

        baseline_mean - deferred_mean

    Returns:
        observed_difference,
        lower_95_ci,
        upper_95_ci
    """
    rng = np.random.default_rng(seed)

    n_baseline = len(baseline)
    n_deferred = len(deferred)

    baseline_samples = rng.choice(
        baseline,
        size=(iterations, n_baseline),
        replace=True,
    )

    deferred_samples = rng.choice(
        deferred,
        size=(iterations, n_deferred),
        replace=True,
    )

    differences = (
        baseline_samples.mean(axis=1)
        - deferred_samples.mean(axis=1)
    )

    observed = float(baseline.mean() - deferred.mean())

    lower = float(np.percentile(differences, 2.5))
    upper = float(np.percentile(differences, 97.5))

    return observed, lower, upper


def cohens_d_independent(
    baseline: np.ndarray,
    deferred: np.ndarray,
) -> float:
    """Cohen's d using pooled standard deviation."""
    n1 = len(baseline)
    n2 = len(deferred)

    s1 = np.std(baseline, ddof=1)
    s2 = np.std(deferred, ddof=1)

    pooled = np.sqrt(
        (
            (n1 - 1) * s1**2
            + (n2 - 1) * s2**2
        )
        / (n1 + n2 - 2)
    )

    return float((baseline.mean() - deferred.mean()) / pooled)


def cliffs_delta(
    baseline: np.ndarray,
    deferred: np.ndarray,
) -> float:
    """
    Cliff's delta.

    Positive values indicate that baseline observations
    tend to be larger than deferred observations.
    """
    greater = 0
    less = 0

    for x in baseline:
        greater += np.sum(x > deferred)
        less += np.sum(x < deferred)

    return float(
        (greater - less)
        / (len(baseline) * len(deferred))
    )


def main() -> None:
    print("=" * 78)
    print("LARA EXPERIMENT 5B — STATISTICAL VALIDATION")
    print("=" * 78)

    if not BASELINE.exists():
        raise FileNotFoundError(f"Missing baseline CSV: {BASELINE}")

    if not DEFERRED.exists():
        raise FileNotFoundError(f"Missing deferred CSV: {DEFERRED}")

    baseline_df = pd.read_csv(BASELINE)
    deferred_df = pd.read_csv(DEFERRED)

    metric = "total_server_ms"

    if metric not in baseline_df.columns:
        raise ValueError(
            f"{metric} not found in baseline CSV"
        )

    if metric not in deferred_df.columns:
        raise ValueError(
            f"{metric} not found in deferred CSV"
        )

    baseline = baseline_df[metric].dropna().to_numpy(dtype=float)
    deferred = deferred_df[metric].dropna().to_numpy(dtype=float)

    print(f"Baseline observations : {len(baseline)}")
    print(f"Deferred observations : {len(deferred)}")
    print()

    if len(baseline) != 50 or len(deferred) != 50:
        raise ValueError(
            "Expected exactly 50 observations in each final dataset."
        )

    baseline_mean = float(np.mean(baseline))
    deferred_mean = float(np.mean(deferred))

    baseline_median = float(np.median(baseline))
    deferred_median = float(np.median(deferred))

    mean_reduction_pct = (
        (baseline_mean - deferred_mean)
        / baseline_mean
        * 100
    )

    median_reduction_pct = (
        (baseline_median - deferred_median)
        / baseline_median
        * 100
    )

    # Welch's independent two-sample t-test.
    welch = stats.ttest_ind(
        baseline,
        deferred,
        equal_var=False,
    )

    # Mann-Whitney U: non-parametric distribution comparison.
    mann_whitney = stats.mannwhitneyu(
        baseline,
        deferred,
        alternative="two-sided",
    )

    # Distributional comparison.
    ks = stats.ks_2samp(
        baseline,
        deferred,
        alternative="two-sided",
        method="auto",
    )

    # Effect sizes.
    d = cohens_d_independent(
        baseline,
        deferred,
    )

    delta = cliffs_delta(
        baseline,
        deferred,
    )

    # Bootstrap CI for mean difference.
    mean_difference, ci_lower, ci_upper = (
        bootstrap_mean_difference(
            baseline,
            deferred,
        )
    )

    baseline_std = float(np.std(baseline, ddof=1))
    deferred_std = float(np.std(deferred, ddof=1))

    result = {
        "experiment": "5B",
        "comparison": {
            "baseline_file": str(BASELINE),
            "deferred_file": str(DEFERRED),
            "baseline_n": len(baseline),
            "deferred_n": len(deferred),
            "metric": metric,
        },
        "descriptive_statistics": {
            "baseline": {
                "mean_ms": baseline_mean,
                "median_ms": baseline_median,
                "std_ms": baseline_std,
                "min_ms": float(np.min(baseline)),
                "max_ms": float(np.max(baseline)),
            },
            "deferred": {
                "mean_ms": deferred_mean,
                "median_ms": deferred_median,
                "std_ms": deferred_std,
                "min_ms": float(np.min(deferred)),
                "max_ms": float(np.max(deferred)),
            },
        },
        "latency_reduction": {
            "mean_difference_ms": mean_difference,
            "mean_reduction_percent": mean_reduction_pct,
            "median_reduction_percent": median_reduction_pct,
        },
        "bootstrap": {
            "mean_difference_ms": mean_difference,
            "ci_95_lower_ms": ci_lower,
            "ci_95_upper_ms": ci_upper,
            "iterations": 20_000,
            "seed": 42,
        },
        "statistical_tests": {
            "welch_t_test": {
                "statistic": float(welch.statistic),
                "p_value": float(welch.pvalue),
            },
            "mann_whitney_u": {
                "statistic": float(mann_whitney.statistic),
                "p_value": float(mann_whitney.pvalue),
            },
            "kolmogorov_smirnov": {
                "statistic": float(ks.statistic),
                "p_value": float(ks.pvalue),
            },
        },
        "effect_sizes": {
            "cohens_d": d,
            "cliffs_delta": delta,
        },
    }

    OUTPUT.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 78)

    print(
        f"Baseline mean     : {baseline_mean:.2f} ms"
    )
    print(
        f"Deferred mean     : {deferred_mean:.2f} ms"
    )
    print(
        f"Mean reduction    : {mean_reduction_pct:.2f}%"
    )
    print()

    print(
        f"Baseline median   : {baseline_median:.2f} ms"
    )
    print(
        f"Deferred median   : {deferred_median:.2f} ms"
    )
    print(
        f"Median reduction  : {median_reduction_pct:.2f}%"
    )
    print()

    print("=" * 78)
    print("95% BOOTSTRAP CI — MEAN LATENCY DIFFERENCE")
    print("=" * 78)

    print(
        f"Difference        : {mean_difference:.2f} ms"
    )
    print(
        f"95% CI            : "
        f"[{ci_lower:.2f}, {ci_upper:.2f}] ms"
    )
    print()

    print("=" * 78)
    print("STATISTICAL TESTS")
    print("=" * 78)

    print(
        f"Welch t-test      : "
        f"t={welch.statistic:.4f}, "
        f"p={welch.pvalue:.6g}"
    )

    print(
        f"Mann-Whitney U    : "
        f"U={mann_whitney.statistic:.2f}, "
        f"p={mann_whitney.pvalue:.6g}"
    )

    print(
        f"KS test            : "
        f"D={ks.statistic:.4f}, "
        f"p={ks.pvalue:.6g}"
    )

    print()

    print("=" * 78)
    print("EFFECT SIZES")
    print("=" * 78)

    print(f"Cohen's d         : {d:.4f}")
    print(f"Cliff's delta     : {delta:.4f}")
    print()

    print("=" * 78)
    print("OUTPUT")
    print("=" * 78)
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()