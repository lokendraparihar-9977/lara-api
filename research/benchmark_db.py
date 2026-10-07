from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from supabase import create_client


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    rank = (len(values) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(values))

    if lower == upper:
        return values[lower]

    fraction = rank - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def benchmark_operation(
    name: str,
    operation: Callable[[], None],
    warmup: int,
    iterations: int,
) -> list[float]:

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Warm-up:   {warmup}")
    print(f"Measured:  {iterations}")
    print()

    print("Warm-up...")

    for _ in range(warmup):
        operation()

    print("Warm-up complete.")
    print()

    latencies: list[float] = []

    for i in range(iterations):
        start = time.perf_counter()

        try:
            operation()
        except Exception as exc:
            print(f"[{i + 1:02d}/{iterations}] ERROR: {exc}")
            continue

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)

        print(
            f"[{i + 1:02d}/{iterations}] "
            f"{elapsed_ms:.2f} ms"
        )

    return latencies


def summarize(
    name: str,
    values: list[float],
) -> dict[str, float | str]:

    if not values:
        raise RuntimeError(f"No successful measurements for {name}")

    return {
        "operation": name,
        "samples": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "std_ms": statistics.stdev(values)
        if len(values) > 1
        else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description="LARA Supabase database latency benchmark."
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/results/db_latency_50.csv"
        ),
    )

    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("iterations must be greater than zero")

    if args.warmup < 0:
        raise ValueError("warmup cannot be negative")

    # --------------------------------------------------
    # Environment
    # --------------------------------------------------

    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    owner_api_key = os.getenv("OWNER_API_KEY")

    if not supabase_url:
        raise RuntimeError("SUPABASE_URL is missing from .env")

    if not supabase_key:
        raise RuntimeError("SUPABASE_KEY is missing from .env")

    if not owner_api_key:
        raise RuntimeError("OWNER_API_KEY is missing from .env")

    supabase = create_client(
        supabase_url,
        supabase_key,
    )

    print("=" * 70)
    print("LARA SUPABASE DATABASE LATENCY BENCHMARK")
    print("=" * 70)
    print(f"Warm-up runs: {args.warmup}")
    print(f"Measured runs: {args.iterations}")
    print()
    print(
        "Operations:"
    )
    print("  1. Authentication lookup")
    print("  2. Usage-count query")
    print("  3. Usage insertion")
    print("  4. Combined database pipeline")
    print()

    # --------------------------------------------------
    # Database operations matching main.py
    # --------------------------------------------------

    def get_user() -> None:
        result = (
            supabase
            .table("users")
            .select("*")
            .eq("api_key", owner_api_key)
            .execute()
        )

        if not result.data:
            raise RuntimeError(
                "OWNER_API_KEY did not resolve to a user"
            )

    def count_usage() -> None:
        result = (
            supabase
            .table("usage")
            .select("id")
            .eq("api_key", owner_api_key)
            .execute()
        )

        # Match LARA's implementation.
        _ = len(result.data)

    def log_usage() -> None:
        (
            supabase
            .table("usage")
            .insert(
                {
                    "api_key": owner_api_key,
                    "endpoint": "/research-db-benchmark",
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
            )
            .execute()
        )

    def combined_database_pipeline() -> None:

        user_result = (
            supabase
            .table("users")
            .select("*")
            .eq("api_key", owner_api_key)
            .execute()
        )

        if not user_result.data:
            raise RuntimeError(
                "OWNER_API_KEY did not resolve to a user"
            )

        usage_result = (
            supabase
            .table("usage")
            .select("id")
            .eq("api_key", owner_api_key)
            .execute()
        )

        _ = len(usage_result.data)

        (
            supabase
            .table("usage")
            .insert(
                {
                    "api_key": owner_api_key,
                    "endpoint": "/research-db-benchmark",
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
            )
            .execute()
        )

    # --------------------------------------------------
    # Run experiments
    # --------------------------------------------------

    authentication = benchmark_operation(
        "1. AUTHENTICATION LOOKUP",
        get_user,
        args.warmup,
        args.iterations,
    )

    usage_count = benchmark_operation(
        "2. USAGE-COUNT QUERY",
        count_usage,
        args.warmup,
        args.iterations,
    )

    usage_logging = benchmark_operation(
        "3. DATABASE LOGGING / INSERT",
        log_usage,
        args.warmup,
        args.iterations,
    )

    combined = benchmark_operation(
        "4. COMBINED DATABASE PIPELINE",
        combined_database_pipeline,
        args.warmup,
        args.iterations,
    )

    summaries = [
        summarize("authentication_lookup", authentication),
        summarize("usage_count_query", usage_count),
        summarize("database_logging_insert", usage_logging),
        summarize("combined_database_pipeline", combined),
    ]

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print()
    print("=" * 105)
    print("DATABASE LATENCY RESULTS")
    print("=" * 105)

    print(
        f"{'Operation':<30}"
        f"{'Mean':>12}"
        f"{'Median':>12}"
        f"{'P95':>12}"
        f"{'P99':>12}"
        f"{'Std':>12}"
    )

    print("-" * 105)

    for result in summaries:

        print(
            f"{str(result['operation']):<30}"
            f"{float(result['mean_ms']):>12.2f}"
            f"{float(result['median_ms']):>12.2f}"
            f"{float(result['p95_ms']):>12.2f}"
            f"{float(result['p99_ms']):>12.2f}"
            f"{float(result['std_ms']):>12.2f}"
        )

    print("=" * 105)

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "operation",
        "samples",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "p99_ms",
        "std_ms",
        "min_ms",
        "max_ms",
    ]

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summaries)

    print()
    print("Results saved to:")
    print(args.output)
    print()
    print(
        "NOTE: Database logging measurements inserted "
        f"{len(usage_logging)} benchmark records."
    )


if __name__ == "__main__":
    main()
