from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import requests


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)

    if not values:
        return 0.0

    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values))

    if lower == upper:
        return values[lower]

    weight = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * weight
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/detect",
    )

    parser.add_argument(
        "--api-key",
        required=True,
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=Path(
            "research/test_images/test.jpg"
        ),
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/results/"
            "api_decomposition.csv"
        ),
    )

    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(
            f"Image not found: {args.image}"
        )

    session = requests.Session()

    headers = {
        "x-api-key": args.api_key,
    }

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    print("=" * 70)
    print("LARA API LATENCY DECOMPOSITION")
    print("=" * 70)

    print(
        f"Endpoint:     {args.url}"
    )

    print(
        f"Warm-up:      {args.warmup}"
    )

    print(
        f"Measured:     {args.requests}"
    )

    print()

    print("Warm-up...")

    for _ in range(args.warmup):

        with args.image.open("rb") as file:

            response = session.post(
                args.url,
                headers=headers,
                files={
                    "file": (
                        args.image.name,
                        file,
                        "image/jpeg",
                    )
                },
                timeout=120,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Warm-up failed: "
                f"{response.status_code} "
                f"{response.text}"
            )

    print("Warm-up complete.")
    print()

    # ---------------------------------------------------------
    # Measurements
    # ---------------------------------------------------------

    records: list[dict[str, float]] = []

    print(
        f"Running {args.requests} "
        "measured requests..."
    )

    for request_number in range(
        1,
        args.requests + 1,
    ):

        with args.image.open("rb") as file:

            response = session.post(
                args.url,
                headers=headers,
                files={
                    "file": (
                        args.image.name,
                        file,
                        "image/jpeg",
                    )
                },
                timeout=120,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Request {request_number} failed: "
                f"{response.status_code} "
                f"{response.text}"
            )

        payload = response.json()

        benchmark = payload.get(
            "benchmark"
        )

        if not benchmark:
            raise RuntimeError(
                "Response does not contain "
                "'benchmark' timings."
            )

        record = {
            "request": request_number,
            "auth_ms": float(
                benchmark["auth_ms"]
            ),
            "usage_check_ms": float(
                benchmark["usage_check_ms"]
            ),
            "upload_read_ms": float(
                benchmark["upload_read_ms"]
            ),
            "image_decode_ms": float(
                benchmark["image_decode_ms"]
            ),
            "inference_ms": float(
                benchmark["inference_ms"]
            ),
            "postprocess_ms": float(
                benchmark["postprocess_ms"]
            ),
            "database_log_ms": float(
                benchmark["database_log_ms"]
            ),
            "response_build_ms": float(
                benchmark["response_build_ms"]
            ),
            "total_server_ms": float(
                benchmark["total_server_ms"]
            ),
        }

        records.append(record)

        print(
            f"[{request_number:02d}/{args.requests}] "
            f"total={record['total_server_ms']:.2f} ms | "
            f"inference={record['inference_ms']:.2f} ms | "
            f"auth={record['auth_ms']:.2f} ms"
        )

    session.close()

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    fields = [
        "auth_ms",
        "usage_check_ms",
        "upload_read_ms",
        "image_decode_ms",
        "inference_ms",
        "postprocess_ms",
        "database_log_ms",
        "response_build_ms",
        "total_server_ms",
    ]

    print()
    print("=" * 100)
    print("LATENCY DECOMPOSITION RESULTS")
    print("=" * 100)

    print(
        f"{'Stage':<22}"
        f"{'Mean':>12}"
        f"{'Median':>12}"
        f"{'P95':>12}"
        f"{'P99':>12}"
        f"{'Std':>12}"
    )

    print("-" * 100)

    for field in fields:

        values = [
            record[field]
            for record in records
        ]

        print(
            f"{field:<22}"
            f"{statistics.mean(values):>12.2f}"
            f"{statistics.median(values):>12.2f}"
            f"{percentile(values, 0.95):>12.2f}"
            f"{percentile(values, 0.99):>12.2f}"
            f"{statistics.stdev(values):>12.2f}"
        )

    print("-" * 100)

    total_values = [
        record["total_server_ms"]
        for record in records
    ]

    print(
        f"Mean server latency: "
        f"{statistics.mean(total_values):.2f} ms"
    )

    print(
        f"Median server latency: "
        f"{statistics.median(total_values):.2f} ms"
    )

    print(
        f"P95 server latency: "
        f"{percentile(total_values, 0.95):.2f} ms"
    )

    print(
        f"P99 server latency: "
        f"{percentile(total_values, 0.99):.2f} ms"
    )

    # ---------------------------------------------------------
    # Save raw data
    # ---------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "request",
                *fields,
            ],
        )

        writer.writeheader()
        writer.writerows(records)

    print()
    print(
        f"Raw results saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()