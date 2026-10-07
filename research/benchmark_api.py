from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from statistics import mean, median, stdev

import requests


def benchmark_api(
    url: str,
    api_key: str,
    images: list[Path],
    warmup: int,
    iterations: int,
) -> list[float]:

    session = requests.Session()

    headers = {
        "x-api-key": api_key,
    }

    print("=" * 70)
    print("LARA END-TO-END API BENCHMARK")
    print("=" * 70)

    print(f"Endpoint:       {url}")
    print(f"Images:         {len(images)}")
    print(f"Warm-up:        {warmup}")
    print(f"Iterations:     {iterations}")
    print()

    print("Warm-up...")

    for i in range(warmup):

        image_path = images[
            i % len(images)
        ]

        with image_path.open("rb") as file:

            response = session.post(
                url,
                headers=headers,
                files={
                    "file": (
                        image_path.name,
                        file,
                        "image/jpeg",
                    )
                },
                timeout=120,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Warm-up request failed: "
                f"{response.status_code} "
                f"{response.text}"
            )

    print("Warm-up complete.")
    print()

    latencies: list[float] = []

    successful = 0
    failed = 0

    total_requests = (
        len(images) * iterations
    )

    print(
        f"Benchmarking: "
        f"{len(images)} images × "
        f"{iterations} iterations = "
        f"{total_requests} requests..."
    )

    for iteration in range(iterations):

        for image_path in images:

            with image_path.open("rb") as file:

                start = time.perf_counter()

                response = session.post(
                    url,
                    headers=headers,
                    files={
                        "file": (
                            image_path.name,
                            file,
                            "image/jpeg",
                        )
                    },
                    timeout=120,
                )

                elapsed_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

            if response.status_code == 200:
                successful += 1
                latencies.append(
                    elapsed_ms
                )
            else:
                failed += 1

                print(
                    f"\nRequest failed: "
                    f"{response.status_code}"
                )

    session.close()

    if not latencies:
        raise RuntimeError(
            "No successful requests."
        )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    mean_latency = mean(latencies)
    median_latency = median(latencies)

    std_latency = (
        stdev(latencies)
        if len(latencies) > 1
        else 0.0
    )

    min_latency = min(latencies)
    max_latency = max(latencies)

    throughput = (
        1000.0 / mean_latency
    )

    success_rate = (
        successful
        / total_requests
        * 100.0
    )

    print(
        f"Successful requests: "
        f"{successful}"
    )

    print(
        f"Failed requests:     "
        f"{failed}"
    )

    print(
        f"Success rate:        "
        f"{success_rate:.2f}%"
    )

    print(
        f"Mean latency:         "
        f"{mean_latency:.2f} ms"
    )

    print(
        f"Median latency:       "
        f"{median_latency:.2f} ms"
    )

    print(
        f"Std deviation:        "
        f"{std_latency:.2f} ms"
    )

    print(
        f"Minimum latency:      "
        f"{min_latency:.2f} ms"
    )

    print(
        f"Maximum latency:      "
        f"{max_latency:.2f} ms"
    )

    print(
        f"Throughput:           "
        f"{throughput:.2f} req/s"
    )

    return latencies


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "End-to-end LARA API benchmark."
        )
    )

    parser.add_argument(
        "--url",
        type=str,
        default=(
            "http://127.0.0.1:8000/detect"
        ),
    )

    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(
            "datasets/"
            "coco-val2017-mini500/"
            "images/val2017"
        ),
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/results/"
            "api_benchmark_val500.csv"
        ),
    )

    args = parser.parse_args()

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    images = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in extensions
    )

    if not images:
        raise FileNotFoundError(
            f"No images found in "
            f"{args.image_dir}"
        )

    images = images[
        :args.limit
    ]

    print(
        f"Selected images: "
        f"{len(images)}"
    )

    latencies = benchmark_api(
        url=args.url,
        api_key=args.api_key,
        images=images,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["request", "latency_ms"]
        )

        for index, latency in enumerate(
            latencies,
            start=1,
        ):

            writer.writerow(
                [index, latency]
            )

    print()
    print(
        f"Raw results saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()