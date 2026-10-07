from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(image_dir: Path, max_images: int | None) -> list[Path]:
    """Collect image files from a directory."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(
            f"No supported images found in: {image_dir}"
        )

    if max_images is not None:
        images = images[:max_images]

    return images


def benchmark_image(
    model: YOLO,
    image_path: Path,
    iterations: int,
) -> dict[str, float | int | str]:
    """Benchmark YOLO inference on one image."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    latencies_ms: list[float] = []

    for _ in range(iterations):
        start = time.perf_counter()

        model.predict(
            source=image,
            imgsz=640,
            device="cpu",
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    mean_latency = statistics.mean(latencies_ms)
    median_latency = statistics.median(latencies_ms)

    return {
        "image": image_path.name,
        "width": width,
        "height": height,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": median_latency,
        "min_latency_ms": min(latencies_ms),
        "max_latency_ms": max(latencies_ms),
        "std_latency_ms": (
            statistics.stdev(latencies_ms)
            if len(latencies_ms) > 1
            else 0.0
        ),
        "fps": 1000.0 / mean_latency,
        "iterations": iterations,
    }


def save_results(
    results: list[dict[str, float | int | str]],
    output_path: Path,
) -> None:
    """Save benchmark results to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    fieldnames = list(results[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(
    results: list[dict[str, float | int | str]],
) -> None:
    """Print aggregate benchmark statistics."""

    mean_latencies = [
        float(result["mean_latency_ms"])
        for result in results
    ]

    fps_values = [
        float(result["fps"])
        for result in results
    ]

    print()
    print("=" * 70)
    print("LARA MULTI-IMAGE CPU BASELINE")
    print("=" * 70)

    print(f"Images tested:       {len(results)}")
    print(
        f"Mean latency:        "
        f"{statistics.mean(mean_latencies):.2f} ms"
    )
    print(
        f"Median image latency:"
        f" {statistics.median(mean_latencies):.2f} ms"
    )
    print(
        f"Best image latency:  "
        f"{min(mean_latencies):.2f} ms"
    )
    print(
        f"Worst image latency: "
        f"{max(mean_latencies):.2f} ms"
    )
    print(
        f"Mean throughput:     "
        f"{statistics.mean(fps_values):.2f} FPS"
    )

    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LARA's YOLOv8n CPU inference."
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolov8n.pt"),
        help="Path to YOLO model.",
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing benchmark images.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Inference iterations per image.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up inference runs.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to benchmark.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/baseline_multimage.csv"),
        help="CSV output path.",
    )

    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than 0")

    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")

    print("=" * 70)
    print("LARA CPU BASELINE BENCHMARK")
    print("=" * 70)

    print(f"Model:          {args.model}")
    print(f"Image directory:{args.image_dir}")
    print(f"PyTorch:        {torch.__version__}")
    print(f"CPU threads:    {torch.get_num_threads()}")
    print("Device:         CPU")

    images = collect_images(
        image_dir=args.image_dir,
        max_images=args.max_images,
    )

    print(f"Images found:   {len(images)}")

    print()
    print(f"Loading model: {args.model}")
    model = YOLO(str(args.model))
    print("Model loaded.")

    if args.warmup > 0:
        print()
        print(f"Warm-up: {args.warmup} runs...")

        warmup_image = cv2.imread(str(images[0]))

        if warmup_image is None:
            raise ValueError(
                f"Could not read warm-up image: {images[0]}"
            )

        for _ in range(args.warmup):
            model.predict(
                source=warmup_image,
                imgsz=640,
                device="cpu",
                verbose=False,
            )

        print("Warm-up complete.")

    print()
    print(
        f"Benchmarking {len(images)} images "
        f"x {args.iterations} iterations..."
    )

    results: list[dict[str, float | int | str]] = []

    for index, image_path in enumerate(images, start=1):
        print(
            f"[{index:>3}/{len(images)}] "
            f"{image_path.name}"
        )

        result = benchmark_image(
            model=model,
            image_path=image_path,
            iterations=args.iterations,
        )

        results.append(result)

    save_results(
        results=results,
        output_path=args.output,
    )

    print_summary(results)

    print()
    print(f"Detailed results saved to:")
    print(args.output)


if __name__ == "__main__":
    main()