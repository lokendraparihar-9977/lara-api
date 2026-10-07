from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from openvino import Core


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "research" / "evaluation" / "coco_val500" / "images"

FP32_MODEL = ROOT / "yolov8n_openvino_model" / "yolov8n.xml"
INT8_MODEL = ROOT / "yolov8n_int8_openvino_model" / "yolov8n.xml"

OUTPUT = ROOT / "research" / "results" / "e3_multimage_fp32_vs_int8_latency.json"

INPUT_SIZE = 640
WARMUP = 10

# Set to None to benchmark all images.
MAX_IMAGES: int | None = 500


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(latencies: list[float]) -> dict[str, float]:
    mean_ms = statistics.fmean(latencies)

    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(latencies),
        "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p90_ms": percentile(latencies, 90),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "fps": 1000.0 / mean_ms,
        "samples": len(latencies),
    }


def load_images() -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not images:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")

    if MAX_IMAGES is not None:
        images = images[:MAX_IMAGES]

    return images


def preprocess(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")

    image = cv2.resize(
        image,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = image.transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    tensor /= 255.0

    return tensor[None, ...]


def create_compiled_model(
    core: Core,
    model_path: Path,
):
    model = core.read_model(str(model_path))
    return core.compile_model(
        model,
        "CPU",
        config={"PERFORMANCE_HINT": "LATENCY"},
    )


def benchmark_model(
    compiled_model: Any,
    images: list[Path],
    name: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:

    infer_request = compiled_model.create_infer_request()
    input_layer = compiled_model.input(0)

    tensors = []

    print()
    print(f"Preparing {name} inputs...")

    for path in images:
        tensor = preprocess(path)
        tensors.append((path, tensor))

    print(f"{name}: {len(tensors)} images loaded")

    # ------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------

    warmup_tensor = tensors[0][1]

    print(f"{name}: {WARMUP} warm-up iterations...")

    for _ in range(WARMUP):
        infer_request.infer({input_layer: warmup_tensor})

    # ------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------

    latencies: list[float] = []
    records: list[dict[str, Any]] = []

    print(f"{name}: measuring {len(tensors)} images...")

    for index, (path, tensor) in enumerate(tensors, start=1):

        start = time.perf_counter()

        infer_request.infer({
            input_layer: tensor
        })

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        latencies.append(elapsed_ms)

        records.append({
            "image": str(path.relative_to(ROOT)),
            "latency_ms": elapsed_ms,
        })

        if index % 50 == 0 or index == len(tensors):
            print(
                f"\r{name}: "
                f"{index}/{len(tensors)} "
                f"| latest={elapsed_ms:.3f} ms",
                end="",
                flush=True,
            )

    print()

    return summarize(latencies), records


def main() -> None:

    print("=" * 78)
    print("LARA E3b — MULTI-IMAGE CONTROLLED FP32 vs INT8")
    print("OPENVINO CPU LATENCY BENCHMARK")
    print("=" * 78)

    print(f"Image directory : {IMAGE_DIR}")
    print(f"Input size      : {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Warm-up         : {WARMUP}")
    print(f"Max images      : {MAX_IMAGES}")
    print("Device          : CPU")
    print("Performance     : LATENCY")
    print()

    images = load_images()

    print(f"Images selected : {len(images)}")

    core = Core()

    print()
    print("Loading FP32 model...")
    fp32 = create_compiled_model(core, FP32_MODEL)

    print("Loading INT8 model...")
    int8 = create_compiled_model(core, INT8_MODEL)

    fp32_summary, fp32_records = benchmark_model(
        fp32,
        images,
        "FP32",
    )

    int8_summary, int8_records = benchmark_model(
        int8,
        images,
        "INT8",
    )

    # ------------------------------------------------------------
    # Comparative statistics
    # ------------------------------------------------------------

    mean_reduction = (
        (fp32_summary["mean_ms"] - int8_summary["mean_ms"])
        / fp32_summary["mean_ms"]
    ) * 100.0

    p95_reduction = (
        (fp32_summary["p95_ms"] - int8_summary["p95_ms"])
        / fp32_summary["p95_ms"]
    ) * 100.0

    p99_reduction = (
        (fp32_summary["p99_ms"] - int8_summary["p99_ms"])
        / fp32_summary["p99_ms"]
    ) * 100.0

    speedup = (
        fp32_summary["mean_ms"]
        / int8_summary["mean_ms"]
    )

    throughput_gain = (
        (int8_summary["fps"] - fp32_summary["fps"])
        / fp32_summary["fps"]
    ) * 100.0

    results = {
        "experiment": "E3b",
        "description": (
            "Multi-image controlled FP32 vs INT8 OpenVINO CPU "
            "inference latency benchmark"
        ),
        "configuration": {
            "image_count": len(images),
            "input_size": [1, 3, INPUT_SIZE, INPUT_SIZE],
            "batch_size": 1,
            "device": "CPU",
            "performance_hint": "LATENCY",
            "warmup_iterations": WARMUP,
        },
        "fp32": {
            "model": str(FP32_MODEL.relative_to(ROOT)),
            "summary": fp32_summary,
            "per_image": fp32_records,
        },
        "int8": {
            "model": str(INT8_MODEL.relative_to(ROOT)),
            "summary": int8_summary,
            "per_image": int8_records,
        },
        "comparison": {
            "mean_latency_reduction_percent": mean_reduction,
            "p95_latency_reduction_percent": p95_reduction,
            "p99_latency_reduction_percent": p99_reduction,
            "mean_speedup": speedup,
            "throughput_gain_percent": throughput_gain,
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)

    print(
        f"FP32 mean latency : "
        f"{fp32_summary['mean_ms']:.3f} ms"
    )

    print(
        f"INT8 mean latency : "
        f"{int8_summary['mean_ms']:.3f} ms"
    )

    print(
        f"Mean reduction    : "
        f"{mean_reduction:.2f}%"
    )

    print(
        f"Mean speedup      : "
        f"{speedup:.3f}x"
    )

    print()

    print(
        f"FP32 median       : "
        f"{fp32_summary['median_ms']:.3f} ms"
    )

    print(
        f"INT8 median       : "
        f"{int8_summary['median_ms']:.3f} ms"
    )

    print()

    print(
        f"FP32 P95          : "
        f"{fp32_summary['p95_ms']:.3f} ms"
    )

    print(
        f"INT8 P95          : "
        f"{int8_summary['p95_ms']:.3f} ms"
    )

    print(
        f"P95 reduction     : "
        f"{p95_reduction:.2f}%"
    )

    print()

    print(
        f"FP32 P99          : "
        f"{fp32_summary['p99_ms']:.3f} ms"
    )

    print(
        f"INT8 P99          : "
        f"{int8_summary['p99_ms']:.3f} ms"
    )

    print(
        f"P99 reduction     : "
        f"{p99_reduction:.2f}%"
    )

    print()

    print(
        f"FP32 throughput   : "
        f"{fp32_summary['fps']:.3f} FPS"
    )

    print(
        f"INT8 throughput   : "
        f"{int8_summary['fps']:.3f} FPS"
    )

    print(
        f"Throughput gain   : "
        f"{throughput_gain:.2f}%"
    )

    print()
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()