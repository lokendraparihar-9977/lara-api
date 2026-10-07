"""
LARA E5 — CONTROLLED BATCH-SIZE SCALING BENCHMARK

FP32 vs INT8 OpenVINO CPU inference.

Purpose:
    Measure how batch size affects:
        - Total batch latency
        - Per-image latency
        - Throughput
        - P95 batch latency
        - P99 batch latency
        - FP32 vs INT8 speedup

Experimental controls:
    - Same 500-image COCO validation subset
    - Same 640x640 input resolution
    - Same OpenVINO CPU device
    - 10 warm-up iterations per configuration
    - Batch sizes: 1, 2, 4, 8
    - Same preprocessing for both models
    - No asynchronous inference
    - No concurrent requests
    - LATENCY performance hint

Output:
    research/results/e5_batch_scaling_fp32_vs_int8.json
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openvino import Core


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

IMAGE_DIR = BASE_DIR / "research" / "evaluation" / "coco_val500" / "images"

FP32_MODEL = BASE_DIR / "yolov8n_openvino_model" / "yolov8n.xml"
INT8_MODEL = BASE_DIR / "yolov8n_int8_openvino_model" / "yolov8n.xml"

RESULTS_DIR = BASE_DIR / "research" / "results"
OUTPUT_FILE = RESULTS_DIR / "e5_batch_scaling_fp32_vs_int8.json"

IMAGE_SIZE = 640
MAX_IMAGES = 500
WARMUP_ITERATIONS = 10

BATCH_SIZES = [1, 2, 4, 8]

DEVICE = "CPU"


# ============================================================================
# UTILITIES
# ============================================================================

def percentile(values: list[float], p: float) -> float:
    """Calculate percentile using NumPy's linear interpolation."""
    if not values:
        raise ValueError("Cannot calculate percentile of an empty list.")

    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def discover_images(image_dir: Path, max_images: int) -> list[Path]:
    """Discover deterministic image list."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    images = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    return images[:max_images]


def preprocess_image(path: Path, size: int) -> np.ndarray:
    """
    Preprocess image into NCHW float32 tensor.

    Pipeline:
        BGR uint8
        -> RGB
        -> resize
        -> float32
        -> [0, 1]
        -> CHW
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_LINEAR,
    )

    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))

    return np.ascontiguousarray(image)


def load_model(
    core: Core,
    model_path: Path,
    batch_size: int,
) -> Any:
    """Load and compile an OpenVINO model for a specific static batch size."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = core.read_model(str(model_path))

    input_port = model.inputs[0]

    # Reshape static batch dimension.
    original_shape = input_port.partial_shape

    if original_shape.rank != 4:
        raise RuntimeError(
            f"Unexpected model input rank: {original_shape.rank}"
        )

    model.reshape(
        {
            input_port: [batch_size, 3, IMAGE_SIZE, IMAGE_SIZE]
        }
    )

    compiled_model = core.compile_model(
        model,
        DEVICE,
        {
            "PERFORMANCE_HINT": "LATENCY",
        },
    )

    return compiled_model


def run_batch(
    compiled_model: Any,
    batch: np.ndarray,
) -> None:
    """Execute one synchronous OpenVINO inference."""
    request = compiled_model.create_infer_request()
    request.infer({compiled_model.input(0): batch})


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_model(
    core: Core,
    model_path: Path,
    image_paths: list[Path],
    batch_size: int,
    model_name: str,
) -> dict[str, Any]:

    print()
    print(f"{model_name}: loading batch={batch_size}...")

    compiled_model = load_model(
        core=core,
        model_path=model_path,
        batch_size=batch_size,
    )

    # ------------------------------------------------------------------------
    # Prepare all inputs.
    # ------------------------------------------------------------------------

    print(f"{model_name}: preparing inputs...")

    tensors: list[np.ndarray] = []

    for path in image_paths:
        tensors.append(
            preprocess_image(
                path,
                IMAGE_SIZE,
            )
        )

    print(
        f"{model_name}: {len(tensors)} images prepared"
    )

    # ------------------------------------------------------------------------
    # Create batches.
    #
    # The final incomplete batch is discarded deliberately.
    # This keeps every measured inference at the same batch size.
    # ------------------------------------------------------------------------

    usable_count = (len(tensors) // batch_size) * batch_size

    if usable_count == 0:
        raise RuntimeError(
            f"Not enough images for batch size {batch_size}"
        )

    tensors = tensors[:usable_count]

    batches = [
        np.stack(
            tensors[i : i + batch_size],
            axis=0,
        )
        for i in range(
            0,
            usable_count,
            batch_size,
        )
    ]

    print(
        f"{model_name}: "
        f"{len(batches)} batches × {batch_size} images"
    )

    # ------------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------------

    warmup_batch = batches[0]

    print(
        f"{model_name}: "
        f"{WARMUP_ITERATIONS} warm-up iterations..."
    )

    for _ in range(WARMUP_ITERATIONS):
        run_batch(
            compiled_model,
            warmup_batch,
        )

    # ------------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------------

    print(
        f"{model_name}: "
        f"measuring {len(batches)} batches..."
    )

    latencies_ms: list[float] = []

    benchmark_start = time.perf_counter()

    for index, batch in enumerate(batches, start=1):

        start = time.perf_counter()

        run_batch(
            compiled_model,
            batch,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        latencies_ms.append(elapsed_ms)

        if index == len(batches) or index % 50 == 0:
            print(
                f"{model_name}: "
                f"{index}/{len(batches)} | "
                f"latest={elapsed_ms:.3f} ms"
            )

    benchmark_elapsed_s = (
        time.perf_counter() - benchmark_start
    )

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    mean_batch_ms = statistics.mean(latencies_ms)
    median_batch_ms = statistics.median(latencies_ms)

    p95_batch_ms = percentile(
        latencies_ms,
        95,
    )

    p99_batch_ms = percentile(
        latencies_ms,
        99,
    )

    mean_per_image_ms = (
        mean_batch_ms / batch_size
    )

    median_per_image_ms = (
        median_batch_ms / batch_size
    )

    p95_per_image_ms = (
        p95_batch_ms / batch_size
    )

    p99_per_image_ms = (
        p99_batch_ms / batch_size
    )

    throughput_fps = (
        usable_count / benchmark_elapsed_s
    )

    batch_throughput = (
        len(batches) / benchmark_elapsed_s
    )

    return {
        "model": model_name,
        "model_path": str(model_path),
        "batch_size": batch_size,
        "images_measured": usable_count,
        "batches_measured": len(batches),
        "warmup_iterations": WARMUP_ITERATIONS,
        "input_size": IMAGE_SIZE,
        "device": DEVICE,
        "performance_hint": "LATENCY",

        "batch_latency_ms": {
            "mean": mean_batch_ms,
            "median": median_batch_ms,
            "p95": p95_batch_ms,
            "p99": p99_batch_ms,
        },

        "per_image_latency_ms": {
            "mean": mean_per_image_ms,
            "median": median_per_image_ms,
            "p95": p95_per_image_ms,
            "p99": p99_per_image_ms,
        },

        "throughput_fps": throughput_fps,
        "batch_throughput": batch_throughput,

        "benchmark_wall_time_s": benchmark_elapsed_s,
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main() -> None:

    print("=" * 78)
    print(
        "LARA E5 — BATCH-SIZE SCALING"
    )
    print(
        "FP32 vs INT8 OPENVINO CPU BENCHMARK"
    )
    print("=" * 78)

    print(
        f"Image directory : {IMAGE_DIR}"
    )

    print(
        f"Input size      : {IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Warm-up         : {WARMUP_ITERATIONS}"
    )

    print(
        f"Max images      : {MAX_IMAGES}"
    )

    print(
        f"Batch sizes     : {BATCH_SIZES}"
    )

    print(
        f"Device          : {DEVICE}"
    )

    print(
        f"Performance     : LATENCY"
    )

    # ------------------------------------------------------------------------
    # Discover images
    # ------------------------------------------------------------------------

    image_paths = discover_images(
        IMAGE_DIR,
        MAX_IMAGES,
    )

    print(
        f"Images selected : {len(image_paths)}"
    )

    # ------------------------------------------------------------------------
    # OpenVINO Core
    # ------------------------------------------------------------------------

    core = Core()

    # ------------------------------------------------------------------------
    # Benchmark all batch sizes
    # ------------------------------------------------------------------------

    all_results: list[dict[str, Any]] = []

    for batch_size in BATCH_SIZES:

        print()
        print("-" * 78)
        print(
            f"BATCH SIZE = {batch_size}"
        )
        print("-" * 78)

        fp32_result = benchmark_model(
            core=core,
            model_path=FP32_MODEL,
            image_paths=image_paths,
            batch_size=batch_size,
            model_name="FP32",
        )

        int8_result = benchmark_model(
            core=core,
            model_path=INT8_MODEL,
            image_paths=image_paths,
            batch_size=batch_size,
            model_name="INT8",
        )

        # --------------------------------------------------------------------
        # Relative comparison
        # --------------------------------------------------------------------

        fp32_mean = fp32_result[
            "per_image_latency_ms"
        ]["mean"]

        int8_mean = int8_result[
            "per_image_latency_ms"
        ]["mean"]

        fp32_p95 = fp32_result[
            "per_image_latency_ms"
        ]["p95"]

        int8_p95 = int8_result[
            "per_image_latency_ms"
        ]["p95"]

        fp32_fps = fp32_result[
            "throughput_fps"
        ]

        int8_fps = int8_result[
            "throughput_fps"
        ]

        mean_speedup = (
            fp32_mean / int8_mean
        )

        mean_reduction = (
            (fp32_mean - int8_mean)
            / fp32_mean
        ) * 100.0

        p95_reduction = (
            (fp32_p95 - int8_p95)
            / fp32_p95
        ) * 100.0

        throughput_gain = (
            (int8_fps - fp32_fps)
            / fp32_fps
        ) * 100.0

        comparison = {
            "batch_size": batch_size,

            "fp32": fp32_result,
            "int8": int8_result,

            "comparison": {
                "mean_per_image_latency_reduction_percent":
                    mean_reduction,

                "mean_per_image_speedup":
                    mean_speedup,

                "p95_per_image_latency_reduction_percent":
                    p95_reduction,

                "throughput_gain_percent":
                    throughput_gain,
            },
        }

        all_results.append(comparison)

        # --------------------------------------------------------------------
        # Console summary
        # --------------------------------------------------------------------

        print()
        print(
            f"BATCH {batch_size} RESULTS"
        )
        print("-" * 78)

        print(
            f"FP32 mean/image : "
            f"{fp32_mean:.3f} ms"
        )

        print(
            f"INT8 mean/image : "
            f"{int8_mean:.3f} ms"
        )

        print(
            f"Mean reduction  : "
            f"{mean_reduction:.2f}%"
        )

        print(
            f"Mean speedup    : "
            f"{mean_speedup:.3f}x"
        )

        print(
            f"FP32 P95/image  : "
            f"{fp32_p95:.3f} ms"
        )

        print(
            f"INT8 P95/image  : "
            f"{int8_p95:.3f} ms"
        )

        print(
            f"P95 reduction   : "
            f"{p95_reduction:.2f}%"
        )

        print(
            f"FP32 throughput : "
            f"{fp32_fps:.3f} FPS"
        )

        print(
            f"INT8 throughput : "
            f"{int8_fps:.3f} FPS"
        )

        print(
            f"Throughput gain : "
            f"{throughput_gain:.2f}%"
        )

    # =========================================================================
    # FINAL SUMMARY TABLE
    # =========================================================================

    print()
    print("=" * 78)
    print("E5 RESULTS — BATCH-SIZE SCALING")
    print("=" * 78)

    print(
        f"{'Batch':>6} | "
        f"{'FP32 ms/img':>13} | "
        f"{'INT8 ms/img':>13} | "
        f"{'FP32 FPS':>10} | "
        f"{'INT8 FPS':>10} | "
        f"{'Speedup':>9}"
    )

    print("-" * 78)

    for result in all_results:

        batch_size = result["batch_size"]

        fp32_mean = result[
            "fp32"
        ]["per_image_latency_ms"]["mean"]

        int8_mean = result[
            "int8"
        ]["per_image_latency_ms"]["mean"]

        fp32_fps = result[
            "fp32"
        ]["throughput_fps"]

        int8_fps = result[
            "int8"
        ]["throughput_fps"]

        speedup = result[
            "comparison"
        ]["mean_per_image_speedup"]

        print(
            f"{batch_size:>6} | "
            f"{fp32_mean:>13.3f} | "
            f"{int8_mean:>13.3f} | "
            f"{fp32_fps:>10.3f} | "
            f"{int8_fps:>10.3f} | "
            f"{speedup:>8.3f}x"
        )

    # =========================================================================
    # SAVE
    # =========================================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": "E5",
        "name": "Batch-size scaling: FP32 vs INT8 OpenVINO",
        "image_directory": str(IMAGE_DIR),
        "image_count_requested": MAX_IMAGES,
        "image_count_selected": len(image_paths),
        "input_size": IMAGE_SIZE,
        "warmup_iterations": WARMUP_ITERATIONS,
        "batch_sizes": BATCH_SIZES,
        "device": DEVICE,
        "performance_hint": "LATENCY",
        "fp32_model": str(FP32_MODEL),
        "int8_model": str(INT8_MODEL),
        "results": all_results,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()