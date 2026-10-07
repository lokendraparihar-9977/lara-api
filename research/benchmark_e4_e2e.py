from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openvino import Core


ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "research" / "evaluation" / "coco_val500" / "images"

FP32_MODEL = ROOT / "yolov8n_openvino_model" / "yolov8n.xml"
INT8_MODEL = ROOT / "yolov8n_int8_openvino_model" / "yolov8n.xml"

OUTPUT = (
    ROOT
    / "research"
    / "results"
    / "e4_fp32_vs_int8_end_to_end.json"
)

INPUT_SIZE = 640
WARMUP = 10
MAX_IMAGES = 500

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def summary(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)

    return {
        "mean_ms": mean,
        "median_ms": statistics.median(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "fps": 1000.0 / mean,
        "samples": len(values),
    }


def load_images() -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    images = sorted(
        p
        for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not images:
        raise RuntimeError(f"No images found: {IMAGE_DIR}")

    return images[:MAX_IMAGES]


def preprocess(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    image = cv2.resize(
        image,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.transpose(2, 0, 1)
    image = np.ascontiguousarray(image, dtype=np.float32)

    image /= 255.0

    return image[None, ...]


def postprocess(
    output: np.ndarray,
    original_width: int,
    original_height: int,
) -> list[dict[str, Any]]:
    """
    YOLOv8 detection output processing.

    Expected output:
        [1, 84, N]

    4 box coordinates + 80 class scores.
    """

    predictions = np.squeeze(output)

    if predictions.ndim != 2:
        raise RuntimeError(
            f"Unexpected model output shape: {output.shape}"
        )

    # Convert [84, N] -> [N, 84]
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    mask = scores >= CONF_THRESHOLD

    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # cx, cy, w, h -> x1, y1, x2, y2
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    scale_x = original_width / INPUT_SIZE
    scale_y = original_height / INPUT_SIZE

    x1 *= scale_x
    x2 *= scale_x
    y1 *= scale_y
    y2 *= scale_y

    result: list[dict[str, Any]] = []

    # Class-aware greedy NMS.
    for class_id in np.unique(class_ids):

        indices = np.where(class_ids == class_id)[0]

        order = indices[
            np.argsort(scores[indices])[::-1]
        ]

        while len(order) > 0:

            current = order[0]

            result.append(
                {
                    "class_id": int(class_ids[current]),
                    "confidence": float(scores[current]),
                    "bbox": [
                        float(x1[current]),
                        float(y1[current]),
                        float(x2[current]),
                        float(y2[current]),
                    ],
                }
            )

            if len(order) == 1:
                break

            remaining = order[1:]

            xx1 = np.maximum(x1[current], x1[remaining])
            yy1 = np.maximum(y1[current], y1[remaining])
            xx2 = np.minimum(x2[current], x2[remaining])
            yy2 = np.minimum(y2[current], y2[remaining])

            intersection = np.maximum(
                0,
                xx2 - xx1,
            ) * np.maximum(
                0,
                yy2 - yy1,
            )

            area_current = (
                (x2[current] - x1[current])
                * (y2[current] - y1[current])
            )

            area_remaining = (
                (x2[remaining] - x1[remaining])
                * (y2[remaining] - y1[remaining])
            )

            union = (
                area_current
                + area_remaining
                - intersection
            )

            iou = intersection / np.maximum(union, 1e-12)

            order = remaining[iou < IOU_THRESHOLD]

    return result


def compile_model(core: Core, path: Path):
    model = core.read_model(str(path))

    return core.compile_model(
        model,
        "CPU",
        config={"PERFORMANCE_HINT": "LATENCY"},
    )


def benchmark(
    compiled_model: Any,
    images: list[Path],
    name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:

    request = compiled_model.create_infer_request()
    input_layer = compiled_model.input(0)

    # ------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------

    warmup_tensor = preprocess(images[0])

    print(f"{name}: {WARMUP} warm-up iterations...")

    for _ in range(WARMUP):
        request.infer({input_layer: warmup_tensor})

    # ------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------

    total_latencies: list[float] = []
    preprocess_times: list[float] = []
    inference_times: list[float] = []
    postprocess_times: list[float] = []

    records: list[dict[str, Any]] = []

    print(f"{name}: benchmarking {len(images)} images...")

    for index, image_path in enumerate(images, start=1):

        # --------------------------------------------------------
        # Preprocessing
        # --------------------------------------------------------

        start = time.perf_counter()

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Could not read {image_path}"
            )

        original_height, original_width = image.shape[:2]

        image = cv2.resize(
            image,
            (INPUT_SIZE, INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        tensor = image.transpose(2, 0, 1)
        tensor = np.ascontiguousarray(
            tensor,
            dtype=np.float32,
        )

        tensor /= 255.0
        tensor = tensor[None, ...]

        preprocess_ms = (
            time.perf_counter() - start
        ) * 1000.0

        # --------------------------------------------------------
        # Inference
        # --------------------------------------------------------

        start = time.perf_counter()

        request.infer({
            input_layer: tensor
        })

        output = request.get_output_tensor(0).data.copy()

        inference_ms = (
            time.perf_counter() - start
        ) * 1000.0

        # --------------------------------------------------------
        # Post-processing
        # --------------------------------------------------------

        start = time.perf_counter()

        detections = postprocess(
            output,
            original_width,
            original_height,
        )

        postprocess_ms = (
            time.perf_counter() - start
        ) * 1000.0

        total_ms = (
            preprocess_ms
            + inference_ms
            + postprocess_ms
        )

        preprocess_times.append(preprocess_ms)
        inference_times.append(inference_ms)
        postprocess_times.append(postprocess_ms)
        total_latencies.append(total_ms)

        records.append(
            {
                "image": str(
                    image_path.relative_to(ROOT)
                ),
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "total_ms": total_ms,
                "detections": len(detections),
            }
        )

        if index % 50 == 0 or index == len(images):
            print(
                f"\r{name}: "
                f"{index}/{len(images)} "
                f"| total={total_ms:.3f} ms",
                end="",
                flush=True,
            )

    print()

    return (
        {
            "total": summary(total_latencies),
            "preprocess": summary(preprocess_times),
            "inference": summary(inference_times),
            "postprocess": summary(postprocess_times),
        },
        records,
    )


def main() -> None:

    print("=" * 78)
    print("LARA E4 — END-TO-END FP32 vs INT8 OPENVINO BENCHMARK")
    print("=" * 78)

    print(f"Images       : {IMAGE_DIR}")
    print(f"Image count  : {MAX_IMAGES}")
    print(f"Input size   : {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Warm-up      : {WARMUP}")
    print(f"Device       : CPU")
    print(f"Confidence   : {CONF_THRESHOLD}")
    print(f"IoU threshold: {IOU_THRESHOLD}")

    images = load_images()

    print(f"\nImages selected: {len(images)}")

    core = Core()

    print("\nLoading FP32...")
    fp32 = compile_model(core, FP32_MODEL)

    print("Loading INT8...")
    int8 = compile_model(core, INT8_MODEL)

    fp32_summary, fp32_records = benchmark(
        fp32,
        images,
        "FP32",
    )

    int8_summary, int8_records = benchmark(
        int8,
        images,
        "INT8",
    )

    fp32_total = fp32_summary["total"]
    int8_total = int8_summary["total"]

    mean_reduction = (
        (
            fp32_total["mean_ms"]
            - int8_total["mean_ms"]
        )
        / fp32_total["mean_ms"]
    ) * 100

    p95_reduction = (
        (
            fp32_total["p95_ms"]
            - int8_total["p95_ms"]
        )
        / fp32_total["p95_ms"]
    ) * 100

    p99_reduction = (
        (
            fp32_total["p99_ms"]
            - int8_total["p99_ms"]
        )
        / fp32_total["p99_ms"]
    ) * 100

    speedup = (
        fp32_total["mean_ms"]
        / int8_total["mean_ms"]
    )

    throughput_gain = (
        (
            int8_total["fps"]
            - fp32_total["fps"]
        )
        / fp32_total["fps"]
    ) * 100

    result = {
        "experiment": "E4",
        "description": (
            "End-to-end local LARA inference pipeline "
            "benchmark comparing FP32 and INT8 OpenVINO"
        ),
        "configuration": {
            "images": len(images),
            "input_size": [
                1,
                3,
                INPUT_SIZE,
                INPUT_SIZE,
            ],
            "batch_size": 1,
            "device": "CPU",
            "performance_hint": "LATENCY",
            "warmup": WARMUP,
            "confidence_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
        },
        "fp32": {
            "summary": fp32_summary,
            "per_image": fp32_records,
        },
        "int8": {
            "summary": int8_summary,
            "per_image": int8_records,
        },
        "comparison": {
            "mean_total_latency_reduction_percent":
                mean_reduction,
            "p95_total_latency_reduction_percent":
                p95_reduction,
            "p99_total_latency_reduction_percent":
                p99_reduction,
            "mean_total_speedup":
                speedup,
            "throughput_gain_percent":
                throughput_gain,
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("E4 RESULTS — TOTAL PIPELINE")
    print("=" * 78)

    print(
        f"FP32 mean total : "
        f"{fp32_total['mean_ms']:.3f} ms"
    )

    print(
        f"INT8 mean total : "
        f"{int8_total['mean_ms']:.3f} ms"
    )

    print(
        f"Mean reduction  : "
        f"{mean_reduction:.2f}%"
    )

    print(
        f"Mean speedup    : "
        f"{speedup:.3f}x"
    )

    print()

    print(
        f"FP32 P95        : "
        f"{fp32_total['p95_ms']:.3f} ms"
    )

    print(
        f"INT8 P95        : "
        f"{int8_total['p95_ms']:.3f} ms"
    )

    print(
        f"P95 reduction   : "
        f"{p95_reduction:.2f}%"
    )

    print()

    print(
        f"FP32 P99        : "
        f"{fp32_total['p99_ms']:.3f} ms"
    )

    print(
        f"INT8 P99        : "
        f"{int8_total['p99_ms']:.3f} ms"
    )

    print(
        f"P99 reduction   : "
        f"{p99_reduction:.2f}%"
    )

    print()

    print(
        f"FP32 throughput : "
        f"{fp32_total['fps']:.3f} FPS"
    )

    print(
        f"INT8 throughput : "
        f"{int8_total['fps']:.3f} FPS"
    )

    print(
        f"Throughput gain : "
        f"{throughput_gain:.2f}%"
    )

    print("\nComponent means:")
    print(
        f"FP32 preprocessing : "
        f"{fp32_summary['preprocess']['mean_ms']:.3f} ms"
    )
    print(
        f"INT8 preprocessing : "
        f"{int8_summary['preprocess']['mean_ms']:.3f} ms"
    )
    print(
        f"FP32 inference     : "
        f"{fp32_summary['inference']['mean_ms']:.3f} ms"
    )
    print(
        f"INT8 inference     : "
        f"{int8_summary['inference']['mean_ms']:.3f} ms"
    )
    print(
        f"FP32 postprocess   : "
        f"{fp32_summary['postprocess']['mean_ms']:.3f} ms"
    )
    print(
        f"INT8 postprocess   : "
        f"{int8_summary['postprocess']['mean_ms']:.3f} ms"
    )

    print()
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()