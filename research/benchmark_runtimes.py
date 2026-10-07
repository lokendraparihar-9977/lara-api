from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort
import openvino as ov
import torch
from ultralytics import YOLO


RuntimeFn = Callable[[np.ndarray], None]


def load_images(image_dir: Path, size: int) -> list[np.ndarray]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    images: list[np.ndarray] = []

    for path in paths:
        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(f"Could not read image: {path}")

        image = cv2.resize(image, (size, size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # HWC -> CHW, uint8 -> float32, normalize to [0, 1]
        tensor = image.transpose(2, 0, 1).astype(np.float32) / 255.0

        images.append(tensor)

    return images


def benchmark(
    name: str,
    inference_fn: RuntimeFn,
    images: list[np.ndarray],
    warmup: int,
    iterations: int,
) -> dict[str, float | str]:

    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"{'=' * 70}")

    # Warm-up
    print(f"Warm-up: {warmup} runs...")

    for i in range(warmup):
        inference_fn(images[i % len(images)])

    print("Warm-up complete.")

    latencies: list[float] = []

    total_runs = len(images) * iterations

    print(
        f"Benchmarking: {len(images)} images × "
        f"{iterations} iterations = {total_runs} runs..."
    )

    for iteration in range(iterations):
        for image in images:

            start = time.perf_counter()

            inference_fn(image)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

    mean_latency = mean(latencies)
    median_latency = median(latencies)
    std_latency = stdev(latencies) if len(latencies) > 1 else 0.0
    min_latency = min(latencies)
    max_latency = max(latencies)

    throughput = 1000.0 / mean_latency

    result = {
        "runtime": name,
        "images": len(images),
        "iterations": iterations,
        "total_runs": total_runs,
        "mean_ms": mean_latency,
        "median_ms": median_latency,
        "std_ms": std_latency,
        "min_ms": min_latency,
        "max_ms": max_latency,
        "fps": throughput,
    }

    print()
    print(f"Mean latency:    {mean_latency:.2f} ms")
    print(f"Median latency:  {median_latency:.2f} ms")
    print(f"Std deviation:   {std_latency:.2f} ms")
    print(f"Minimum latency: {min_latency:.2f} ms")
    print(f"Maximum latency: {max_latency:.2f} ms")
    print(f"Throughput:      {throughput:.2f} FPS")

    return result


def create_pytorch_runner(model_path: Path) -> RuntimeFn:
    model = YOLO(str(model_path)).model
    model.eval()
    model.to("cpu")

    torch.set_num_threads(2)

    @torch.inference_mode()
    def inference(image: np.ndarray) -> None:
        tensor = torch.from_numpy(image).unsqueeze(0)
        model(tensor)

    return inference


def create_onnx_runner(model_path: Path) -> RuntimeFn:
    session_options = ort.SessionOptions()

    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    def inference(image: np.ndarray) -> None:
        batch = np.expand_dims(image, axis=0)
        session.run(None, {input_name: batch})

    return inference


def create_openvino_runner(model_path: Path) -> RuntimeFn:
    core = ov.Core()

    model = core.read_model(str(model_path))

    compiled_model = core.compile_model(
        model,
        "CPU",
        {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": 1,
            "INFERENCE_NUM_THREADS": 2,
        },
    )

    input_layer = compiled_model.input(0)

    def inference(image: np.ndarray) -> None:
        batch = np.expand_dims(image, axis=0)
        compiled_model({input_layer: batch})

    return inference


def save_results(
    results: list[dict[str, float | str]],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)


def print_comparison(results: list[dict[str, float | str]]) -> None:
    print()
    print("=" * 90)
    print("CONTROLLED CPU RUNTIME COMPARISON")
    print("=" * 90)

    print(
        f"{'Runtime':<20}"
        f"{'Mean ms':>12}"
        f"{'Median ms':>14}"
        f"{'FPS':>12}"
    )

    print("-" * 90)

    for result in results:
        print(
            f"{str(result['runtime']):<20}"
            f"{float(result['mean_ms']):>12.2f}"
            f"{float(result['median_ms']):>14.2f}"
            f"{float(result['fps']):>12.2f}"
        )

    fastest = min(
        results,
        key=lambda x: float(x["mean_ms"]),
    )

    print("-" * 90)

    print(
        f"Fastest runtime: "
        f"{fastest['runtime']} "
        f"({float(fastest['mean_ms']):.2f} ms)"
    )

    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled CPU benchmark for LARA runtimes."
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("datasets/coco128/images/train2017"),
    )

    parser.add_argument(
        "--size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
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
            "research/results/runtime_comparison.csv"
        ),
    )

    args = parser.parse_args()

    project_root = Path.cwd()

    pytorch_model = project_root / "yolov8n.pt"
    onnx_model = project_root / "yolov8n.onnx"
    openvino_model = (
        project_root
        / "yolov8n_openvino_model"
        / "yolov8n.xml"
    )

    openvino_int8_model = (
        project_root
        / "yolov8n_int8_openvino_model"
        / "yolov8n.xml"
    )

    print("=" * 70)
    print("LARA CONTROLLED CPU RUNTIME BENCHMARK")
    print("=" * 70)

    print(f"Image directory: {args.image_dir}")
    print(f"Input size:      {args.size} × {args.size}")
    print(f"Iterations:      {args.iterations}")
    print(f"Warm-up runs:    {args.warmup}")
    print("CPU threads:     2")
    print()

    images = load_images(
        args.image_dir,
        args.size,
    )

    print(f"Images loaded:   {len(images)}")

    if len(images) != 128:
        print(
            f"WARNING: Expected 128 COCO128 images, "
            f"but found {len(images)}."
        )

    results: list[dict[str, float | str]] = []

    # ---------------------------------------------------------
    # 1. PyTorch
    # ---------------------------------------------------------

    pytorch_runner = create_pytorch_runner(
        pytorch_model
    )

    results.append(
        benchmark(
            "PyTorch CPU",
            pytorch_runner,
            images,
            args.warmup,
            args.iterations,
        )
    )

    # ---------------------------------------------------------
    # 2. ONNX Runtime
    # ---------------------------------------------------------

    onnx_runner = create_onnx_runner(
        onnx_model
    )

    results.append(
        benchmark(
            "ONNX Runtime CPU",
            onnx_runner,
            images,
            args.warmup,
            args.iterations,
        )
    )

    # ---------------------------------------------------------
    # 3. OpenVINO
    # ---------------------------------------------------------

    openvino_runner = create_openvino_runner(
        openvino_model
    )

    results.append(
        benchmark(
            "OpenVINO CPU",
            openvino_runner,
            images,
            args.warmup,
            args.iterations,
        )
    )

    # ---------------------------------------------------------
    # 4. OpenVINO INT8
    # ---------------------------------------------------------

    openvino_int8_runner = create_openvino_runner(
        openvino_int8_model
    )

    results.append(
        benchmark(
            "OpenVINO INT8 CPU",
            openvino_int8_runner,
            images,
            args.warmup,
            args.iterations,
        )
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print_comparison(results)

    save_results(
        results,
        args.output,
    )

    print()
    print(f"Results saved to:")
    print(args.output)


if __name__ == "__main__":
    main()