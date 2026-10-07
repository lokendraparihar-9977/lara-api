from __future__ import annotations

import argparse
import csv
import gc
import os
import threading
import time
from pathlib import Path
from statistics import mean, median
from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort
import openvino as ov
import psutil
import torch
from ultralytics import YOLO


RuntimeFn = Callable[[np.ndarray], None]

PROCESS = psutil.Process(os.getpid())


def load_images(
    image_dir: Path,
    size: int,
) -> list[np.ndarray]:

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    )

    if not paths:
        raise FileNotFoundError(
            f"No images found in {image_dir}"
        )

    images: list[np.ndarray] = []

    for path in paths:

        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(
                f"Could not read image: {path}"
            )

        image = cv2.resize(
            image,
            (size, size),
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        tensor = (
            image
            .transpose(2, 0, 1)
            .astype(np.float32)
            / 255.0
        )

        images.append(tensor)

    return images


def memory_mb() -> float:
    return (
        PROCESS.memory_info().rss
        / (1024.0 * 1024.0)
    )


class ResourceMonitor:
    """
    Low-overhead background resource monitor.

    CPU utilization is sampled independently from
    the inference timing loop so monitoring overhead
    does not contaminate latency measurements.
    """

    def __init__(
        self,
        interval: float = 0.25,
    ) -> None:

        self.interval = interval

        self._stop_event = (
            threading.Event()
        )

        self.cpu_samples: list[float] = []
        self.memory_samples: list[float] = []

        self._thread = threading.Thread(
            target=self._monitor,
            daemon=True,
        )

    def start(self) -> None:

        PROCESS.cpu_percent(
            interval=None
        )

        self._thread.start()

    def stop(self) -> None:

        self._stop_event.set()
        self._thread.join()

    def _monitor(self) -> None:

        while not self._stop_event.wait(
            self.interval
        ):

            cpu = PROCESS.cpu_percent(
                interval=None
            )

            ram = memory_mb()

            self.cpu_samples.append(
                cpu
            )

            self.memory_samples.append(
                ram
            )


def benchmark(
    name: str,
    inference_fn: RuntimeFn,
    images: list[np.ndarray],
    warmup: int,
    iterations: int,
) -> dict[str, float | str]:

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    print(
        f"Warm-up: {warmup} runs..."
    )

    for index in range(warmup):

        inference_fn(
            images[index % len(images)]
        )

    print("Warm-up complete.")

    gc.collect()

    # ---------------------------------------------------------
    # Baseline memory
    # ---------------------------------------------------------

    memory_before = memory_mb()

    total_runs = (
        len(images) * iterations
    )

    print(
        f"Benchmarking: "
        f"{len(images)} images × "
        f"{iterations} iterations = "
        f"{total_runs} runs..."
    )

    # ---------------------------------------------------------
    # Start background resource monitor
    # ---------------------------------------------------------

    monitor = ResourceMonitor(
        interval=0.25
    )

    monitor.start()

    latencies: list[float] = []

    try:

        for iteration in range(
            iterations
        ):

            for image in images:

                start = time.perf_counter()

                inference_fn(image)

                elapsed_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                latencies.append(
                    elapsed_ms
                )

    finally:

        monitor.stop()

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    mean_latency = mean(
        latencies
    )

    median_latency = median(
        latencies
    )

    throughput = (
        1000.0 / mean_latency
    )

    peak_memory = max(
        monitor.memory_samples,
        default=memory_before,
    )

    memory_delta = (
        peak_memory
        - memory_before
    )

    mean_cpu = mean(
        monitor.cpu_samples
    ) if monitor.cpu_samples else 0.0

    result = {
        "runtime": name,
        "images": len(images),
        "iterations": iterations,
        "total_runs": total_runs,
        "mean_ms": mean_latency,
        "median_ms": median_latency,
        "fps": throughput,
        "mean_cpu_percent": mean_cpu,
        "memory_before_mb": memory_before,
        "peak_memory_mb": peak_memory,
        "memory_delta_mb": memory_delta,
    }

    print()
    print(
        f"Mean latency:       "
        f"{mean_latency:.2f} ms"
    )

    print(
        f"Median latency:     "
        f"{median_latency:.2f} ms"
    )

    print(
        f"Throughput:         "
        f"{throughput:.2f} FPS"
    )

    print(
        f"Mean CPU usage:     "
        f"{mean_cpu:.2f}%"
    )

    print(
        f"Memory before:      "
        f"{memory_before:.2f} MB"
    )

    print(
        f"Peak memory:        "
        f"{peak_memory:.2f} MB"
    )

    print(
        f"Memory increase:    "
        f"{memory_delta:.2f} MB"
    )

    return result


def create_pytorch_runner(
    model_path: Path,
) -> RuntimeFn:

    torch.set_num_threads(2)

    model = YOLO(
        str(model_path)
    ).model

    model.eval()
    model.to("cpu")

    @torch.inference_mode()
    def inference(
        image: np.ndarray,
    ) -> None:

        tensor = (
            torch.from_numpy(image)
            .unsqueeze(0)
        )

        model(tensor)

    return inference


def create_onnx_runner(
    model_path: Path,
) -> RuntimeFn:

    session_options = (
        ort.SessionOptions()
    )

    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    input_name = (
        session.get_inputs()[0].name
    )

    def inference(
        image: np.ndarray,
    ) -> None:

        batch = np.expand_dims(
            image,
            axis=0,
        )

        session.run(
            None,
            {
                input_name: batch
            },
        )

    return inference


def create_openvino_runner(
    model_path: Path,
) -> RuntimeFn:

    core = ov.Core()

    model = core.read_model(
        str(model_path)
    )

    compiled_model = (
        core.compile_model(
            model,
            "CPU",
            {
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": 1,
                "INFERENCE_NUM_THREADS": 2,
            },
        )
    )

    input_layer = (
        compiled_model.input(0)
    )

    def inference(
        image: np.ndarray,
    ) -> None:

        batch = np.expand_dims(
            image,
            axis=0,
        )

        compiled_model(
            {
                input_layer: batch
            }
        )

    return inference


def save_results(
    results: list[
        dict[str, float | str]
    ],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        results[0].keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def print_comparison(
    results: list[
        dict[str, float | str]
    ],
) -> None:

    print()
    print("=" * 110)
    print(
        "CPU RESOURCE EFFICIENCY COMPARISON"
    )
    print("=" * 110)

    print(
        f"{'Runtime':<22}"
        f"{'Mean ms':>12}"
        f"{'FPS':>10}"
        f"{'CPU %':>12}"
        f"{'Peak RAM':>15}"
        f"{'RAM Δ':>15}"
    )

    print("-" * 110)

    for result in results:

        print(
            f"{str(result['runtime']):<22}"
            f"{float(result['mean_ms']):>12.2f}"
            f"{float(result['fps']):>10.2f}"
            f"{float(result['mean_cpu_percent']):>12.2f}"
            f"{float(result['peak_memory_mb']):>15.2f}"
            f"{float(result['memory_delta_mb']):>15.2f}"
        )

    print("-" * 110)
    print("=" * 110)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "CPU resource efficiency "
            "benchmark for LARA."
        )
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
            "research/results/"
            "resource_comparison_val500.csv"
        ),
    )

    args = parser.parse_args()

    project_root = Path.cwd()

    pytorch_model = (
        project_root
        / "yolov8n.pt"
    )

    onnx_model = (
        project_root
        / "yolov8n.onnx"
    )

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
    print(
        "LARA CPU RESOURCE EFFICIENCY BENCHMARK"
    )
    print("=" * 70)

    print(
        f"Image directory: "
        f"{args.image_dir}"
    )

    print(
        f"Input size:      "
        f"{args.size} × {args.size}"
    )

    print(
        f"Iterations:      "
        f"{args.iterations}"
    )

    print(
        f"Warm-up runs:    "
        f"{args.warmup}"
    )

    print(
        "CPU threads:     2"
    )

    print()

    images = load_images(
        args.image_dir,
        args.size,
    )

    print(
        f"Images loaded:   "
        f"{len(images)}"
    )

    results: list[
        dict[str, float | str]
    ] = []

    results.append(
        benchmark(
            "PyTorch CPU",
            create_pytorch_runner(
                pytorch_model
            ),
            images,
            args.warmup,
            args.iterations,
        )
    )

    results.append(
        benchmark(
            "ONNX Runtime CPU",
            create_onnx_runner(
                onnx_model
            ),
            images,
            args.warmup,
            args.iterations,
        )
    )

    results.append(
        benchmark(
            "OpenVINO FP32 CPU",
            create_openvino_runner(
                openvino_model
            ),
            images,
            args.warmup,
            args.iterations,
        )
    )

    results.append(
        benchmark(
            "OpenVINO INT8 CPU",
            create_openvino_runner(
                openvino_int8_model
            ),
            images,
            args.warmup,
            args.iterations,
        )
    )

    print_comparison(results)

    save_results(
        results,
        args.output,
    )

    print()
    print(
        "Results saved to:"
    )
    print(args.output)


if __name__ == "__main__":
    main()