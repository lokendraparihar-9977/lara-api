import time
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
from openvino import Core


MODELS = {
    "FP32": r"yolov8n_openvino_model\yolov8n.xml",
    "INT8": r"yolov8n_int8_openvino_model\yolov8n.xml",
}

IMAGE = r"research\test_images\test.jpg"
INPUT_SIZE = 640
WARMUP = 10
ITERATIONS = 100


def percentile(values, p):
    return float(np.percentile(values, p))


def prepare_input(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return np.ascontiguousarray(image)


def benchmark_model(core, model_path, input_tensor):
    model = core.read_model(model_path)
    compiled = core.compile_model(model, "CPU")
    infer_request = compiled.create_infer_request()

    input_layer = compiled.input(0)

    # Warm-up
    for _ in range(WARMUP):
        infer_request.infer({input_layer: input_tensor})

    latencies_ms = []

    for _ in range(ITERATIONS):
        start = time.perf_counter()
        infer_request.infer({input_layer: input_tensor})
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "std_ms": statistics.stdev(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "throughput_fps": 1000.0 / statistics.mean(latencies_ms),
        "raw_ms": latencies_ms,
    }


def main():
    print("=" * 78)
    print("LARA E3 — CONTROLLED FP32 vs INT8 OPENVINO LATENCY BENCHMARK")
    print("=" * 78)
    print(f"Image:       {IMAGE}")
    print(f"Input size:  {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Warm-up:     {WARMUP}")
    print(f"Iterations:  {ITERATIONS}")
    print("Device:      CPU")
    print()

    input_tensor = prepare_input(IMAGE)
    core = Core()

    results = {}

    for name, model_path in MODELS.items():
        print(f"Benchmarking {name}...")
        results[name] = benchmark_model(core, model_path, input_tensor)

        r = results[name]

        print(
            f"{name}: "
            f"mean={r['mean_ms']:.3f} ms | "
            f"median={r['median_ms']:.3f} ms | "
            f"P95={r['p95_ms']:.3f} ms | "
            f"P99={r['p99_ms']:.3f} ms | "
            f"FPS={r['throughput_fps']:.3f}"
        )
        print()

    fp32 = results["FP32"]
    int8 = results["INT8"]

    mean_speedup = fp32["mean_ms"] / int8["mean_ms"]
    mean_reduction = (
        (fp32["mean_ms"] - int8["mean_ms"])
        / fp32["mean_ms"]
        * 100.0
    )

    p95_reduction = (
        (fp32["p95_ms"] - int8["p95_ms"])
        / fp32["p95_ms"]
        * 100.0
    )

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"Mean latency FP32 : {fp32['mean_ms']:.3f} ms")
    print(f"Mean latency INT8 : {int8['mean_ms']:.3f} ms")
    print(f"Mean reduction    : {mean_reduction:.2f}%")
    print(f"Mean speedup      : {mean_speedup:.3f}x")
    print()
    print(f"P95 latency FP32  : {fp32['p95_ms']:.3f} ms")
    print(f"P95 latency INT8  : {int8['p95_ms']:.3f} ms")
    print(f"P95 reduction     : {p95_reduction:.2f}%")
    print()
    print(f"Throughput FP32   : {fp32['throughput_fps']:.3f} FPS")
    print(f"Throughput INT8   : {int8['throughput_fps']:.3f} FPS")

    output = {
        "experiment": "E3",
        "title": "Controlled FP32 vs INT8 OpenVINO latency benchmark",
        "configuration": {
            "device": "CPU",
            "input_size": INPUT_SIZE,
            "batch_size": 1,
            "warmup": WARMUP,
            "iterations": ITERATIONS,
            "image": IMAGE,
        },
        "models": {
            "FP32": MODELS["FP32"],
            "INT8": MODELS["INT8"],
        },
        "results": {
            "FP32": {
                k: v for k, v in fp32.items() if k != "raw_ms"
            },
            "INT8": {
                k: v for k, v in int8.items() if k != "raw_ms"
            },
        },
        "comparison": {
            "mean_latency_reduction_percent": mean_reduction,
            "mean_speedup_x": mean_speedup,
            "p95_latency_reduction_percent": p95_reduction,
        },
        "raw_latencies_ms": {
            "FP32": fp32["raw_ms"],
            "INT8": int8["raw_ms"],
        },
    }

    output_path = Path(
        "research/results/e3_fp32_vs_int8_controlled_latency.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
