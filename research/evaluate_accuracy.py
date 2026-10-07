from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def evaluate_model(
    name: str,
    model_path: Path,
    data_path: Path,
    image_size: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Evaluate one YOLO detection model using identical validation settings."""

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Model: {model_path}")
    print(f"Dataset: {data_path}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print()

    # Explicit task definition is important for exported models.
    model = YOLO(
        str(model_path),
        task="detect",
    )

    metrics = model.val(
        data=str(data_path),
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=0,
        verbose=True,
        plots=False,
        save_json=False,
    )

    result = {
        "runtime": name,
        "model": str(model_path),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    print()
    print("-" * 70)
    print("ACCURACY RESULTS")
    print("-" * 70)

    print(f"Precision:   {result['precision']:.6f}")
    print(f"Recall:      {result['recall']:.6f}")
    print(f"mAP@50:      {result['map50']:.6f}")
    print(f"mAP@50-95:   {result['map50_95']:.6f}")

    return result


def save_results(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save evaluation results to CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "runtime",
        "model",
        "precision",
        "recall",
        "map50",
        "map50_95",
    ]

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
    results: list[dict[str, Any]],
) -> None:
    """Print a controlled accuracy comparison."""

    print()
    print("=" * 95)
    print("CONTROLLED ACCURACY COMPARISON")
    print("=" * 95)

    print(
        f"{'Runtime':<25}"
        f"{'Precision':>14}"
        f"{'Recall':>14}"
        f"{'mAP@50':>14}"
        f"{'mAP@50-95':>16}"
    )

    print("-" * 95)

    for result in results:

        print(
            f"{result['runtime']:<25}"
            f"{result['precision']:>14.4f}"
            f"{result['recall']:>14.4f}"
            f"{result['map50']:>14.4f}"
            f"{result['map50_95']:>16.4f}"
        )

    print("=" * 95)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Controlled accuracy evaluation for "
            "LARA YOLO runtimes."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "research/evaluation/coco_val500/data.yaml"
        ),
    )

    parser.add_argument(
        "--size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/results/accuracy_comparison_val500.csv"
        ),
    )

    args = parser.parse_args()

    project_root = Path.cwd()

    models = [
        (
        "PyTorch FP32",
        project_root / "yolov8n.pt",
        ),
        (
        "ONNX Runtime",
        project_root / "yolov8n.onnx",
        ),
        (
        "OpenVINO FP32",
        project_root / "yolov8n_openvino_model",
        ),
        (
        "OpenVINO INT8",
        project_root / "yolov8n_int8_openvino_model",
        ),
    ]

    print("=" * 70)
    print("LARA CONTROLLED ACCURACY EVALUATION")
    print("=" * 70)

    print(f"Dataset:    {args.data}")
    print(f"Image size: {args.size}")
    print(f"Batch size: {args.batch}")
    print(f"Device:     {args.device}")
    print()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}"
        )

    for _, model_path in models:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

    results: list[dict[str, Any]] = []

    for runtime_name, model_path in models:

        result = evaluate_model(
            name=runtime_name,
            model_path=model_path,
            data_path=args.data,
            image_size=args.size,
            batch_size=args.batch,
            device=args.device,
        )

        results.append(result)

    print_comparison(results)

    save_results(
        results,
        args.output,
    )

    print()
    print("Results saved to:")
    print(args.output)


if __name__ == "__main__":
    main()