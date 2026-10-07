# LARA — Lightweight Adaptive Recognition API

LARA (Lightweight Adaptive Recognition API) is a computer-vision API designed to provide object detection on resource-constrained systems without requiring a dedicated GPU.

The research component of this repository evaluates practical CPU inference optimization using YOLOv8n, OpenVINO, and INT8 quantization.

---

## Research Objective

The objective of this study is to evaluate whether INT8 OpenVINO optimization can reduce CPU inference latency while maintaining comparable object-detection accuracy on a resource-constrained system.

The evaluation focuses on:

- FP32 vs INT8 inference latency
- Multi-image latency behavior
- End-to-end inference latency
- Accuracy preservation after INT8 quantization
- Batch-size scaling
- API-level latency optimization through deferred database logging

---

## Hardware

All CPU inference experiments were performed on the following system:

| Component | Specification |
|---|---|
| CPU | Intel Core i3-1115G4 |
| Physical cores | 2 |
| Logical processors | 4 |
| GPU | Intel UHD Graphics |
| RAM | 8 GB |
| Dedicated GPU | None |

The experiments were conducted using CPU inference.

---

## Software Environment

The main software environment used for the experiments was:

| Software | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.13.0+cpu |
| Ultralytics | 8.4.118 |
| ONNX Runtime | 1.23.2 |
| OpenVINO | 2026.3.0 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |

---

## Model

The evaluated object-detection model is **YOLOv8n**.

Two OpenVINO representations were evaluated:

- FP32 OpenVINO model
- INT8 OpenVINO model

The model weights and exported model binaries are intentionally excluded from version control.

---

## Evaluation Dataset

The inference and accuracy experiments use a **500-image subset of the COCO validation dataset**.

The evaluation subset contains:

- 500 images
- 3,504 annotated object instances
- 80 COCO object classes

The dataset itself is not committed to the repository.

The repository contains the evaluation configuration and scripts required to reproduce the evaluation after obtaining the dataset.

---

# Experiments

## E3 — Controlled Single-Image FP32 vs INT8 Latency

A controlled latency experiment was performed at 640×640 resolution using OpenVINO CPU inference.

Configuration:

- 10 warm-up runs
- 100 measured iterations
- Batch size: 1
- CPU execution
- 640×640 input resolution

### Results

| Metric | FP32 | INT8 |
|---|---:|---:|
| Mean latency | 51.774 ms | 21.997 ms |
| Median latency | 50.539 ms | 19.716 ms |
| P95 latency | 60.184 ms | 27.439 ms |
| P99 latency | 63.336 ms | 36.104 ms |
| Throughput | 19.315 FPS | 45.460 FPS |

INT8 reduced mean latency by approximately **57.5%** and produced approximately **2.35×** the FP32 throughput.

---

## E3b — 500-Image Latency Evaluation

A larger controlled evaluation was performed using the 500-image COCO subset.

Configuration:

- 500 images
- 640×640 input resolution
- Batch size: 1
- 10 warm-up runs
- CPU execution
- OpenVINO LATENCY mode

### Results

| Metric | FP32 | INT8 |
|---|---:|---:|
| Mean latency | 68.579 ms | 23.527 ms |
| Median latency | 61.822 ms | 22.389 ms |
| P95 latency | 100.789 ms | 28.851 ms |
| P99 latency | 138.710 ms | 38.362 ms |
| Throughput | 14.582 FPS | 42.504 FPS |

INT8 reduced mean latency by approximately **65.7%** and increased measured throughput by approximately **191.5%** relative to FP32.

---

## Accuracy Evaluation

Accuracy was evaluated on the same 500-image COCO subset.

### Results

| Metric | FP32 | INT8 |
|---|---:|---:|
| mAP@50 | 0.5571 | 0.5547 |
| mAP@50:95 | 0.4015 | 0.4012 |

The observed difference in mAP was small on this evaluation subset:

- mAP@50 difference: approximately **0.0024**
- mAP@50:95 difference: approximately **0.0003**

These values are descriptive results from the evaluated subset and should not be interpreted as a statistical proof of general accuracy equivalence.

---

## E4 — End-to-End FP32 vs INT8 Inference

The end-to-end benchmark measures the complete inference pipeline, including:

1. Preprocessing
2. Model inference
3. Postprocessing

Configuration:

- 500 images
- 640×640 input resolution
- Batch size: 1
- 10 warm-up runs
- Confidence threshold: 0.25
- IoU threshold: 0.45
- CPU execution

### Results

| Metric | FP32 | INT8 |
|---|---:|---:|
| Mean total latency | 77.705 ms | 34.378 ms |
| P95 latency | 97.620 ms | 45.809 ms |
| P99 latency | 116.921 ms | 66.680 ms |
| Throughput | 12.869 FPS | 29.088 FPS |

INT8 reduced mean end-to-end latency by approximately **55.8%** and increased measured throughput by approximately **126.0%**.

### Mean Pipeline Components

| Component | FP32 | INT8 |
|---|---:|---:|
| Preprocessing | 15.484 ms | 6.758 ms |
| Inference | 59.971 ms | 25.276 ms |
| Postprocessing | 2.250 ms | 2.345 ms |

---

## E5 — Batch-Size Scaling

Batch-size scaling was evaluated using batch sizes of:

- 1
- 2
- 4
- 8

The experiment evaluated per-image latency and throughput for FP32 and INT8 OpenVINO inference.

### Results

| Batch Size | FP32 ms/img | INT8 ms/img | FP32 FPS | INT8 FPS |
|---:|---:|---:|---:|---:|
| 1 | 75.409 | 49.893 | 13.257 | 20.030 |
| 2 | 79.905 | 54.312 | 12.513 | 18.394 |
| 4 | 82.829 | 57.118 | 12.072 | 17.504 |
| 8 | 112.537 | 44.950 | 8.885 | 22.242 |

INT8 maintained lower measured per-image latency and higher throughput than FP32 across the tested batch sizes.

The magnitude of the improvement varied with batch size. These results describe the behavior observed on the tested hardware and should not be interpreted as establishing a universally optimal batch size.

---

# API-Level Optimization

In addition to model-level CPU optimization, an API-level experiment evaluated **deferred database logging**.

The experiment compared synchronous database logging with deferred logging using 50 requests.

### Results

| Metric | Improvement |
|---|---:|
| Mean latency reduction | 39.31% |
| Median latency reduction | 40.53% |
| P95 latency reduction | 51.19% |
| P99 latency reduction | 69.51% |
| Standard-deviation reduction | 86.65% |

Statistical validation included:

- Bootstrap confidence interval
- Welch's t-test
- Mann–Whitney U test
- Kolmogorov–Smirnov test
- Cohen's d
- Cliff's delta

The purpose of this experiment was to evaluate whether moving database logging out of the critical request path could reduce API response latency and variability.

---

# Repository Structure

The research-related components are organized as follows:

```text
LARA/
│
├── main.py
├── main_deferred.py
├── requirements.txt
├── Dockerfile
│
├── research/
│   ├── benchmark_e3_latency.py
│   ├── benchmark_e3_multimage.py
│   ├── benchmark_e4_e2e.py
│   ├── benchmark_e5_batch_scaling.py
│   ├── evaluate_accuracy.py
│   ├── analyze_experiment_5b.py
│   ├── benchmark_api.py
│   ├── benchmark_api_decomposition.py
│   ├── benchmark_baseline.py
│   ├── benchmark_db.py
│   ├── benchmark_resources.py
│   ├── benchmark_runtimes.py
│   ├── prepare_coco_val500.py
│   ├── validate_experiment_5b.py
│   │
│   ├── evaluation/
│   │   └── coco_val500/
│   │       └── data.yaml
│   │
│   └── results/
│       ├── e3_fp32_vs_int8_controlled_latency.json
│       ├── e3_multimage_fp32_vs_int8_latency.json
│       ├── e4_fp32_vs_int8_end_to_end.json
│       ├── e5_batch_scaling_fp32_vs_int8.json
│       ├── accuracy_comparison_val500.csv
│       ├── experiment_5b_final_analysis.json
│       └── experiment_5b_statistical_validation.json
│
└── ...