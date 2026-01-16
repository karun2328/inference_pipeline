# Results Summary (Latency + Throughput)

All timings use:
- Model: distilbert-base-uncased-finetuned-sst-2-english
- Sequence length: max_length=128
- Warmup: 10 runs
- Timed runs: 50 runs

## Stage 1 — PyTorch GPU FP32 (Baseline)
Device: NVIDIA GeForce GTX 1650 (CUDA)

| Batch | Avg Latency (ms) | Throughput (samples/s) |
|------:|------------------:|------------------------:|
| 1     | 9.68              | 103.27                  |
| 4     | 29.89             | 133.82                  |
| 8     | 66.48             | 120.34                  |
| 16    | 129.44            | 123.60                  |

## Stage 2A — CPU INT8 (PyTorch Dynamic Quantization)
Device: CPU

| Batch | Avg Latency (ms) | Throughput (samples/s) |
|------:|------------------:|------------------------:|
| 1     | 35.58             | 28.11                   |
| 4     | 30.21             | 132.40                  |
| 8     | 46.19             | 173.20                  |
| 16    | 71.21             | 224.60                  |

## Stage 3 — ONNXRuntime GPU (CUDAExecutionProvider) FP32
Device: NVIDIA GeForce GTX 1650 (CUDA)

| Batch | Avg Latency (ms) | Throughput (samples/s) |
|------:|------------------:|------------------------:|
| 1     | 3.73              | 268.17                  |
| 4     | 10.71             | 373.53                  |
| 8     | 10.98             | 728.80                  |
| 16    | 15.28             | 1046.88                 |

## Stage 4 — ONNXRuntime TensorRT EP (Engine) (FP16 enabled, INT8 tested but worse)
Device: NVIDIA GeForce GTX 1650

| Batch | Avg Latency (ms) | Throughput (samples/s) |
|------:|------------------:|------------------------:|
| 1     | 11.83             | 84.52                   |
| 4     | 55.50             | 72.08                   |
| 8     | 78.01             | 102.55                  |
| 16    | 120.46            | 132.83                  |

## Cross-Stage Comparison (FP32 vs INT8 vs Runtime)

This section compares **average latency and throughput for the same batch size**
across different runtimes and precisions.  
All runs use the same model, sequence length, and benchmarking methodology.

---

### Batch = 1

| Runtime / Stage             | Device | Precision | Avg Latency (ms) | Throughput (samples/s) |
|-----------------------------|--------|-----------|------------------:|------------------------:|
| PyTorch (Stage 1)           | GPU    | FP32      | 9.68              | 103.27                  |
| PyTorch INT8 (Stage 2A)     | CPU    | INT8      | 35.58             | 28.11                   |
| ONNXRuntime CUDA (Stage 3)  | GPU    | FP32      | **3.73**          | **268.17**              |
| TensorRT EP (Stage 4)       | GPU    | FP16/FP32 | 11.83             | 84.52                   |

---

### Batch = 4

| Runtime / Stage             | Device | Precision | Avg Latency (ms) | Throughput (samples/s) |
|-----------------------------|--------|-----------|------------------:|------------------------:|
| PyTorch (Stage 1)           | GPU    | FP32      | 29.89             | 133.82                  |
| PyTorch INT8 (Stage 2A)     | CPU    | INT8      | 30.21             | 132.40                  |
| ONNXRuntime CUDA (Stage 3)  | GPU    | FP32      | **10.71**         | **373.53**              |
| TensorRT EP (Stage 4)       | GPU    | FP16/FP32 | 55.50             | 72.08                   |

---

### Batch = 8

| Runtime / Stage             | Device | Precision | Avg Latency (ms) | Throughput (samples/s) |
|-----------------------------|--------|-----------|------------------:|------------------------:|
| PyTorch (Stage 1)           | GPU    | FP32      | 66.48             | 120.34                  |
| PyTorch INT8 (Stage 2A)     | CPU    | INT8      | 46.19             | 173.20                  |
| ONNXRuntime CUDA (Stage 3)  | GPU    | FP32      | **10.98**         | **728.80**              |
| TensorRT EP (Stage 4)       | GPU    | FP16/FP32 | 78.01             | 102.55                  |

---

### Batch = 16

| Runtime / Stage             | Device | Precision | Avg Latency (ms) | Throughput (samples/s) |
|-----------------------------|--------|-----------|------------------:|------------------------:|
| PyTorch (Stage 1)           | GPU    | FP32      | 129.44            | 123.60                  |
| PyTorch INT8 (Stage 2A)     | CPU    | INT8      | 71.21             | 224.60                  |
| ONNXRuntime CUDA (Stage 3)  | GPU    | FP32      | **15.28**         | **1046.88**             |
| TensorRT EP (Stage 4)       | GPU    | FP16/FP32 | 120.46            | 132.83                  |

---
