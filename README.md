# Why this matters
Production inference performance is often limited by runtime overhead, batching behavior, and hardware constraints—not just model architecture.
This project isolates those factors empirically and shows why assumptions like “INT8 is always faster” or “TensorRT always wins” are hardware- and workload-dependent.

# Inference Runtime Optimization Pipeline (GPU & CPU)

This project presents an end-to-end **inference optimization and benchmarking pipeline** for Transformer models, focusing on **runtime behavior, batching effects, precision tradeoffs, and hardware-aware performance analysis**.

The goal is not model training, but to **understand how inference performance changes across execution backends** (PyTorch eager, ONNXRuntime, TensorRT) and why empirical benchmarking is critical in real systems.

---

## What This Project Demonstrates

- How inference performance is affected more by **runtime execution model** than by numerical precision alone
- Latency vs throughput tradeoffs across **batch sizes**
- CPU INT8 dynamic quantization behavior (overhead vs amortization)
- GPU inference differences between:
  - PyTorch eager execution
  - ONNXRuntime CUDA execution
  - TensorRT Execution Provider (engine-based execution)
- Why **TensorRT does not always outperform CUDA runtimes**, depending on hardware and model characteristics

This project reflects **production-style inference thinking**, not academic ML experimentation.

---

## Model & Hardware

- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Task:** Sentiment classification (SST-2)
- **Sequence length:** 128 tokens
- **GPU:** NVIDIA GeForce GTX 1650
- **Frameworks:** PyTorch, ONNXRuntime
- **Precision modes evaluated:** FP32, INT8 (CPU), FP16 enablement tested in TensorRT

---

## Pipeline Stages

### Stage 1 — PyTorch GPU FP32 (Baseline)
- Runtime: PyTorch eager execution
- Device: GPU
- Purpose: Establish baseline latency and throughput
- Script: `src/infer_fp32.py`

---

### Stage 2A — CPU INT8 (Dynamic Quantization)
- Runtime: PyTorch dynamic quantization
- Device: CPU
- Purpose: Observe quantization overhead vs batching benefits
- Script: `src/infer_int8_cpu.py`

> Note: GPU INT8 is **not meaningful** with PyTorch dynamic quantization; CPU is the correct evaluation target here.

---

### Stage 3 — ONNXRuntime GPU (CUDAExecutionProvider)
- Runtime: ONNXRuntime (CUDA EP)
- Device: GPU
- Precision: FP32
- Purpose: Measure performance gains from graph-based execution and reduced Python overhead
- Scripts:
  - Export: `src/export_onnx.py`
  - Inference: `src/infer_onnx_gpu.py`

This stage produced the **best performance** on the target hardware.

---

### Stage 4 — TensorRT Execution Provider (Engine)
- Runtime: ONNXRuntime + TensorRT EP
- Device: GPU
- Precision: FP16 enablement tested, INT8 toggle evaluated
- Purpose: Evaluate engine-based inference vs CUDA EP
- Script: `src/infer_onnx_trt_ep.py`

> On GTX 1650, TensorRT EP underperformed CUDA EP for this model, highlighting hardware-specific limitations.

---

## Results

Detailed latency and throughput numbers are provided in:

📄 `results/summary_table.md`

Metrics include:
- p50 / p95 / average latency
- Throughput (samples/sec)
- Batch sizes: 1, 4, 8, 16
- Warmup vs timed runs (to avoid cold-start bias)

---

## Key Engineering Observations

A detailed, non-generic analysis is documented in:

📄 `notes/engineering_observations.md`

Highlights include:
- Runtime overhead dominating performance more than precision changes
- Why CPU INT8 increases batch-1 latency but improves large-batch throughput
- Why ONNXRuntime CUDA outperformed TensorRT EP on this specific GPU
- Why enabling FP16 does not guarantee speedups
- Why empirical benchmarking is mandatory for inference systems

---

## How to Run

### 1) Environment Setup
```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\activate
pip install -r requirements.txt
`
2) Stage 1 — PyTorch GPU FP32
python src/infer_fp32.py

3) Stage 2A — CPU INT8
python src/infer_int8_cpu.py

4) Stage 3 — ONNXRuntime GPU
python src/export_onnx.py
python src/infer_onnx_gpu.py

5) Stage 4 — TensorRT EP
python src/infer_onnx_trt_ep.py

The first TensorRT run may be slower due to engine build and caching.`
