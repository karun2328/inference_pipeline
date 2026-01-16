# Engineering Observations (Inference Runtime Analysis)

This document summarizes non-obvious, system-level observations derived from benchmarking multiple inference execution paths for a Transformer-based sentiment model. The focus is on runtime behavior, hardware interaction, and performance tradeoffs rather than model accuracy.

---

## 1. Runtime execution model dominated performance more than numerical precision

The largest performance improvement was observed when moving from PyTorch eager GPU execution (Stage 1) to ONNXRuntime CUDA execution (Stage 3), without changing numerical precision (FP32 in both cases).

This indicates that Python interpreter overhead, eager execution scheduling, and less optimal kernel dispatch were the primary bottlenecks in the baseline. Executing the model as a static computation graph via ONNXRuntime enabled more efficient kernel scheduling and reduced per-request overhead, resulting in significantly lower latency and higher throughput.

---

## 2. CPU INT8 dynamic quantization increased batch-1 latency due to fixed overhead

Dynamic INT8 quantization on CPU (Stage 2A) increased batch-1 latency compared to FP32 execution. This behavior is explained by fixed per-inference overhead introduced by quantization and dequantization steps, which dominate total execution time for small batch sizes.

As batch size increased, this fixed overhead was amortized across more samples, leading to improved throughput and lower average latency per sample. This highlights that quantization benefits are workload-dependent and should not be evaluated using batch-1 latency alone.

---

## 3. Batching behavior revealed GPU underutilization in eager execution

ONNXRuntime CUDA execution showed strong throughput scaling with batch size, reaching over 1000 samples/sec at batch=16. This scaling behavior suggests that the GPU was underutilized during PyTorch eager execution, particularly at higher batch sizes.

The improved batching efficiency in ONNXRuntime indicates better kernel fusion, memory scheduling, and execution planning, allowing the GPU to operate closer to its practical utilization limits for this workload.

---

## 4. TensorRT EP did not outperform CUDA EP on GTX 1650 for this model

TensorRT Execution Provider (Stage 4) produced higher latency and lower throughput compared to ONNXRuntime CUDA EP on the GTX 1650 GPU. This behavior is consistent with hardware and workload characteristics rather than implementation issues.

Possible contributing factors include:
- Limited or no Tensor Core acceleration on GTX 1650 for FP16/INT8 workloads
- Reduced benefit from kernel fusion for a relatively lightweight model such as DistilBERT
- Additional engine build and shape-handling overhead when compared to already-optimized CUDA kernels

This observation reinforces that TensorRT performance gains are highly dependent on GPU architecture and model characteristics, and should be validated empirically rather than assumed.

---

## 5. Enabling FP16 or INT8 does not guarantee performance improvement

Enabling FP16 or INT8 execution paths in TensorRT does not guarantee improved performance. Precision selection is subject to hardware support, kernel availability, and runtime heuristics.

On this hardware, enabling FP16 and INT8 resulted in worse performance, indicating that reduced precision alone is insufficient to offset overheads when hardware acceleration paths are limited. This highlights the importance of benchmarking actual execution paths rather than relying on theoretical performance expectations.

---

## 6. Correct runtime configuration is critical for stability with dynamic shapes

ONNXRuntime initially produced a buffer reuse error when running multiple batch sizes within the same session. This was resolved by disabling memory pattern optimization and memory reuse.

This behavior demonstrates that dynamic input shapes combined with aggressive allocator reuse can lead to runtime instability if not configured correctly. Proper session configuration is essential when benchmarking across varying batch sizes in production-style inference pipelines.

---

## 7. Inference performance gains were driven by system-level decisions

Overall performance improvements in this pipeline were driven primarily by system-level decisions—runtime selection, execution model, batching strategy, and hardware awareness—rather than changes to model architecture or training.

This reinforces the distinction between model development and inference systems engineering, and highlights the importance of empirical measurement and hardware-aware optimization when deploying models in production environments.
