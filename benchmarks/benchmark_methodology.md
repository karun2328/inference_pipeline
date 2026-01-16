# Benchmark Methodology

This document describes the methodology used to benchmark inference latency and throughput across different execution paths.

## Metrics
- Latency: measured in milliseconds (ms)
  - p50 (median)
  - p95 (tail latency)
  - average latency
- Throughput: samples processed per second

## Measurement Procedure
- Each benchmark consists of:
  - Warmup runs: 10 iterations (not measured)
  - Timed runs: 50 iterations
- Timing measured using high-resolution wall-clock timers
- First-run effects (e.g., engine build, kernel caching) are excluded via warmup

## Batch Sizes
Benchmarks were run for batch sizes:
- 1, 4, 8, 16

This range captures:
- batch-1 latency sensitivity
- batching efficiency
- GPU utilization scaling

## Input Configuration
- Model: distilbert-base-uncased-finetuned-sst-2-english
- Max sequence length: 128 tokens
- Inputs padded and truncated consistently across all stages

## Fairness Across Stages
To ensure fair comparison:
- The same model weights were used across all stages
- Numerical precision was kept constant unless explicitly stated (e.g., INT8 CPU)
- Identical batch sizes and input distributions were used
- Each stage was benchmarked in isolation

## Notes on Variability
- Results are hardware-specific
- GPU behavior depends on architecture, available acceleration units, and driver/runtime versions
- Absolute numbers may vary across systems; relative trends are the primary focus
