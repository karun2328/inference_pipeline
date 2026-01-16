# infer_onnx_gpu.py
"""
STAGE 3 (GPU) — ONNXRuntime inference on GPU + benchmarking

What you get:
- GPU inference via ONNXRuntime (CUDAExecutionProvider)
- p50 / p95 / avg latency
- throughput for batch sizes
- quick sanity check predictions
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


@dataclass
class BenchConfig:
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    onnx_path: str = "distilbert_sst2.onnx"
    batch_sizes: Tuple[int, ...] = (1, 4, 8, 16)
    max_length: int = 128
    warmup_runs: int = 10
    timed_runs: int = 50


def build_inputs_np(tokenizer, texts: List[str], max_length: int) -> Dict[str, np.ndarray]:
    enc = tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    # ONNXRuntime expects int64 for ids/mask
    return {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }


def pick_providers() -> List[str]:
    avail = ort.get_available_providers()
    print("Available providers:", avail)
    # Prefer CUDA if available
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def bench(session: ort.InferenceSession, inputs: Dict[str, np.ndarray], warmup_runs: int, timed_runs: int) -> Dict[str, float]:
    # Warmup
    for _ in range(warmup_runs):
        _ = session.run(None, inputs)

    lat = []
    bs = inputs["input_ids"].shape[0]

    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = session.run(None, inputs)
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000.0)

    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted) // 2]
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1]
    avg = sum(lat) / len(lat)
    throughput = bs / (avg / 1000.0)

    return {"p50_ms": p50, "p95_ms": p95, "avg_ms": avg, "throughput": throughput}


def main():
    cfg = BenchConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    providers = pick_providers()
    so = ort.SessionOptions()
    so.enable_mem_pattern = False          # IMPORTANT: prevents batch-shape reuse crash
    so.enable_mem_reuse = False            # extra safety for dynamic shapes
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


    session = ort.InferenceSession(cfg.onnx_path, sess_options=so, providers=providers)


    print("\n==== Stage 3 GPU: ONNXRuntime Benchmark ====")
    print(f"ONNX model: {cfg.onnx_path}")
    print(f"Warmup: {cfg.warmup_runs} | Timed: {cfg.timed_runs}")
    print(f"Max length: {cfg.max_length}\n")

    base_texts = [
        "I love this movie. It was amazing and inspiring!",
        "This was the worst film I have ever seen.",
        "It was okay, not great, not terrible.",
        "The acting was brilliant and the story was engaging.",
    ]

    # Sanity check predictions (argmax over logits)
    inputs = build_inputs_np(tokenizer, base_texts, cfg.max_length)
    logits = session.run(None, inputs)[0]
    preds = np.argmax(logits, axis=-1).tolist()
    label_map = {0: "NEG", 1: "POS"}

    print("==== Quick sanity check (predictions) ====")
    for t, p in zip(base_texts, preds):
        print(f"{label_map[p]}  | {t}")
    print("=========================================\n")

    for bs in cfg.batch_sizes:
        texts = (base_texts * ((bs + len(base_texts) - 1) // len(base_texts)))[:bs]
        batch_inputs = build_inputs_np(tokenizer, texts, cfg.max_length)

        r = bench(session, batch_inputs, cfg.warmup_runs, cfg.timed_runs)
        print(
            f"ONNX | Batch={bs:>3} | p50={r['p50_ms']:.2f} ms | "
            f"p95={r['p95_ms']:.2f} ms | avg={r['avg_ms']:.2f} ms | "
            f"throughput={r['throughput']:.2f} samples/s"
        )

    print("\n Stage 3 complete: ONNXRuntime inference benchmark done.")
    print("Next: Stage 4 TensorRT (ONNX → TensorRT engine) for NVIDIA-style optimization.\n")


if __name__ == "__main__":
    main()
