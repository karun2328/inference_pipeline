# infer_onnx_trt_ep.py
"""
STAGE 4 — TensorRT (via ONNXRuntime TensorRT Execution Provider)

What this does:
- Uses ONNXRuntime + TensorrtExecutionProvider (TRT EP)
- Builds a TensorRT engine the first time (slower first run)
- Caches the engine to disk
- Benchmarks p50/p95/avg latency + throughput

Run:
  python infer_onnx_trt_ep.py
"""

import os
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
    engine_cache_dir: str = "trt_engine_cache"


def build_inputs_np(tokenizer, texts: List[str], max_length: int) -> Dict[str, np.ndarray]:
    enc = tokenizer(
        texts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }


def bench(session: ort.InferenceSession, inputs: Dict[str, np.ndarray], warmup_runs: int, timed_runs: int) -> Dict[str, float]:
    # Warmup (not measured). First ever run may also build the TRT engine (slow).
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
    os.makedirs(cfg.engine_cache_dir, exist_ok=True)

    print("Available providers:", ort.get_available_providers())

    # IMPORTANT: avoid dynamic-shape buffer reuse issues
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    so.enable_mem_reuse = False
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # TensorRT EP options (engine cache is key)
    trt_options = {
        "trt_engine_cache_enable": 1,
        "trt_engine_cache_path": cfg.engine_cache_dir,
        # optional knobs (safe defaults):
        "trt_fp16_enable": 0,      
        "trt_int8_enable": 1,       
    }

    providers = [
        ("TensorrtExecutionProvider", trt_options),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    session = ort.InferenceSession(cfg.onnx_path, sess_options=so, providers=providers)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    base_texts = [
        "I love this movie. It was amazing and inspiring!",
        "This was the worst film I have ever seen.",
        "It was okay, not great, not terrible.",
        "The acting was brilliant and the story was engaging.",
    ]

    # Sanity check
    inputs = build_inputs_np(tokenizer, base_texts, cfg.max_length)
    logits = session.run(None, inputs)[0]
    preds = np.argmax(logits, axis=-1).tolist()
    label_map = {0: "NEG", 1: "POS"}

    print("\n==== Quick sanity check (predictions) ====")
    for t, p in zip(base_texts, preds):
        print(f"{label_map[p]}  | {t}")
    print("=========================================\n")

    print("==== Stage 4: TensorRT EP Benchmark ====")
    print(f"ONNX model: {cfg.onnx_path}")
    print(f"Engine cache dir: {cfg.engine_cache_dir}")
    print("NOTE: First run may be slower because TensorRT builds the engine.\n")

    for bs in cfg.batch_sizes:
        texts = (base_texts * ((bs + len(base_texts) - 1) // len(base_texts)))[:bs]
        batch_inputs = build_inputs_np(tokenizer, texts, cfg.max_length)

        r = bench(session, batch_inputs, cfg.warmup_runs, cfg.timed_runs)
        print(
            f"TRT-EP | Batch={bs:>3} | p50={r['p50_ms']:.2f} ms | "
            f"p95={r['p95_ms']:.2f} ms | avg={r['avg_ms']:.2f} ms | "
            f"throughput={r['throughput']:.2f} samples/s"
        )

    print("\n Stage 4 complete: TensorRT EP benchmark done.\n")
    print("Next: Stage 5 Packaging (README + comparison table + story).")


if __name__ == "__main__":
    main()
