# infer_int8_cpu.py
"""
STAGE 2 — INT8 Quantization Baseline (CPU)

Why CPU?
- PyTorch dynamic quantization primarily accelerates CPU inference for Linear layers.
- GPU INT8 gains are usually realized via ONNXRuntime/TensorRT (Stage 3/4).

What this script does:
1) Loads DistilBERT (FP32)
2) Benchmarks FP32 on CPU
3) Applies dynamic INT8 quantization (Linear layers)
4) Benchmarks INT8 on CPU
5) Compares latency + throughput

Run:
  python infer_int8_cpu.py
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class BenchConfig:
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    batch_sizes: Tuple[int, ...] = (1, 4, 8, 16)
    max_length: int = 128
    warmup_runs: int = 10
    timed_runs: int = 50


def build_inputs(tokenizer, texts: List[str], max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def run_inference(model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(**inputs)
    return out.logits


def bench(model, inputs: Dict[str, torch.Tensor], warmup_runs: int, timed_runs: int) -> Dict[str, float]:
    # Warmup
    for _ in range(warmup_runs):
        _ = run_inference(model, inputs)

    lat = []
    bs = next(iter(inputs.values())).shape[0]

    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = run_inference(model, inputs)
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000.0)

    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted) // 2]
    p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1]
    avg = sum(lat) / len(lat)
    throughput = bs / (avg / 1000.0)

    return {"p50_ms": p50, "p95_ms": p95, "avg_ms": avg, "throughput": throughput}


def print_row(tag: str, bs: int, r: Dict[str, float]) -> None:
    print(
        f"{tag:<6} | Batch={bs:>3} | p50={r['p50_ms']:.2f} ms | "
        f"p95={r['p95_ms']:.2f} ms | avg={r['avg_ms']:.2f} ms | "
        f"throughput={r['throughput']:.2f} samples/s"
    )


def main() -> None:
    cfg = BenchConfig()
    device = torch.device("cpu")

    print("==== Stage 2A: CPU INT8 Dynamic Quantization ====")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    base_texts = [
        "I love this movie. It was amazing and inspiring!",
        "This was the worst film I have ever seen.",
        "It was okay, not great, not terrible.",
        "The acting was brilliant and the story was engaging.",
    ]

    # Load FP32 model on CPU
    fp32_model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
    fp32_model.eval()
    fp32_model.to(device)

    # Quantize (dynamic) — targets Linear layers
    int8_model = torch.ao.quantization.quantize_dynamic(
        fp32_model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    int8_model.eval()
    int8_model.to(device)

    print("Sanity check predictions (FP32 vs INT8):")
    inputs = build_inputs(tokenizer, base_texts, cfg.max_length, device)

    fp32_logits = run_inference(fp32_model, inputs)
    int8_logits = run_inference(int8_model, inputs)

    fp32_preds = torch.argmax(fp32_logits, dim=-1).tolist()
    int8_preds = torch.argmax(int8_logits, dim=-1).tolist()

    label_map = {0: "NEG", 1: "POS"}
    for t, p1, p2 in zip(base_texts, fp32_preds, int8_preds):
        print(f"FP32={label_map[p1]} | INT8={label_map[p2]} | {t}")

    print("\n==== Benchmark (CPU) ====")
    print(f"Model: {cfg.model_name}")
    print(f"Max length: {cfg.max_length}")
    print(f"Warmup: {cfg.warmup_runs} | Timed: {cfg.timed_runs}\n")

    for bs in cfg.batch_sizes:
        texts = (base_texts * ((bs + len(base_texts) - 1) // len(base_texts)))[:bs]
        batch_inputs = build_inputs(tokenizer, texts, cfg.max_length, device)

        r_fp32 = bench(fp32_model, batch_inputs, cfg.warmup_runs, cfg.timed_runs)
        r_int8 = bench(int8_model, batch_inputs, cfg.warmup_runs, cfg.timed_runs)

        print_row("FP32", bs, r_fp32)
        print_row("INT8", bs, r_int8)
        print("-" * 90)

    print("\n Stage 2A complete: You have CPU FP32 vs INT8 benchmarks.")
    print("Next: Stage 3 will move INT8 to GPU using ONNXRuntime/TensorRT.\n")


if __name__ == "__main__":
    main()
