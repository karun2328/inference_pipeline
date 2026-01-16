# infer_fp32.py
"""
STAGE 1 (Inference Basics) — FP32 Baseline

What this script does (Stage 1 complete):
1) Loads a small pretrained Transformer model (DistilBERT SST-2)
2) Runs inference ONLY (no training)
3) Measures latency + throughput for different batch sizes
4) Runs on GPU if available, else CPU
5) Prints predicted labels for a small sample

Run:
  python infer_fp32.py

Notes:
- This is your FP32 baseline. Stage 2 will add INT8 quantization and compare.
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataclasses import dataclass
from typing import List,Dict,Any,Tuple

@dataclass
class BenchConfig:
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    # Test different batch sizes. You can add more later.
    batch_sizes: Tuple[int, ...] = (1, 4, 8, 16)
    # "Sequence length" is controlled by tokenizer padding/truncation.
    max_length: int = 128
    # Warmup runs (not measured) to stabilize GPU/CPU caches
    warmup_runs: int = 10
    # Measured runs
    timed_runs: int = 50
    # Use GPU if available
    prefer_gpu: bool = True

def pick_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def model_info(device: torch.device) -> None:
    print("==== Environment ====")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {device}")
    if device.type == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
        # Optional: show current memory usage baseline (may be 0 if nothing allocated yet)
        torch.cuda.synchronize()
    print("=====================\n")

def build_inputs(
        tokenizer: AutoTokenizer,
        texts: List[str],
        max_length: int,
        device: torch.device
) -> Dict[str, torch.Tensor]:

    encoded=tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    return {k: v.to(device) for k, v in encoded.items()}

@torch.no_grad()
def run_inference(
    model: AutoModelForSequenceClassification,
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    One forward pass. Returns logits.
    """
    outputs = model(**inputs)
    # Make sure ops are finished before timing ends on GPU
    if device.type == "cuda":
        torch.cuda.synchronize()
    return outputs.logits


def time_forward_pass(
    model: AutoModelForSequenceClassification,
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
    warmup_runs: int,
    timed_runs: int,
) -> Dict[str, float]:
    """
    Measures:
    - p50 latency (ms)
    - p95 latency (ms)
    - avg latency (ms)
    - throughput (samples/sec)
    """
    # Warmup runs
    for _ in range(warmup_runs):
        _ = run_inference(model, inputs, device)

    # Timed runs
    latencies = []
    batch_size = next(iter(inputs.values())).shape[0]  # first tensor's batch dim

    for _ in range(timed_runs):
        start = time.perf_counter()
        _ = run_inference(model, inputs, device)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # ms

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95) - 1]
    avg = sum(latencies) / len(latencies)

    # Throughput: samples/sec based on average latency for the batch
    # avg_ms per batch => avg_s per batch => batch_size / avg_s
    throughput = batch_size / (avg / 1000.0)

    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "avg_ms": avg,
        "throughput_samples_per_sec": throughput,
    }


def pretty_print_bench(batch_size: int, results: Dict[str, float]) -> None:
    print(
        f"Batch={batch_size:>3} | "
        f"p50={results['p50_ms']:.2f} ms | "
        f"p95={results['p95_ms']:.2f} ms | "
        f"avg={results['avg_ms']:.2f} ms | "
        f"throughput={results['throughput_samples_per_sec']:.2f} samples/s"
    )


def main() -> None:
    cfg = BenchConfig()

    device = pick_device(cfg.prefer_gpu)
    model_info(device)

    print("Loading tokenizer + model (FP32 baseline)...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
    model.eval()
    model.to(device)

    # A small set of texts; we will replicate to form batches
    base_texts = [
        "I love this movie. It was amazing and inspiring!",
        "This was the worst film I have ever seen.",
        "It was okay, not great, not terrible.",
        "The acting was brilliant and the story was engaging.",
    ]

    print("\n==== Quick sanity check (predictions) ====")
    inputs = build_inputs(tokenizer, base_texts, cfg.max_length, device)
    logits = run_inference(model, inputs, device)
    preds = torch.argmax(logits, dim=-1).tolist()

    # SST-2 label mapping: 0=NEGATIVE, 1=POSITIVE
    label_map = {0: "NEG", 1: "POS"}
    for t, p in zip(base_texts, preds):
        print(f"{label_map[p]}  | {t}")
    print("=========================================\n")

    print("==== Benchmark: FP32 Inference (no training) ====")
    print(f"Model: {cfg.model_name}")
    print(f"Max length: {cfg.max_length}")
    print(f"Warmup runs: {cfg.warmup_runs} | Timed runs: {cfg.timed_runs}\n")

    for bs in cfg.batch_sizes:
        # Create a batch by repeating texts until we hit batch size
        texts = (base_texts * ((bs + len(base_texts) - 1) // len(base_texts)))[:bs]
        batch_inputs = build_inputs(tokenizer, texts, cfg.max_length, device)

        results = time_forward_pass(
            model=model,
            inputs=batch_inputs,
            device=device,
            warmup_runs=cfg.warmup_runs,
            timed_runs=cfg.timed_runs,
        )
        pretty_print_bench(bs, results)

    print("\n Stage 1 complete: You now have an FP32 inference baseline.")
    print("Next (Stage 2): INT8 quantization + compare FP32 vs INT8.\n")


if __name__ == "__main__":
    main()