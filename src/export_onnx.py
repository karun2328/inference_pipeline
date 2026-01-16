# export_onnx.py
"""
STAGE 3 (GPU) — Export PyTorch model to ONNX

Output: distilbert_sst2.onnx
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
ONNX_PATH = "distilbert_sst2.onnx"
MAX_LENGTH = 128

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    # Dummy input for export
    dummy_text = ["Exporting ONNX is useful for production inference."]
    encoded = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # DistilBERT needs input_ids + attention_mask
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Export with dynamic batch and sequence length
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=18,
    )

    print(f" Exported ONNX model to: {ONNX_PATH}")

if __name__ == "__main__":
    main()
