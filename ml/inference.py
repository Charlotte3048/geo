# ml/inference.py
import os
import torch
import numpy as np
from typing import List, Union

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from peft import PeftModel

from pathlib import Path

# ======================================================
# 1. Config
# ======================================================
BASE_MODEL_NAME = "bert-base-uncased"

# Adapter version can be switched via environment variable
BASE_DIR = Path(__file__).resolve().parent
LORA_ADAPTER_PATH = BASE_DIR / "artifacts" / "lora_adapter_v1"
BASE_MODEL_NAME = "bert-base-uncased"

MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sentiment label mapping (index -> human-readable)
ID2LABEL = {
    0: "strong_negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "strong_positive"
}


# ======================================================
# 2. Load Model & Tokenizer
# ======================================================
def load_model():
    """
    Load base model and attach LoRA adapter.
    """
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_PATH)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(ID2LABEL)
    )

    model = PeftModel.from_pretrained(
        base_model,
        LORA_ADAPTER_PATH
    )

    model.to(DEVICE)
    model.eval()

    return model, tokenizer


MODEL, TOKENIZER = load_model()


def load_baseline_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(ID2LABEL)
    )

    model.to(DEVICE)
    model.eval()
    return model, tokenizer


# ======================================================
# 3. Inference Utilities
# ======================================================
def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum(axis=-1, keepdims=True)


# ======================================================
# 4. Public Prediction API
# ======================================================
@torch.no_grad()
def _predict_with_model(
    model,
    tokenizer,
    texts: List[str],
    return_probs: bool = False
):
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = model(**inputs)
    logits = outputs.logits.cpu().numpy()
    probs = _softmax(logits)
    preds = probs.argmax(axis=1)

    results = []
    for i, idx in enumerate(preds):
        result = {
            "label": ID2LABEL[int(idx)],
            "confidence": float(probs[i][idx])
        }
        if return_probs:
            result["probs"] = {
                ID2LABEL[j]: float(probs[i][j])
                for j in range(len(ID2LABEL))
            }
        results.append(result)

    return results



# ======================================================
# 5. CLI Test (Optional)
# ======================================================

def compare_with_baseline(texts: List[str]):
    lora_model, lora_tokenizer = MODEL, TOKENIZER
    base_model, base_tokenizer = load_baseline_model()

    lora_preds = _predict_with_model(lora_model, lora_tokenizer, texts, return_probs=True)
    base_preds = _predict_with_model(base_model, base_tokenizer, texts, return_probs=True)

    comparisons = []

    for text, lp, bp in zip(texts, lora_preds, base_preds):
        comparisons.append({
            "text": text,
            "baseline_label": bp["label"],
            "baseline_conf": bp["confidence"],
            "lora_label": lp["label"],
            "lora_conf": lp["confidence"],
            "confidence_gain": lp["confidence"] - bp["confidence"],
            "label_changed": bp["label"] != lp["label"]
        })

    return comparisons


if __name__ == "__main__":
    samples = [
        "The food was absolutely terrible and the service was even worse.",
        "It was okay, nothing special but not bad either.",
        "Amazing experience! I would definitely come back again."
    ]

    results = compare_with_baseline(samples)

    for r in results:
        print("=" * 60)
        print(f"Text: {r['text']}")
        print(f"Baseline → {r['baseline_label']} ({r['baseline_conf']:.3f})")
        print(f"LoRA     → {r['lora_label']} ({r['lora_conf']:.3f})")
        print(f"Δ confidence: {r['confidence_gain']:+.3f}")
        print(f"Label changed: {r['label_changed']}")

