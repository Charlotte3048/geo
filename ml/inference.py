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


# ======================================================
# 1. Config
# ======================================================
BASE_MODEL_NAME = "bert-base-uncased"

# Adapter version can be switched via environment variable
LORA_ADAPTER_PATH = os.getenv(
    "LORA_ADAPTER_PATH",
    "ml/artifacts/lora_adapter_v1"
)

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
def predict(
    texts: Union[str, List[str]],
    return_probs: bool = False
):
    """
    Run sentiment inference.

    Args:
        texts: Single text or list of texts
        return_probs: Whether to return full probability distribution

    Returns:
        List of prediction dicts
    """
    if isinstance(texts, str):
        texts = [texts]

    inputs = TOKENIZER(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = MODEL(**inputs)
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
if __name__ == "__main__":
    samples = [
        "The food was absolutely terrible and the service was even worse.",
        "It was okay, nothing special but not bad either.",
        "Amazing experience! I would definitely come back again."
    ]

    predictions = predict(samples, return_probs=True)

    for text, pred in zip(samples, predictions):
        print("=" * 60)
        print(f"Text: {text}")
        print(f"Prediction: {pred}")
