import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
from tqdm import tqdm
import random

# =========================
# 配置区（根据你的项目）
# =========================

BASE_MODEL_NAME = "bert-base-uncased"
LORA_ADAPTER_PATH = "artifacts/lora_adapter_v1"  # 相对项目根目录
NUM_LABELS = 5
NUM_SAMPLES = 500
SEED = 42
MAX_LENGTH = 256

LABEL2ID = {
    "strong_negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "strong_positive": 4
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# =========================
# 工具函数
# =========================

def load_models():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    print("Loading baseline model...")
    baseline_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS
    )
    baseline_model.eval()

    print("Loading LoRA model...")
    lora_base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS
    )
    lora_model = PeftModel.from_pretrained(
        lora_base,
        LORA_ADAPTER_PATH
    )
    lora_model.eval()

    return tokenizer, baseline_model, lora_model


def predict(model, tokenizer, text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).squeeze()
        pred_id = torch.argmax(probs).item()
    return pred_id


def load_test_samples(n=500, seed=42):
    dataset = load_dataset("yelp_review_full", split="test")
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(n))
    return dataset


# =========================
# 主评估逻辑
# =========================

def evaluate():
    tokenizer, baseline_model, lora_model = load_models()
    dataset = load_test_samples(NUM_SAMPLES, SEED)

    y_true = []
    y_pred_base = []
    y_pred_lora = []

    label_shift = defaultdict(int)

    print(f"\nRunning evaluation on {NUM_SAMPLES} Yelp test samples...\n")

    for item in tqdm(dataset):
        text = item["text"]
        true_label = int(item["label"])  # Yelp: 0–4

        base_pred = predict(baseline_model, tokenizer, text)
        lora_pred = predict(lora_model, tokenizer, text)

        y_true.append(true_label)
        y_pred_base.append(base_pred)
        y_pred_lora.append(lora_pred)

        if base_pred != lora_pred:
            label_shift[(base_pred, lora_pred)] += 1

    # 计算指标
    base_acc = accuracy_score(y_true, y_pred_base)
    lora_acc = accuracy_score(y_true, y_pred_lora)
    base_f1 = f1_score(y_true, y_pred_base, average="macro")
    lora_f1 = f1_score(y_true, y_pred_lora, average="macro")

    # =========================
    # 输出结果
    # =========================

    print("\n===== Evaluation Results (Yelp Test) =====")
    print(f"Baseline Accuracy : {base_acc:.4f}")
    print(f"LoRA Accuracy     : {lora_acc:.4f}")
    print(f"Baseline Macro-F1 : {base_f1:.4f}")
    print(f"LoRA Macro-F1     : {lora_f1:.4f}")

    print("\n===== Label Shift (Baseline → LoRA) =====")
    for (b, l), cnt in sorted(label_shift.items(), key=lambda x: -x[1]):
        print(f"{ID2LABEL[b]:>15} → {ID2LABEL[l]:<15} : {cnt}")

    print("\n===== Summary =====")
    print(f"Accuracy Gain     : {lora_acc - base_acc:+.4f}")
    print(f"Macro-F1 Gain     : {lora_f1 - base_f1:+.4f}")


# =========================
# 入口
# =========================

if __name__ == "__main__":
    evaluate()
