# ml/train.py
import os
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score


# ======================================================
# 1. Reproducibility
# ======================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# ======================================================
# 2. Config
# ======================================================
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 5
MAX_LENGTH = 256

TRAIN_SAMPLES = 8000
VAL_SAMPLES = 1000

OUTPUT_DIR = "ml/artifacts/lora_adapter_v1"


# ======================================================
# 3. Load Dataset (Yelp Review Full)
# ======================================================
dataset = load_dataset("yelp_review_full")

def preprocess_dataset(split, n_samples):
    """
    Yelp labels: 0–4  ->  1–5 stars
    We keep them as 0–4 for classification.
    """
    split = split.shuffle(seed=42).select(range(n_samples))
    return split.map(
        lambda x: {
            "text": x["text"],
            "label": int(x["label"])
        }
    )


train_dataset = preprocess_dataset(dataset["train"], TRAIN_SAMPLES)
val_dataset = preprocess_dataset(dataset["test"], VAL_SAMPLES)


# ======================================================
# 4. Tokenization
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)


# ======================================================
# 5. Model + LoRA
# ======================================================
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()


# ======================================================
# 6. Metrics
# ======================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


# ======================================================
# 7. Training Arguments
# ======================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    save_total_limit=1,
    report_to="none"
)


# ======================================================
# 8. Trainer
# ======================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)


# ======================================================
# 9. Train
# ======================================================
trainer.train()


# ======================================================
# 10. Save LoRA Adapter
# ======================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"LoRA adapter saved to {OUTPUT_DIR}")
