# Data Description & Selection Rationale

## 1. Dataset Overview

This project uses the **Yelp Review Polarity** dataset, a publicly available benchmark derived from the Yelp Reviews corpus.  
Each sample consists of a user-written review text and a corresponding sentiment label.

- **Source**: HuggingFace Datasets (`yelp_polarity`)
- **Task**: Binary sentiment classification
- **Input**: Free-form user review text
- **Output**: Sentiment label (Positive / Negative)

The dataset is widely adopted in sentiment analysis research and industrial prototyping, providing clean annotations and realistic review-style text.

---

## 2. Label Definition

The original Yelp ratings range from 1 to 5 stars and inherently encode **ordinal sentiment intensity** rather than simple binary polarity.

In this project, ratings are mapped to a **five-level sentiment scale** as follows:

| Original Rating | Sentiment Label        |
|-----------------|------------------------|
| 1 star          | Strong Negative        |
| 2 stars         | Negative               |
| 3 stars         | Neutral                |
| 4 stars         | Positive               |
| 5 stars         | Strong Positive        |

**Rationale**:
- Star ratings reflect graded user sentiment rather than binary opinions.
- Differentiating strong and mild sentiments better aligns with real-world business use cases, such as prioritizing severe complaints or identifying highly satisfied customers.
- Modeling sentiment intensity enables finer-grained analysis beyond simple positive/negative classification.

---

## 3. Modeling Considerations

Although binary sentiment classification is commonly used, this project adopts a multi-class sentiment formulation to preserve sentiment intensity information inherent in the rating system.

This choice allows the model to:
- Distinguish between extreme and moderate user opinions
- Capture more nuanced sentiment signals
- Support downstream decision-making scenarios that require different handling strategies for varying sentiment strengths

The trade-off is increased classification complexity, which is mitigated by using parameter-efficient fine-tuning with LoRA to maintain training efficiency.

---

## 4. Data Splitting

The dataset is split into training, validation, and test sets using a fixed random seed to ensure reproducibility.

- Split ratio: approximately 80 / 10 / 10
- Random seed is explicitly set during sampling and splitting.

**Considerations**:
- Random splitting is acceptable as Yelp reviews are independent user-generated texts.
- Data leakage is minimized by ensuring no overlap between splits.

---

## 5. Text Preprocessing

Minimal preprocessing is applied to preserve the original semantic and emotional content of the reviews.

- Text is tokenized using the tokenizer associated with the pre-trained base model.
- Maximum sequence length is capped (e.g., 128 or 256 tokens).
- No aggressive text cleaning or sentiment-specific heuristics are applied.

**Rationale**:
- Transformer-based models are robust to raw text.
- Over-processing may remove meaningful sentiment cues.
- Keeping preprocessing minimal improves reproducibility and deployment consistency.

---

## 6. Data Versioning & Reproducibility

To support reproducibility and experiment tracking:

- The sampled dataset is treated as a **versioned artifact**.
- Dataset version identifiers (e.g., `data_v1`) are logged alongside training runs.
- Random seeds and sampling parameters are recorded.

This design ensures that model performance comparisons are attributable to training or fine-tuning changes rather than hidden data variations.

---

## 7. Limitations & Assumptions

- The dataset represents English-language restaurant and business reviews only.
- Domain-specific sentiment (e.g., medical or financial text) is not covered.
- Results are intended to demonstrate workflow validity rather than domain generalization.

---

## 8. Summary

This dataset selection balances **realism, reproducibility, and efficiency**, making it suitable for demonstrating:
- Parameter-efficient fine-tuning with LoRA
- End-to-end training and deployment workflows
- Practical trade-offs between data scale and system validation
