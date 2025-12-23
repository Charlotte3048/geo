# domestic/sentiment_analyzer.py
"""
BERT情感分析模块
用于对品牌相关句子进行五级情感分析
"""
import os
import sys
import torch
import numpy as np
from typing import List, Dict
from pathlib import Path
import warnings

# 忽略一些警告
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# 设置环境变量，避免某些库的兼容性问题
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel
except ImportError as e:
    print(f"❌ 依赖库导入失败: {e}")
    print("请运行: pip install transformers peft torch")
    raise

# ======================================================
# 配置
# ======================================================
BASE_MODEL_NAME = "bert-base-uncased"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LORA_ADAPTER_PATH = PROJECT_ROOT / "ml" / "artifacts" / "lora_adapter_v1"
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 情感标签映射
ID2LABEL = {
    0: "strong_negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "strong_positive"
}

# 情感分数映射 (0-100分制)
SENTIMENT_SCORES = {
    "strong_negative": 0,
    "negative": 25,
    "neutral": 50,
    "positive": 75,
    "strong_positive": 100
}


# ======================================================
# 模型加载（单例模式）
# ======================================================
class SentimentAnalyzer:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        print(f"🔄 正在加载BERT情感分析模型...")
        print(f"   设备: {DEVICE}")
        print(f"   Base model: {BASE_MODEL_NAME}")
        print(f"   Adapter路径: {LORA_ADAPTER_PATH}")

        try:
            # 1️⃣ tokenizer 一定来自 base model
            self._tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME,
                local_files_only=False
            )

            # 2️⃣ 加载 base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL_NAME,
                num_labels=len(ID2LABEL),
                torch_dtype=torch.float32
            )

            # 3️⃣ 加载 LoRA adapter（本地路径是完全 OK 的）
            self._model = PeftModel.from_pretrained(
                base_model,
                str(LORA_ADAPTER_PATH),
                torch_dtype=torch.float32
            )

            self._model.to(DEVICE)
            self._model.eval()

            print("✅ BERT模型加载成功\n")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """计算softmax概率"""
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    @torch.no_grad()
    def predict(self, texts: List[str], return_probs: bool = False) -> List[Dict]:
        """
        对文本列表进行情感分析

        Args:
            texts: 待分析的文本列表
            return_probs: 是否返回所有类别的概率分布

        Returns:
            包含情感标签、置信度和分数的字典列表
        """
        if not texts:
            return []

        try:
            # Tokenize
            inputs = self._tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)

            # 推理
            outputs = self._model(**inputs)
            logits = outputs.logits.cpu().numpy()
            probs = self._softmax(logits)
            preds = probs.argmax(axis=1)

            # 构建结果
            results = []
            for i, idx in enumerate(preds):
                label = ID2LABEL[int(idx)]
                result = {
                    "label": label,
                    "confidence": float(probs[i][idx]),
                    "score": SENTIMENT_SCORES[label]  # 0-100分制
                }
                if return_probs:
                    result["probs"] = {
                        ID2LABEL[j]: float(probs[i][j])
                        for j in range(len(ID2LABEL))
                    }
                results.append(result)

            return results

        except Exception as e:
            print(f"⚠️  推理过程出错: {e}")
            # 返回默认中性结果
            return [{
                "label": "neutral",
                "confidence": 0.0,
                "score": 50
            } for _ in texts]

    def analyze_sentence(self, sentence: str) -> Dict:
        """
        分析单个句子的情感

        Args:
            sentence: 待分析的句子

        Returns:
            包含情感标签、置信度和分数的字典
        """
        results = self.predict([sentence])
        return results[0] if results else {
            "label": "neutral",
            "confidence": 0.0,
            "score": 50
        }


# ======================================================
# 全局单例
# ======================================================
_analyzer = None


def get_sentiment_analyzer():
    """获取情感分析器单例"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer


# ======================================================
# 便捷函数
# ======================================================
def analyze_brand_sentiment(sentences: List[str]) -> float:
    """
    分析品牌相关句子的平均情感得分

    Args:
        sentences: 包含品牌的句子列表

    Returns:
        平均情感得分 (0-100)
    """
    if not sentences:
        return 50.0  # 默认中性

    try:
        analyzer = get_sentiment_analyzer()
        results = analyzer.predict(sentences)

        # 计算平均分数
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 50.0

        return avg_score
    except Exception as e:
        print(f"⚠️  情感分析失败: {e}")
        return 50.0  # 返回默认中性分数


# ======================================================
# 测试代码
# ======================================================
if __name__ == "__main__":
    # 测试样例
    test_sentences = [
        "This brand is absolutely amazing and I highly recommend it!",
        "The product quality is terrible and customer service is even worse.",
        "It's okay, nothing special but not bad either.",
        "Best purchase I've ever made! Will definitely buy again.",
        "Completely disappointed with this brand."
    ]

    print("\n" + "=" * 70)
    print("情感分析模块测试")
    print("=" * 70 + "\n")

    try:
        analyzer = get_sentiment_analyzer()
        results = analyzer.predict(test_sentences, return_probs=True)

        print("测试结果:")
        print("-" * 70)

        for text, result in zip(test_sentences, results):
            print(f"\n文本: {text}")
            print(f"情感: {result['label']}")
            print(f"置信度: {result['confidence']:.3f}")
            print(f"得分: {result['score']}/100")
            if "probs" in result:
                print("概率分布:")
                for label, prob in result["probs"].items():
                    print(f"  {label:20s}: {prob:.3f}")

        print("\n" + "=" * 70)
        print("✅ 测试完成")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
