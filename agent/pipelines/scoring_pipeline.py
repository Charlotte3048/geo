# agent/pipelines/scoring_pipeline.py
import os, json, yaml
from domestic.analyze_results_domestic import calculate_scores  # 先直接复用你现成的
from domestic.sentiment.sentiment_analyzer import SentimentAnalyzer

# ✅ 全局只初始化一次（进程级）
_SENTIMENT_ANALYZER = SentimentAnalyzer.get_instance()


def run_scoring_pipeline(category: str, results_path: str) -> str:
    # 1) results
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results不存在：{results_path}")

    # 2) brand config：你项目里通常是 domestic/config/brand_dictionary_{category}.yaml
    brand_cfg_path = f"domestic/config/brand_dictionary_{category}.yaml"
    if not os.path.exists(brand_cfg_path):
        raise FileNotFoundError(f"品牌词典不存在：{brand_cfg_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    with open(brand_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    brand_dictionary = cfg["brand_dictionary"]
    whitelist = set(cfg.get("brands_whitelist", []))
    weights = cfg.get("weights", {})

    # 3) score
    final_scores, question_level = calculate_scores(
        data_list,
        brand_dictionary,
        whitelist,
        weights,
        analyzer=_SENTIMENT_ANALYZER,  # ✅ 核心改动
        return_question_level=True
    )

    out = {
        "category": category,
        "meta": {
            "results_path": results_path,
            "brand_cfg_path": brand_cfg_path,
            "total_items": len(data_list),
        },
        "brand_scores": final_scores,
        "question_level": question_level
    }

    # 4) write
    os.makedirs("domestic/scores", exist_ok=True)
    out_path = f"domestic/scores/scores_{category}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out_path
