import streamlit as st
import pandas as pd
import time
from collections import defaultdict

# --- 复用 analyze_results.py 中的核心代码 ---
# 为了让 dashboard.py 能独立运行，我们将分析逻辑直接整合进来。
# (在此处粘贴 analyze_results.py 中的函数和配置)

import json
import os
import re
import math

# --- 1. 配置 (从 analyze_results.py 复制) ---
RESULTS_FILE = "results.json"
BRAND_DICTIONARY = {
    # ... (在此处粘贴您已有的完整词典)
}
CHINESE_BRANDS_WHITELIST = {
    # ... (在此处粘贴您已有的完整白名单)
}
WEIGHTS_V3 = {
    "visibility": 20, "mention_rate": 20, "ai_ranking": 20,
    "ref_depth": 15, "mind_share": 15, "competitiveness": 10,
}


# --- 2. 核心分析函数 (从 analyze_results.py 复制) ---
def normalize_score(value, max_value, min_value=0, scale=100):
    if max_value == min_value: return scale if value > 0 else 0
    log_value = math.log1p(value)
    log_max = math.log1p(max_value)
    return (log_value / log_max) * scale if log_max > 0 else 0


def analyze_single_answer(answer_text: str, references: list, brand_map: dict):
    raw_metrics = defaultdict(
        lambda: {"mentioned": 0, "first_pos": float('inf'), "is_strong": 0, "ref_count": 0, "mention_count": 0})
    answer_lower = answer_text.lower()
    for std_brand, aliases in brand_map.items():
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower in answer_lower:
                raw_metrics[std_brand]["mentioned"] = 1
                raw_metrics[std_brand]["mention_count"] += answer_lower.count(alias_lower)
                try:
                    pos = answer_lower.index(alias_lower)
                    if pos < raw_metrics[std_brand]["first_pos"]: raw_metrics[std_brand]["first_pos"] = pos
                except ValueError:
                    pass
    sentences = re.split(r'[。\n]', answer_text)
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for brand, metrics in raw_metrics.items():
            if metrics["mentioned"] and brand.lower() in sentence_lower:
                if any(word in sentence_lower for word in
                       ["首选", "最佳", "强烈推荐", "best", "top pick", "most recommended"]): raw_metrics[brand][
                    "is_strong"] = 1
    if references:
        for brand, metrics in raw_metrics.items():
            if metrics["mentioned"]:
                for ref in references:
                    if brand.lower() in ref.lower(): raw_metrics[brand]["ref_count"] += 1
    return raw_metrics


# --- 3. 数据处理与缓存 ---
@st.cache_data(ttl=600)  # 缓存数据10分钟，避免每次刷新都重新计算
def get_ranking_data():
    """执行完整的分析流程并返回格式化的数据"""
    try:
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return None

    category_raw_metrics = defaultdict(lambda: defaultdict(
        lambda: {"total_mentions": 0, "first_pos_sum": 0, "strong_recommend_count": 0, "total_ref_count": 0,
                 "mention_in_answers": 0}))
    total_brand_mentions_across_all = 0
    for item in data:
        category = item.get("category", "Uncategorized")
        answer = item.get("response", {}).get("answer", "")
        references = item.get("response", {}).get("references", [])
        if not answer: continue
        answer_metrics = analyze_single_answer(answer, references, BRAND_DICTIONARY)
        for brand, metrics in answer_metrics.items():
            if brand in CHINESE_BRANDS_WHITELIST:
                cat_brand_metrics = category_raw_metrics[category][brand]
                cat_brand_metrics["total_mentions"] += metrics["mention_count"]
                if metrics["first_pos"] != float('inf'): cat_brand_metrics["first_pos_sum"] += metrics["first_pos"]
                cat_brand_metrics["strong_recommend_count"] += metrics["is_strong"]
                cat_brand_metrics["total_ref_count"] += metrics["ref_count"]
                cat_brand_metrics["mention_in_answers"] += 1
                total_brand_mentions_across_all += metrics["mention_count"]

    final_data_for_df = []
    for category, brands_metrics in category_raw_metrics.items():
        if not brands_metrics: continue
        max_mentions = max(m["total_mentions"] for m in brands_metrics.values()) if brands_metrics else 0
        min_pos_avg = min(m["first_pos_sum"] / m["mention_in_answers"] for m in brands_metrics.values() if
                          m["mention_in_answers"] > 0) if any(
            m["mention_in_answers"] > 0 for m in brands_metrics.values()) else 0
        max_strong = max(m["strong_recommend_count"] for m in brands_metrics.values()) if brands_metrics else 0
        max_refs = max(m["total_ref_count"] for m in brands_metrics.values()) if brands_metrics else 0

        for brand, metrics in brands_metrics.items():
            scores = {}
            avg_pos = metrics["first_pos_sum"] / metrics["mention_in_answers"] if metrics[
                                                                                      "mention_in_answers"] > 0 else float(
                'inf')
            scores["visibility"] = (1 - normalize_score(avg_pos, min_pos_avg * 5, min_pos_avg) / 100) * WEIGHTS_V3[
                "visibility"]
            scores["mention_rate"] = normalize_score(metrics["total_mentions"], max_mentions) / 100 * WEIGHTS_V3[
                "mention_rate"]
            scores["ai_ranking"] = normalize_score(metrics["strong_recommend_count"], max_strong) / 100 * WEIGHTS_V3[
                "ai_ranking"]
            scores["ref_depth"] = normalize_score(metrics["total_ref_count"], max_refs) / 100 * WEIGHTS_V3["ref_depth"]
            mind_share_ratio = metrics[
                                   "total_mentions"] / total_brand_mentions_across_all if total_brand_mentions_across_all > 0 else 0
            scores["mind_share"] = mind_share_ratio * 100 * (WEIGHTS_V3["mind_share"] / 5)
            comp_score_avg = (scores["visibility"] + scores["mention_rate"] + scores["ai_ranking"]) / (
                        WEIGHTS_V3["visibility"] + WEIGHTS_V3["mention_rate"] + WEIGHTS_V3["ai_ranking"]) if (
                                                                                                                         WEIGHTS_V3[
                                                                                                                             "visibility"] +
                                                                                                                         WEIGHTS_V3[
                                                                                                                             "mention_rate"] +
                                                                                                                         WEIGHTS_V3[
                                                                                                                             "ai_ranking"]) > 0 else 0
            scores["competitiveness"] = comp_score_avg * WEIGHTS_V3["competitiveness"]
            total_score = sum(scores.values())

            final_data_for_df.append({
                "品类": category,
                "品牌名称": brand,
                "品牌指数": total_score,
                "总提及次数": metrics["total_mentions"],
                "出现次数": metrics["mention_in_answers"],
                "最后更新时间": time.strftime('%Y/%m/%d %H:%M:%S')
            })
    return pd.DataFrame(final_data_for_df)


# --- 4. Streamlit 界面渲染 ---
st.set_page_config(page_title="GenAI品牌排行榜", layout="wide")

st.title("🤖 GenAI心智占有率 - 品牌排行榜")

# 加载数据c
df = get_ranking_data()

if df is not None and not df.empty:
    # 创建筛选器
    all_categories = ["全部品类 (聚合)"] + sorted(df["品类"].unique())
    selected_category = st.selectbox("**筛选品类:**", all_categories)

    # 根据筛选结果展示数据
    if selected_category == "全部品类 (聚合)":
        st.header("家用电器 - 总榜单 (综合排名)")
        # 计算总榜单
        df_agg = df.groupby("品牌名称").agg({
            "品牌指数": "sum",
            "总提及次数": "sum",
            "出现次数": "sum",
        }).reset_index()
        df_agg = df_agg.sort_values(by="品牌指数", ascending=False).reset_index(drop=True)
        df_agg.index = df_agg.index + 1
    else:
        st.header(f"品类 - {selected_category}")
        df_agg = df[df["品类"] == selected_category].sort_values(by="品牌指数", ascending=False).reset_index(drop=True)
        df_agg.index = df_agg.index + 1
        df_agg = df_agg.drop(columns=["品类"])

    # 使用st.dataframe来展示，并可以进行一些样式配置
    st.dataframe(
        df_agg.style.format({"品牌指数": "{:.2f}"}).highlight_max(subset=["品牌指数"], color="lightgreen"),
        use_container_width=True
    )
else:
    st.error("数据加载失败，请确保 'results.json' 文件存在且格式正确。")

