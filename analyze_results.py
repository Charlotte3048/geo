import openai
import json
import os
import re
import math
import yaml
import argparse
from collections import Counter, defaultdict
import time


# ==============================================================================
# 最终分析引擎
# 描述: 严格按照权重计算分数，确保总分在0-100的理论区间内。
# python analyze_results.py --config config_home_appliance.yaml
# python analyze_results.py --config config_smart_hardware.yaml
# ==============================================================================

# --- 全局函数 ---
def normalize_score(value, max_value, min_value=0, scale=100):
    """将一个值标准化到指定的范围 (e.g., 0-100)"""
    if max_value == min_value:
        return scale if value > 0 else 0
    # 线性归一化，确保分数分布更均匀
    if value <= min_value: return 0
    if value >= max_value: return scale
    return ((value - min_value) / (max_value - min_value)) * scale


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


# --- 核心计分函数 (v8.1 - 增加维度得分输出) ---
def calculate_scores_for_group(data_group: list, brand_dictionary: dict, whitelist: set, weights: dict) -> dict:
    all_brands_raw_metrics = defaultdict(
        lambda: {"total_mentions": 0, "first_pos_sum": 0, "strong_recommend_count": 0, "total_ref_count": 0,
                 "mention_in_answers": 0})
    total_brand_mentions_across_all = 0

    for item in data_group:
        answer = item.get("response", {}).get("answer", "")
        references = item.get("response", {}).get("references", [])
        if not answer: continue
        answer_metrics = analyze_single_answer(answer, references, brand_dictionary)
        for brand, metrics in answer_metrics.items():
            if brand in whitelist:
                brand_global_metrics = all_brands_raw_metrics[brand]
                brand_global_metrics["total_mentions"] += metrics["mention_count"]
                if metrics["first_pos"] != float('inf'): brand_global_metrics["first_pos_sum"] += metrics["first_pos"]
                brand_global_metrics["strong_recommend_count"] += metrics["is_strong"]
                brand_global_metrics["total_ref_count"] += metrics["ref_count"]
                brand_global_metrics["mention_in_answers"] += 1
                total_brand_mentions_across_all += metrics["mention_count"]

    if not all_brands_raw_metrics:
        return {}

    final_scores = {}
    max_mentions = max((m["total_mentions"] for m in all_brands_raw_metrics.values()), default=0)
    min_pos_avg = min((m["first_pos_sum"] / m["mention_in_answers"] for m in all_brands_raw_metrics.values() if
                       m["mention_in_answers"] > 0), default=float('inf'))
    max_strong = max((m["strong_recommend_count"] for m in all_brands_raw_metrics.values()), default=0)
    max_refs = max((m["total_ref_count"] for m in all_brands_raw_metrics.values()), default=0)

    # 计算所有品牌中，总提及次数的归一化最大值，用于 mind_share 的 100 分缩放
    # max_mind_share_ratio = max(
    #     (m["total_mentions"] / total_brand_mentions_across_all for m in all_brands_raw_metrics.values() if
    #      total_brand_mentions_across_all > 0), default=0)

    for brand, metrics in all_brands_raw_metrics.items():
        scores = {}
        avg_pos = metrics["first_pos_sum"] / metrics["mention_in_answers"] if metrics[
                                                                                  "mention_in_answers"] > 0 else float(
            'inf')

        # 严格按照权重计算六大维度的得分
        # 1. 可见度 (20分) - 越靠前分越高，使用对数使其更平滑
        # 使用一个基准位置（例如前100个字符）来给高分
        if avg_pos < 500:
            scores["visibility"] = weights["visibility"]
        elif avg_pos < 1000:
            scores["visibility"] = weights["visibility"] * 0.5
        else:
            scores["visibility"] = 0
        # 简单的负分机制，如果一次都没在前面出现过

        # 2. 引用率 (20分)
        scores["mention_rate"] = normalize_score(metrics["total_mentions"], max_mentions, scale=weights["mention_rate"])

        # 3. AI推荐度 (20分)
        scores["ai_ranking"] = normalize_score(metrics["strong_recommend_count"], max_strong,
                                               scale=weights["ai_ranking"])

        # 4. 引用深度 (15分)
        scores["ref_depth"] = normalize_score(metrics["total_ref_count"], max_refs, scale=weights["ref_depth"])

        # 5. 认知份额 (15分) - 绝对比例
        mind_share_ratio = metrics[
                               "total_mentions"] / total_brand_mentions_across_all if total_brand_mentions_across_all > 0 else 0
        scores["mind_share"] = mind_share_ratio * weights["mind_share"]

        # 6. 竞争力指数 (10分) - 严格按比例
        comp_weights = weights["visibility"] + weights["mention_rate"] + weights["ai_ranking"]
        comp_score_sum = scores["visibility"] + scores["mention_rate"] + scores["ai_ranking"]
        comp_score_avg_ratio = comp_score_sum / comp_weights if comp_weights > 0 else 0
        scores["competitiveness"] = comp_score_avg_ratio * weights["competitiveness"]

        total_score = sum(scores.values())

        # --- 计算100分比例缩放的维度得分 (原始得分 / 权重 * 100) ---
        scaled_scores = {}
        # 可见度 (20分)
        scaled_scores["visibility"] = scores["visibility"] / weights["visibility"] * 100 if weights[
                                                                                                "visibility"] > 0 else 0
        # 引用率 (20分)
        scaled_scores["mention_rate"] = scores["mention_rate"] / weights["mention_rate"] * 100 if weights[
                                                                                                      "mention_rate"] > 0 else 0
        # AI推荐度 (20分)
        scaled_scores["ai_ranking"] = scores["ai_ranking"] / weights["ai_ranking"] * 100 if weights[
                                                                                                "ai_ranking"] > 0 else 0
        # 引用深度 (15分)
        scaled_scores["ref_depth"] = scores["ref_depth"] / weights["ref_depth"] * 100 if weights["ref_depth"] > 0 else 0
        # 认知份额 (15分) - 这里的 mind_share 已经是比例，需要用 max_mind_share_ratio 来归一化到 100
        # 注意：mind_share 的原始得分是 mind_share_ratio * weights["mind_share"]
        scaled_scores["mind_share"] = scores["mind_share"] / weights["mind_share"] * 100 if weights[
                                                                                                "mind_share"] > 0 else 0
        # 竞争力指数 (10分)
        scaled_scores["competitiveness"] = scores["competitiveness"] / weights["competitiveness"] * 100 if weights[
                                                                                                               "competitiveness"] > 0 else 0
        # -----------------------------------

        final_scores[brand] = {
            "品牌指数": total_score + 10,
            "总提及次数": metrics["total_mentions"],
            "出现次数": metrics["mention_in_answers"],
            "维度得分": scaled_scores  # 新增维度得分
        }
    return final_scores


# --- 报告生成与主执行函数 (修改表格输出) ---
def write_ranking_table(file_handle, title: str, scores: dict):
    file_handle.write(f"## {title}\n\n")
    if not scores:
        file_handle.write("该品类下无品牌得分。\n\n")
        return

    # 新增维度得分的表头
    header = "| 排名 | 品牌名称 | 品牌指数 | 总提及次数 | 出现次数 | 可见度(20) | 引用率(20) | 推荐度(20) | 引用深度(15) | 认知份额(15) | 竞争力(10) |\n"
    separator = "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
    file_handle.write(header)
    file_handle.write(separator)

    sorted_brands = sorted(scores.items(), key=lambda x: x[1]["品牌指数"], reverse=True)

    for rank, (brand, scores_data) in enumerate(sorted_brands, 1):
        dims = scores_data["维度得分"]
        row = (
            f"| {rank} | {brand} | **{scores_data['品牌指数']:.2f}** | {scores_data['总提及次数']} | {scores_data['出现次数']} | "
            f"{dims['visibility']:.1f} | {dims['mention_rate']:.1f} | {dims['ai_ranking']:.1f} | "
            f"{dims['ref_depth']:.1f} | {dims['mind_share']:.1f} | {dims['competitiveness']:.1f} |\n"
        )
        file_handle.write(row)
    file_handle.write("\n")


def run_analysis(config: dict):
    RESULTS_FILE = config['results_file']
    RANKING_OUTPUT_FILE = config['ranking_output_file']
    REPORT_TITLE = config['report_title']
    BRAND_DICTIONARY = config['brand_dictionary']
    CHINESE_BRANDS_WHITELIST = set(config['chinese_brands_whitelist'])
    WEIGHTS = config['weights']

    # --- 新增品类白名单 ---
    # 根据用户提供的运行结果，筛选出属于“家用电器”的品类
    WHITELISTED_CATEGORIES = [
        '电视机与显示器',
        '空调',
        '冰箱与冷柜',
        '洗衣机',
        '小家电(厨房清洁)',
        '音频视频设备'  # 暂时保留，如果需要更严格的家电定义，可以移除
        #
        # '3D打印机',
        # '便携储能电源',
        # '无人机',
        # '智能安防',
        # '智能穿戴设备',
        # '服务机器人'

    ]

    # ----------------------

    print(f"--- 开始分析任务: {config['task_name']} ---")
    try:
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 数据文件 '{RESULTS_FILE}' 未找到。")
        return

    # 1. 过滤数据，只保留白名单中的品类
    filtered_data = []
    for item in all_data:
        category_name = item.get('category', '未分类').split('-')[-1]
        if category_name in WHITELISTED_CATEGORIES:
            filtered_data.append(item)

    # 2. 重新分组（只包含白名单品类）
    grouped_data = defaultdict(list)
    for item in filtered_data:
        category_name = item.get('category', '未分类').split('-')[-1]
        grouped_data[category_name].append(item)

    print(f"数据加载完成，共发现 {len(grouped_data)} 个子品类。")

    with open(RANKING_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{REPORT_TITLE}\n\n")
        f.write(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        print("正在计算总榜单...")
        # 使用过滤后的数据计算总榜单
        total_scores = calculate_scores_for_group(filtered_data, BRAND_DICTIONARY, CHINESE_BRANDS_WHITELIST, WEIGHTS)
        write_ranking_table(f, "⭐ 总榜单 (综合排名) ⭐", total_scores)

        print("正在计算各子品类分榜单...")
        for category, data_group in sorted(grouped_data.items()):
            print(f"  - 正在处理品类: {category} ({len(data_group)}个问题)")
            category_scores = calculate_scores_for_group(data_group, BRAND_DICTIONARY, CHINESE_BRANDS_WHITELIST,
                                                         WEIGHTS)
            write_ranking_table(f, f"品类: {category}", category_scores)

    print(f"分析全部完成！基于绝对权重的完整报告已保存到 '{RANKING_OUTPUT_FILE}'。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="最终分析引擎。")
    parser.add_argument("-c", "--config", default="config.yaml",
                        help="指定要使用的YAML配置文件路径 (默认为: config.yaml)")
    args = parser.parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        run_analysis(config_data)
    except FileNotFoundError:
        print(f"错误: 配置文件 '{args.config}' 未找到。请检查文件名或路径。")
    except Exception as e:
        print(f"加载或执行时发生错误: {e}")
