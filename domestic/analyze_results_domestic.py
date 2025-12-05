import json
import os
import re
import math
import yaml
import argparse
from collections import Counter, defaultdict
import time
from collections import defaultdict
import numpy as np


# ==============================================================================
# 国内榜单分析引擎 (简化版 - 只生成总榜单)
# 描述: 专门用于国内榜单分析，不分子品类，只生成一个总榜单
# 用法: python analyze_results_domestic.py --task scenic --results results_scenic_merged.json --brands brand_dictionary_scenic.yaml
# python analyze_results_domestic.py --task nev --results results_nev_merged.json --brands brand_dictionary_nev.yaml
# python analyze_results_domestic.py --task phone --results results_phone_merged.json --brands brand_dictionary_phone.yaml
# python analyze_results_domestic.py --task food --results results_food_merged.json --brands brand_dictionary_food.yaml
# python analyze_results_domestic.py --task snack --results merged_results/results_snack_merged.json --brands config/brand_dictionary_snack.yaml
# python analyze_results_domestic.py --task city --results results_city_merged.json --brands brand_dictionary_city.yaml
# python analyze_results_domestic.py --task luxury --results results_luxury_merged.json --brands brand_dictionary_luxury.yaml
# python analyze_results_domestic.py --task beauty --results results_beauty_merged.json --brands brand_dictionary_beauty.yaml
# python analyze_results_domestic.py --task travel --results results_merged_ts.json --brands config/brand_dictionary_ts_travel.yaml
# python analyze_results_domestic.py --task tc_city --results /Users/charlotte/PycharmProjects/GEO/oversea/results_merged_tc.json --brands brand_dictionary_tc.yaml
# ==============================================================================


def analyze_single_answer(answer_text: str, references: list, brand_map: dict):
    """分析单个回答，提取品牌相关指标"""
    raw_metrics = defaultdict(
        lambda: {"mentioned": 0, "first_pos": float('inf'), "is_strong": 0, "ref_count": 0, "mention_count": 0})
    answer_lower = answer_text.lower()

    # 检测品牌提及
    for std_brand, aliases in brand_map.items():
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower in answer_lower:
                raw_metrics[std_brand]["mentioned"] = 1
                raw_metrics[std_brand]["mention_count"] += answer_lower.count(alias_lower)
                try:
                    pos = answer_lower.index(alias_lower)
                    if pos < raw_metrics[std_brand]["first_pos"]:
                        raw_metrics[std_brand]["first_pos"] = pos
                except ValueError:
                    pass

    # 检测强推荐
    sentences = re.split(r'[。\n]', answer_text)

    strong_patterns = [
        r"(强烈)?推荐", r"首选", r"最佳", r"值得.*?(尝试|购买|选择)",
        r"性价比.*?(高|很高)", r"(是|属)?(top|best)[^。]*?(品牌|选择|之一)",
        r"(我|我们)?(最|很)?常买", r"(个人|我)?觉得.*?(最好|最推荐)"
    ]
    negation_keywords = ["不推荐", "不太", "不喜欢", "不值得", "踩雷", "避坑", "最差", "不合适", "不如"]

    for sentence in sentences:
        sentence_lower = sentence.lower()
        for brand, metrics in raw_metrics.items():
            if not metrics["mentioned"]:
                continue
            if brand.lower() in sentence_lower:
                if any(re.search(p, sentence_lower) for p in strong_patterns):
                    if not any(neg in sentence_lower for neg in negation_keywords):
                        raw_metrics[brand]["is_strong"] = 1

    # 检测引用
    # if references:
    #     for brand, metrics in raw_metrics.items():
    #         if metrics["mentioned"]:
    #             for ref in references:
    #                 if brand.lower() in ref.lower():
    #                     raw_metrics[brand]["ref_count"] += 1

    return raw_metrics


import math
from collections import defaultdict


def safe_log1p_scaled(x, k=1000, scale=10):
    """对小数x进行平滑放大和对数缩放，避免负值"""
    return math.log1p(x * k) * scale


def calculate_scores(data_list: list, brand_dictionary: dict, whitelist: set, weights: dict) -> dict:
    """计算所有品牌的得分（保留原结构，仅 mind_share 使用 log 压缩）"""
    all_brands_raw_metrics = defaultdict(
        lambda: {"total_mentions": 0, "first_pos_sum": 0, "strong_recommend_count": 0,
                 "total_ref_count": 0, "mention_in_answers": 0})
    total_brand_mentions_across_all = 0

    # 收集所有原始指标
    for item in data_list:
        answer = item.get("response", {}).get("answer", "")
        references = item.get("response", {}).get("references", [])
        if not answer:
            continue

        answer_metrics = analyze_single_answer(answer, references, brand_dictionary)

        for brand, metrics in answer_metrics.items():
            if brand in whitelist:
                brand_global_metrics = all_brands_raw_metrics[brand]
                brand_global_metrics["total_mentions"] += metrics["mention_count"]
                if metrics["first_pos"] != float('inf'):
                    brand_global_metrics["first_pos_sum"] += metrics["first_pos"]
                brand_global_metrics["strong_recommend_count"] += metrics["is_strong"]
                brand_global_metrics["total_ref_count"] += metrics["ref_count"]
                brand_global_metrics["mention_in_answers"] += 1
                total_brand_mentions_across_all += metrics["mention_count"]

    if not all_brands_raw_metrics:
        return {}

    # 计算归一化参数
    max_mentions = max((m["total_mentions"] for m in all_brands_raw_metrics.values()), default=1)
    max_density = max((m["total_mentions"] / m["mention_in_answers"]
                       for m in all_brands_raw_metrics.values() if m["mention_in_answers"] > 0), default=1)
    max_strong = max((m["strong_recommend_count"] for m in all_brands_raw_metrics.values()), default=1)

    # 计算每个品牌的得分
    final_scores = {}
    for brand, metrics in all_brands_raw_metrics.items():
        avg_pos = metrics["first_pos_sum"] / metrics["mention_in_answers"] if metrics[
                                                                                  "mention_in_answers"] > 0 else float(
            'inf')
        mention_density = metrics["total_mentions"] / metrics["mention_in_answers"] if metrics[
                                                                                           "mention_in_answers"] > 0 else 0

        # 1. 品牌可见度（可线性规则）
        if avg_pos == float('inf'):
            score_visibility = 0
        elif avg_pos < 500:
            score_visibility = 100
        elif avg_pos < 1500:
            score_visibility = 100 * (1 - (avg_pos - 500) / 1000)
        else:
            score_visibility = 0

        # 2. 引用率（总提及次数）
        score_mention_rate = (metrics["total_mentions"] / max_mentions) * 100 if max_mentions > 0 else 0

        # 3. AI认知指数（强推荐次数）
        normalized_strong = (metrics["strong_recommend_count"] + 1) / (max_strong + 1)
        score_ai_ranking = math.sqrt(normalized_strong) * 100

        # 4. 平均引用密度
        norm_density = mention_density / max_density
        score_ref_depth = safe_log1p_scaled(norm_density, k=1000)

        # 5. AI认知份额（仅此项使用 log(1 + kx) × scale）
        normalized_mind_share = metrics["total_mentions"] / total_brand_mentions_across_all
        score_mind_share = safe_log1p_scaled(normalized_mind_share, k=1000, scale=20)

        # 加权平均
        total_score = (
                              score_visibility * weights["visibility"] +
                              score_mention_rate * weights["mention_rate"] +
                              score_ai_ranking * weights["ai_ranking"] +
                              score_ref_depth * weights["ref_depth"] +
                              score_mind_share * weights["mind_share"]
                      ) / 100 # 加常数以避免分数过低

        final_scores[brand] = {
            "品牌指数": total_score,
            "总提及次数": metrics["total_mentions"],
            "出现次数": metrics["mention_in_answers"],
            "强推荐次数": metrics["strong_recommend_count"],
            "平均提及密度": mention_density,
            "维度得分": {
                "visibility": score_visibility,
                "mention_rate": score_mention_rate,
                "ai_ranking": score_ai_ranking,
                "ref_depth": score_ref_depth,
                "mind_share": score_mind_share
            }
        }

    return final_scores


def write_ranking_report(output_file: str, title: str, scores: dict, task_name: str):
    """生成Markdown格式的排名报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n")
        f.write(f"**报告生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**分析任务**: {task_name}\n\n")
        f.write("---\n\n")

        if not scores:
            f.write("⚠️ 未找到任何品牌得分数据。\n\n")
            return

        # 总榜单表格
        f.write("## 📊 品牌排名总榜单\n\n")
        f.write(
            "| 排名 | 品牌名称 | 品牌指数 | 总提及次数 | 出现次数 | 强推荐次数 | 品牌可见度 | 引用率 | 品牌AI认知指数 | 平均引用密度 | 竞争力指数 |\n")
        f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        sorted_brands = sorted(scores.items(), key=lambda x: x[1]["品牌指数"], reverse=True)

        for rank, (brand, data) in enumerate(sorted_brands, 1):
            dims = data["维度得分"]

            f.write(
                f"| {rank} | {brand} | **{data['品牌指数']:.2f}** | "
                f"{data['总提及次数']} | {data['出现次数']} | {data['强推荐次数']} | "
                f"{dims['visibility']:.1f} | {dims['mention_rate']:.1f} | {dims['ai_ranking']:.1f} | "
                f"{dims['ref_depth']:.1f} | {dims['mind_share']:.1f} |\n"
            )

        f.write("\n---\n\n")

        # 统计信息
        f.write("## 📈 统计信息\n\n")
        f.write(f"- **参与排名品牌数**: {len(scores)}\n")
        f.write(f"- **最高品牌指数**: {sorted_brands[0][1]['品牌指数']:.2f} ({sorted_brands[0][0]})\n")
        f.write(f"- **平均品牌指数**: {sum(s['品牌指数'] for s in scores.values()) / len(scores):.2f}\n")
        f.write(f"- **总提及次数**: {sum(s['总提及次数'] for s in scores.values())}\n")
        f.write("\n")

        # 说明
        f.write("## 📝 评分说明\n\n")
        f.write("本榜单采用五维度评分体系，每个指标各占20%权重：\n\n")
        f.write("1. **品牌可见度**: 品牌首次出现的位置越靠前，得分越高\n")
        f.write("2. **引用率**: 品牌被提及的总次数\n")
        f.write("3. **AI认知指数**: 品牌在“强推荐”句式中出现的频次\n")
        f.write("4. **平均引用密度**: 每次回答中该品牌平均被提及的次数，衡量讨论的密度”。\n")
        f.write("5. **AI认知份额 **: 品牌在所有品牌中的被提及占比\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="国内榜单分析引擎")
    parser.add_argument("--task", required=True, help="任务名称 (例如: nev, scenic)")
    parser.add_argument("--results", required=True, help="结果文件路径 (例如: results_nev_merged.json)")
    parser.add_argument("--brands", required=True, help="品牌词典文件路径 (例如: brand_dictionary_scenic.yaml)")
    parser.add_argument("--output", default=None, help="输出报告文件路径 (默认: ranking_report_{task}.md)")
    args = parser.parse_args()

    # 设置输出文件名
    if args.output is None:
        args.output = f"ranking_report_{args.task}.md"

    print(f"\n{'=' * 60}")
    print(f"国内榜单分析引擎")
    print(f"{'=' * 60}\n")
    print(f"📋 任务名称: {args.task}")
    print(f"📁 结果文件: {args.results}")
    print(f"📖 品牌词典: {args.brands}")
    print(f"📄 输出报告: {args.output}\n")

    # 加载品牌词典
    print("正在加载品牌词典...")
    try:
        with open(args.brands, 'r', encoding='utf-8') as f:
            brands_config = yaml.safe_load(f)
        brand_dictionary = brands_config['brand_dictionary']
        brands_whitelist = set(brands_config['brands_whitelist'])
        print(f"✅ 成功加载 {len(brand_dictionary)} 个品牌，白名单包含 {len(brands_whitelist)} 个品牌\n")
    except FileNotFoundError:
        print(f"❌ 错误: 品牌词典文件 '{args.brands}' 未找到。")
        return
    except Exception as e:
        print(f"❌ 错误: 加载品牌词典时出错: {e}")
        return

    # 加载结果数据
    print("正在加载结果数据...")
    try:
        with open(args.results, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        print(f"✅ 成功加载 {len(data_list)} 条回答记录\n")
    except FileNotFoundError:
        print(f"❌ 错误: 结果文件 '{args.results}' 未找到。")
        return
    except Exception as e:
        print(f"❌ 错误: 加载结果数据时出错: {e}")
        return

    # 定义权重（与海外榜单完全一致）
    weights = {
        "visibility": 20,
        "mention_rate": 20,
        "ai_ranking": 20,
        "ref_depth": 20,
        "mind_share": 20,
    }

    # 计算得分
    print("正在计算品牌得分...")
    scores = calculate_scores(data_list, brand_dictionary, brands_whitelist, weights)
    print(f"✅ 成功计算 {len(scores)} 个品牌的得分\n")

    # 生成报告
    print("正在生成排名报告...")
    report_title = f"{args.task.upper()} 品牌GenAI认知指数排行榜"
    write_ranking_report(args.output, report_title, scores, args.task)
    print(f"✅ 报告已保存到: {args.output}\n")

    # 显示前10名
    if scores:
        print("🏆 Top 10 品牌预览:")
        print("-" * 60)
        sorted_brands = sorted(scores.items(), key=lambda x: x[1]["品牌指数"], reverse=True)
        for rank, (brand, data) in enumerate(sorted_brands[:10], 1):
            print(f"  {rank:2d}. {brand:15s} - 品牌指数: {data['品牌指数']:6.2f}")
        print("-" * 60)

    print(f"\n{'=' * 60}")
    print("分析完成！")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
