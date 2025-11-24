import json
import os
import re
import math
import yaml
import argparse
from collections import Counter, defaultdict
import time


# ==============================================================================
# 国内榜单分析引擎 (简化版 - 只生成总榜单)
# 描述: 专门用于国内榜单分析，不分子品类，只生成一个总榜单
# 用法: python analyze_results_domestic.py --task nev --results results_nev_merged.json --brands brand_dictionary_nev.yaml
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
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for brand, metrics in raw_metrics.items():
            if metrics["mentioned"] and brand.lower() in sentence_lower:
                if any(word in sentence_lower for word in
                       ["首选", "最佳", "强烈推荐", "推荐", "值得", "best", "top", "recommended"]):
                    raw_metrics[brand]["is_strong"] = 1

    # 检测引用
    if references:
        for brand, metrics in raw_metrics.items():
            if metrics["mentioned"]:
                for ref in references:
                    if brand.lower() in ref.lower():
                        raw_metrics[brand]["ref_count"] += 1

    return raw_metrics


def calculate_scores(data_list: list, brand_dictionary: dict, whitelist: set, weights: dict) -> dict:
    """计算所有品牌的得分（严格按照海外榜单标准）"""
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
    max_mentions = max((m["total_mentions"] for m in all_brands_raw_metrics.values()), default=0)
    max_strong = max((m["strong_recommend_count"] for m in all_brands_raw_metrics.values()), default=0)
    max_refs = max((m["total_ref_count"] for m in all_brands_raw_metrics.values()), default=0)

    # 计算最大平均提及密度（用于替代ref_depth）
    max_density = max((m["total_mentions"] / m["mention_in_answers"] for m in all_brands_raw_metrics.values()
                       if m["mention_in_answers"] > 0), default=1)

    # 计算每个品牌的得分
    final_scores = {}
    for brand, metrics in all_brands_raw_metrics.items():
        avg_pos = metrics["first_pos_sum"] / metrics["mention_in_answers"] if metrics[
                                                                                  "mention_in_answers"] > 0 else float(
            'inf')

        # 1. 品牌可见度 (Visibility) - 实体首次出现的位置越靠前，得分越高
        if avg_pos == float('inf'):
            score_visibility = 0
        elif avg_pos < 500:
            score_visibility = 100
        elif avg_pos < 1500:
            score_visibility = 100 * (1 - (avg_pos - 500) / 1000)  # 线性递减
        else:
            score_visibility = 0

        # 2. 引用率 (Mention Rate) - 实体被提及的总次数
        score_mention_rate = (metrics["total_mentions"] / max_mentions) * 100 if max_mentions > 0 else 0

        # 3. AI认知排行指数 (AI Ranking) - 基于强推荐次数
        normalized_strong = (metrics["strong_recommend_count"] + 1) / (max_strong + 1)
        score_ai_ranking = math.sqrt(normalized_strong) * 100

        # 4. 正文引用深度 (Ref Depth) - 改为：平均提及密度
        # 衡量品牌在每次出现时的平均提及次数，反映讨论深度
        mention_density = metrics["total_mentions"] / metrics["mention_in_answers"] if metrics[
                                                                                           "mention_in_answers"] > 0 else 0
        score_ref_depth = (mention_density / max_density) * 100 if max_density > 0 else 0

        # 5. AI认知份额 (Mind Share) - 该实体提及次数占所有同类实体总数的比例
        normalized_mind_share = (metrics[
                                     "total_mentions"] / total_brand_mentions_across_all) if total_brand_mentions_across_all > 0 else 0
        score_mind_share = math.sqrt(normalized_mind_share) * 100

        # 6. 竞争力指数 (Competitiveness) - 使用前三个核心指标的平均分
        core_scores_avg = (score_visibility + score_mention_rate + score_ai_ranking) / 3
        score_competitiveness = core_scores_avg

        # 计算加权总分
        total_score = (
                              score_visibility * weights["visibility"] +
                              score_mention_rate * weights["mention_rate"] +
                              score_ai_ranking * weights["ai_ranking"] +
                              score_ref_depth * weights["ref_depth"] +
                              score_mind_share * weights["mind_share"] +
                              score_competitiveness * weights["competitiveness"]
                      ) / 100 + 10  # 除以100是因为权重总和为100

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
                "mind_share": score_mind_share,
                "competitiveness": score_competitiveness
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
            "| 排名 | 品牌名称 | 品牌指数 | 总提及次数 | 出现次数 | 强推荐次数 | 品牌可见度(20) | 引用率(20) | 品牌AI认知排行指数(20) | 正文引用率(15) | 品牌AI认知份额(15) | 竞争力指数(10) |\n")
        f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        sorted_brands = sorted(scores.items(), key=lambda x: x[1]["品牌指数"], reverse=True)

        for rank, (brand, data) in enumerate(sorted_brands, 1):
            dims = data["维度得分"]

            f.write(
                f"| {rank} | {brand} | **{data['品牌指数']:.2f}** | "
                f"{data['总提及次数']} | {data['出现次数']} | {data['强推荐次数']} | "
                f"{dims['visibility']:.1f} | {dims['mention_rate']:.1f} | {dims['ai_ranking']:.1f} | "
                f"{dims['ref_depth']:.1f} | {dims['mind_share']:.1f} | {dims['competitiveness']:.1f} |\n"
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
        f.write("本榜单采用与海外榜单相同的六维度评分体系：\n\n")
        f.write("1. **品牌可见度 (20%)**: 品牌首次出现的位置越靠前，得分越高\n")
        f.write("2. **引用率 (20%)**: 品牌被提及的总次数\n")
        f.write("3. **AI认知排行指数 (20%)**: 基于强推荐次数的评分\n")
        f.write("4. **正文引用率 (15%)**: 品牌在每次出现时的平均提及次数，反映讨论深度\n")
        f.write("5. **AI认知份额 (15%)**: 品牌提及次数占所有品牌总数的比例\n")
        f.write("6. **竞争力指数 (10%)**: 前三个核心指标的综合评分\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="国内榜单分析引擎")
    parser.add_argument("--task", required=True, help="任务名称 (例如: nev, scenic)")
    parser.add_argument("--results", required=True, help="结果文件路径 (例如: results_nev_merged.json)")
    parser.add_argument("--brands", required=True, help="品牌词典文件路径 (例如: brand_dictionary_nev.yaml)")
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
        "ref_depth": 15,
        "mind_share": 15,
        "competitiveness": 10
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
