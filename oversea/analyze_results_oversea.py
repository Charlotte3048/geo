import json
import re
import yaml
import argparse
import time
import math
import os
import sys
from collections import defaultdict

# ==============================================================================
# 海外榜单分析引擎 (支持总榜单 + 子品类榜单)
# 描述: 专门用于海外榜单分析，总榜单汇总所有数据，子品类榜单按子品类分别计算
# 用法:
# python analyze_results_oversea.py --config config/config_ha.yaml
# python analyze_results_oversea.py --config config/config_sh.yaml
# ==============================================================================

# 添加 domestic 目录到 sys.path，以便导入情感分析模块
# 注意：此路径需要根据实际项目结构调整
DOMESTIC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'domestic')
if os.path.exists(DOMESTIC_PATH):
    sys.path.insert(0, DOMESTIC_PATH)

# 导入BERT情感分析模块
try:
    from sentiment.sentiment_analyzer import get_sentiment_analyzer

    USE_BERT_SENTIMENT = True
    print("✅ BERT情感分析模块已启用")
except ImportError as e:
    USE_BERT_SENTIMENT = False
    print(f"⚠️  BERT情感分析模块未找到，将使用规则匹配方式: {e}")


def analyze_single_answer(answer_text: str, references: list, brand_map: dict):
    """分析单个回答，提取品牌相关指标"""
    raw_metrics = defaultdict(
        lambda: {
            "mentioned": 0,
            "first_pos": float('inf'),
            "is_strong": 0,
            "ref_count": 0,
            "mention_count": 0,
            "top10_points": 0,
            "sentiment_sentences": []
        }
    )
    answer_lower = answer_text.lower()

    # --- 1. 检测品牌提及并计算 first_pos ---
    brand_mentions_with_pos = []
    for std_brand, aliases in brand_map.items():
        for alias in aliases:
            alias_lower = alias.lower()
            # Find all occurrences of the alias
            for match in re.finditer(re.escape(alias_lower), answer_lower):
                raw_metrics[std_brand]["mentioned"] = 1
                raw_metrics[std_brand]["mention_count"] += 1

                pos = match.start()
                if pos < raw_metrics[std_brand]["first_pos"]:
                    raw_metrics[std_brand]["first_pos"] = pos

                brand_mentions_with_pos.append({
                    "brand": std_brand,
                    "pos": pos
                })

    # --- 2. 计算 Top 10 积分 (前10可见度) ---
    first_mention_positions = {}
    for mention in brand_mentions_with_pos:
        brand = mention["brand"]
        pos = mention["pos"]
        if brand not in first_mention_positions or pos < first_mention_positions[brand]:
            first_mention_positions[brand] = pos

    sorted_brands_by_pos = sorted(first_mention_positions.items(), key=lambda item: item[1])

    # 给前 10 个品牌分配积分 (1st=10, 10th=1)
    for rank, (brand, pos) in enumerate(sorted_brands_by_pos):
        if rank < 10:
            points = 10 - rank  # 1st (rank 0) gets 10, 10th (rank 9) gets 1
            raw_metrics[brand]["top10_points"] = points
        else:
            break

    # --- 3. 提取包含品牌的句子（用于BERT情感分析）---
    # 适配中英文句子分割
    sentences = re.split(r'[。\n.!?！？]', answer_text)

    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if not sentence_lower:
            continue

        for brand, metrics in raw_metrics.items():
            if not metrics["mentioned"]:
                continue
            # 检查句子中是否包含该品牌（检查所有别名）
            brand_aliases = brand_map.get(brand, [brand])
            for alias in brand_aliases:
                if alias.lower() in sentence_lower:
                    raw_metrics[brand]["sentiment_sentences"].append(sentence)
                    break  # 避免同一句子重复添加

    # --- 4. 检测强推荐 (is_strong) ---
    # 中英文强推荐关键词
    strong_patterns = [
        # 中文
        r"(强烈)?推荐", r"首选", r"最佳", r"值得.*?(尝试|购买|选择)",
        r"性价比.*?(高|很高)", r"(是|属)?(top|best)[^。]*?(品牌|选择|之一)",
        r"(我|我们)?(最|很)?常买", r"(个人|我)?觉得.*?(最好|最推荐)",
        # 英文
        r"highly\s+recommend", r"best\s+choice", r"top\s+pick", r"must\s+have",
        r"excellent", r"outstanding", r"superior", r"first\s+choice",
        r"strongly\s+recommend", r"worth\s+(buying|trying|considering)"
    ]
    negation_keywords = [
        # 中文
        "不推荐", "不太", "不喜欢", "不值得", "踩雷", "避坑", "最差", "不合适", "不如",
        # 英文
        "not recommend", "don't recommend", "wouldn't recommend", "avoid",
        "worst", "disappointing", "poor quality"
    ]

    for sentence in sentences:
        sentence_lower = sentence.lower()
        for brand, metrics in raw_metrics.items():
            if not metrics["mentioned"]:
                continue
            # 检查品牌是否在句子中
            brand_aliases = brand_map.get(brand, [brand])
            brand_in_sentence = any(alias.lower() in sentence_lower for alias in brand_aliases)
            if brand_in_sentence:
                if any(re.search(p, sentence_lower) for p in strong_patterns):
                    if not any(neg in sentence_lower for neg in negation_keywords):
                        raw_metrics[brand]["is_strong"] = 1

    return raw_metrics


def calculate_scores(data_list: list,
                     brand_dictionary: dict,
                     whitelist: set,
                     weights: dict,
                     analyzer=None) -> dict:
    """
    计算所有品牌的得分（集成BERT情感分析）

    返回: final_scores - 品牌得分字典
    """
    all_brands_raw_metrics = defaultdict(
        lambda: {
            "total_mentions": 0,
            "first_pos_sum": 0,
            "top10_score_sum": 0,
            "strong_recommend_count": 0,
            "total_ref_count": 0,
            "mention_in_answers": 0,
            "sentiment_sentences": []
        }
    )
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
                brand_global_metrics["top10_score_sum"] += metrics["top10_points"]
                brand_global_metrics["total_ref_count"] += metrics["ref_count"]
                brand_global_metrics["mention_in_answers"] += 1
                total_brand_mentions_across_all += metrics["mention_count"]

                # 收集情感分析句子
                brand_global_metrics["sentiment_sentences"].extend(metrics["sentiment_sentences"])

    if not all_brands_raw_metrics:
        return {}

    # 初始化情感分析器
    sentiment_analyzer = analyzer
    if sentiment_analyzer is None and USE_BERT_SENTIMENT:
        try:
            sentiment_analyzer = get_sentiment_analyzer()
            print("🤖 使用BERT模型进行情感分析...")
        except Exception as e:
            print(f"⚠️  BERT模型加载失败，回退到规则匹配: {e}")
            sentiment_analyzer = None

    # 计算归一化参数
    max_mentions = max((m["total_mentions"] for m in all_brands_raw_metrics.values()), default=1)
    max_strong = max((m["strong_recommend_count"] for m in all_brands_raw_metrics.values()), default=1)

    # 计算每个品牌的得分
    final_scores = {}
    for brand, metrics in all_brands_raw_metrics.items():
        avg_pos = metrics["first_pos_sum"] / metrics["mention_in_answers"] if metrics[
                                                                                  "mention_in_answers"] > 0 else float(
            'inf')
        mention_density = metrics["total_mentions"] / metrics["mention_in_answers"] if metrics[
                                                                                           "mention_in_answers"] > 0 else 0

        # 1. 品牌回答显著度 (Brand Prominence)
        if avg_pos == float('inf'):
            score_visibility = 0
        elif avg_pos < 500:
            score_visibility = 100
        elif avg_pos < 1500:
            score_visibility = 100 * (1 - (avg_pos - 500) / 1000)
        else:
            score_visibility = 0

        # 2. 声量占比 (Share of Voice)
        share_of_voice_ratio = metrics[
                                   "total_mentions"] / total_brand_mentions_across_all if total_brand_mentions_across_all > 0 else 0
        share_of_voice = (math.log(share_of_voice_ratio * 1000 + 1) / math.log(1001)) * 100

        # 3. 前10可见度 (Top10 Visibility)
        max_top10_score = max((m["top10_score_sum"] for m in all_brands_raw_metrics.values()), default=1)
        normalized_top10 = (metrics["top10_score_sum"] + 1) / (max_top10_score + 1)
        top10_visibility = math.sqrt(normalized_top10) * 100

        # 4. 竞争力指数 (Competitiveness)
        competitiveness = (metrics["total_mentions"] / max_mentions) * 100 if max_mentions > 0 else 0

        # 5. 情感分析 (Sentiment Analysis)
        if sentiment_analyzer and metrics["sentiment_sentences"]:
            sentences_to_analyze = metrics["sentiment_sentences"][:50]
            results = sentiment_analyzer.predict(sentences_to_analyze)
            sentiment_scores = [r["score"] for r in results]
            sentiment_analysis = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 50.0
        else:
            normalized_strong = (metrics["strong_recommend_count"] + 1) / (max_strong + 1)
            sentiment_analysis = math.sqrt(normalized_strong) * 100

        # 加权平均
        w_brand_prominence = weights.get("brand_prominence", 20)
        w_share_of_voice = weights.get("share_of_voice", 20)
        w_top10_visibility = weights.get("top10_visibility", 20)
        w_competitiveness = weights.get("competitiveness", 20)
        w_sentiment = weights.get("sentiment_analysis", 20)

        total_score = (
                              score_visibility * w_brand_prominence +
                              share_of_voice * w_share_of_voice +
                              top10_visibility * w_top10_visibility +
                              competitiveness * w_competitiveness +
                              sentiment_analysis * w_sentiment
                      ) / 100

        final_scores[brand] = {
            "品牌指数": total_score,
            "总提及次数": metrics["total_mentions"],
            "出现次数": metrics["mention_in_answers"],
            "强推荐次数": metrics["strong_recommend_count"],
            "平均提及密度": mention_density,
            "维度得分": {
                "brand_prominence": score_visibility,
                "share_of_voice": share_of_voice,
                "top10_visibility": top10_visibility,
                "competitiveness": competitiveness,
                "sentiment_analysis": sentiment_analysis
            }
        }

    return final_scores


def write_ranking_report(output_file: str, title: str, total_scores: dict, task_name: str,
                         subcategory_scores: dict = None):
    """
    生成Markdown格式的排名报告（包含总榜单和子品类榜单）

    参数:
        output_file: 输出文件路径
        title: 报告标题
        total_scores: 总榜单得分
        task_name: 任务名称
        subcategory_scores: 子品类榜单得分字典 {子品类名: 得分字典}
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{title}\n\n")
        f.write(f"**报告生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**分析任务**: {task_name}\n\n")
        f.write("---\n\n")

        if not total_scores:
            f.write("⚠️ 未找到任何品牌得分数据。\n\n")
            return

        # ==================== 总榜单 ====================
        f.write("## 📊 品牌排名总榜单\n\n")
        f.write(
            "| 排名 | 品牌名称 | 品牌指数 | 总提及次数 | 出现次数 | 强推荐次数 | 品牌显著度 | 声量占比 | 前10可见度 | 竞争力指数 | 情感分析 |\n")
        f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        sorted_brands = sorted(total_scores.items(), key=lambda x: x[1]["品牌指数"], reverse=True)

        for rank, (brand, data) in enumerate(sorted_brands, 1):
            dims = data["维度得分"]
            row_content = (
                f"| {rank} | {brand} | **{data['品牌指数']:.2f}** | "
                f"{data['总提及次数']} | {data['出现次数']} | {data['强推荐次数']} | "
                f"{dims['brand_prominence']:.1f} | {dims['share_of_voice']:.1f} | {dims['top10_visibility']:.1f} | "
                f"{dims['competitiveness']:.1f} | {dims['sentiment_analysis']:.1f} |"
            )
            f.write(row_content + "\n")

        f.write("\n")

        # ==================== 子品类榜单 ====================
        if subcategory_scores:
            f.write("---\n\n")
            f.write("## 📂 子品类榜单\n\n")
            f.write("> 以下榜单按子品类分别计算，每个子品类的品牌指数基于该子品类下的数据独立计算。\n\n")

            for subcategory in sorted(subcategory_scores.keys()):
                scores = subcategory_scores[subcategory]

                if not scores:
                    continue

                f.write(f"### 📌 {subcategory}\n\n")
                # 简化的表格：只显示排名、品牌名称、品牌指数、总提及次数、出现次数
                f.write("| 排名 | 品牌名称 | 品牌指数 | 总提及次数 | 出现次数 |\n")
                f.write("|:---:|:---|:---:|:---:|:---:|\n")

                # 按品牌指数排序
                sorted_subcategory_brands = sorted(scores.items(), key=lambda x: x[1]["品牌指数"], reverse=True)

                for rank, (brand, data) in enumerate(sorted_subcategory_brands, 1):
                    row_content = (
                        f"| {rank} | {brand} | **{data['品牌指数']:.2f}** | "
                        f"{data['总提及次数']} | {data['出现次数']} |"
                    )
                    f.write(row_content + "\n")

                f.write("\n")

        # ==================== 统计信息 ====================
        f.write("---\n\n")
        f.write("## 📈 统计信息\n\n")
        f.write(f"- **参与排名品牌数**: {len(total_scores)}\n")
        f.write(f"- **最高品牌指数**: {sorted_brands[0][1]['品牌指数']:.2f} ({sorted_brands[0][0]})\n")
        f.write(f"- **平均品牌指数**: {sum(s['品牌指数'] for s in total_scores.values()) / len(total_scores):.2f}\n")
        f.write(f"- **总提及次数**: {sum(s['总提及次数'] for s in total_scores.values())}\n")
        if subcategory_scores:
            f.write(f"- **子品类数量**: {len(subcategory_scores)}\n")
        f.write("\n")

        # ==================== 评分说明 ====================
        f.write("## 📝 评分说明\n\n")
        f.write("本榜单采用五维度评分体系，每个指标各占20%权重：\n\n")
        f.write("1. **品牌显著度 (Brand Prominence)**: 品牌在AI回答中出现的位置，位置越靠前，分数越高\n")
        f.write("2. **声量占比 (Share of Voice)**: 品牌提及次数与所有品牌提及次数的比率\n")
        f.write("3. **前10可见度 (Top10 Visibility)**: 品牌在AI回答中前十名出现的效果，名次越高分数越高\n")
        f.write("4. **竞争力指数 (Competitiveness)**: 品牌与提及率最高品牌的提及率之比\n")
        f.write("5. **情感分析 (Sentiment Analysis)**: AI回答中关于品牌的正/负面内容分析\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="海外榜单分析引擎")
    parser.add_argument("--config", required=True, help="配置文件路径 (例如: config_home_appliance.yaml)")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"海外榜单分析引擎")
    print(f"{'=' * 60}\n")

    # 加载配置文件
    print(f"📋 加载配置文件: {args.config}")
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 配置文件 '{args.config}' 未找到。")
        return
    except Exception as e:
        print(f"❌ 错误: 加载配置文件时出错: {e}")
        return

    # 解析配置
    task_name = config.get("task_name", "unknown")
    results_file = config.get("results_file", "")
    output_file = config.get("ranking_output_file", f"ranking_report_{task_name}.md")
    report_title = config.get("report_title", f"# {task_name.upper()} 品牌AI认知指数排行榜")
    weights = config.get("weights", {
        "brand_prominence": 20,
        "share_of_voice": 20,
        "top10_visibility": 20,
        "competitiveness": 20,
        "sentiment_analysis": 20
    })
    brand_dictionary = config.get("brand_dictionary", {})
    brands_whitelist = set(config.get("brands_whitelist", []))

    print(f"📁 任务名称: {task_name}")
    print(f"📁 结果文件: {results_file}")
    print(f"📄 输出报告: {output_file}")
    print(f"📖 品牌词典: {len(brand_dictionary)} 个品牌")
    print(f"📋 白名单: {len(brands_whitelist)} 个品牌\n")

    # 加载结果数据
    print("正在加载结果数据...")
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        print(f"✅ 成功加载 {len(data_list)} 条回答记录\n")
    except FileNotFoundError:
        print(f"❌ 错误: 结果文件 '{results_file}' 未找到。")
        return
    except Exception as e:
        print(f"❌ 错误: 加载结果数据时出错: {e}")
        return

    # 按子品类分组数据
    subcategory_data = defaultdict(list)
    for item in data_list:
        category = item.get("category", "")
        if "-" in category:
            parts = category.split("-", 1)
            if len(parts) > 1:
                subcategory = parts[1]
                subcategory_data[subcategory].append(item)

    print(f"📂 发现 {len(subcategory_data)} 个子品类: {', '.join(sorted(subcategory_data.keys()))}\n")

    # ==================== 计算总榜单 ====================
    print("正在计算总榜单...")
    total_scores = calculate_scores(data_list, brand_dictionary, brands_whitelist, weights)
    print(f"✅ 总榜单: 成功计算 {len(total_scores)} 个品牌的得分\n")

    # ==================== 计算子品类榜单 ====================
    subcategory_scores = {}
    if subcategory_data:
        print("正在计算子品类榜单...")
        for subcategory, sub_data in sorted(subcategory_data.items()):
            print(f"  - 正在处理子品类: {subcategory} ({len(sub_data)} 条记录)")
            scores = calculate_scores(sub_data, brand_dictionary, brands_whitelist, weights)
            subcategory_scores[subcategory] = scores
            print(f"    ✅ 计算了 {len(scores)} 个品牌的得分")
        print()

    # 生成报告
    print("正在生成排名报告...")
    write_ranking_report(
        output_file,
        report_title,
        total_scores,
        task_name,
        subcategory_scores
    )
    print(f"✅ 报告已保存到: {output_file}\n")

    # 显示前10名
    if total_scores:
        print("🏆 Top 10 品牌预览 (总榜单):")
        print("-" * 60)
        sorted_brands = sorted(total_scores.items(), key=lambda x: x[1]["品牌指数"], reverse=True)
        for rank, (brand, data) in enumerate(sorted_brands[:10], 1):
            print(f"  {rank:2d}. {brand:15s} - 品牌指数: {data['品牌指数']:6.2f}")
        print("-" * 60)

    print(f"\n{'=' * 60}")
    print("分析完成！")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()