import math

import openai
import json
import os
import re
from collections import Counter, defaultdict
import time

# --- 1. 配置 (Configuration) ---
RESULTS_FILE = "results.json"
CONFIRMED_BRANDS_FILE = "confirmed_brands.txt"
RANKING_OUTPUT_FILE = "ranking_results.md"

# 用于智能提取的模型 (推荐使用能力强的模型)
EXTRACTION_MODEL = "openai/gpt-4o"


# --- 2. 智能品牌发现 (Smart Brand Discovery) ---
def get_brands_from_text_with_ai(client: openai.OpenAI, text: str) -> list:
    """使用强大的AI模型从文本中提取品牌名称"""
    system_prompt = """
    你是一个专业的市场分析师。你的任务是从给定的文本中，仅抽取出所有明确的、代表公司或产品的品牌名称。
    规则:
    1. 只返回品牌名，例如 "Roborock", "TCL", "美的"。
    2. 忽略技术术语 (如 "Mini LED", "QLED"), 地理位置 (如 "北美", "欧洲"), 通用名词 (如 "电视", "技术") 和网站名 (如 "Amazon", "Rtings.com")。
    3. 如果一个品牌有多种称呼 (如 "石头科技" 和 "Roborock")，请尽量都提取出来。
    4. 以一个JSON数组的格式返回，例如: ["Roborock", "Ecovacs", "美的"]。如果未发现品牌，则返回空数组 []。
    """
    try:
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},  # 强制JSON输出
        )
        # 假设返回的JSON结构是 {"brands": ["brand1", "brand2"]} 或直接是 ["brand1", "brand2"]
        content = json.loads(response.choices[0].message.content)
        if isinstance(content, dict):
            # 兼容 {"brands": [...]} 这种格式
            return content.get("brands", [])
        elif isinstance(content, list):
            # 兼容直接返回 [...] 的格式
            return content
        return []
    except Exception as e:
        print(f"智能提取时发生错误: {e}")
        return []


def discover_and_confirm_brands(client: openai.OpenAI, all_answers: list):
    """遍历所有回答，智能提取品牌，并生成待审核列表"""
    print("--- 阶段一: 智能品牌发现 ---")
    all_extracted_brands = []
    for i, answer in enumerate(all_answers):
        print(f"正在处理回答 {i + 1}/{len(all_answers)}...")
        brands = get_brands_from_text_with_ai(client, answer)
        if brands:
            all_extracted_brands.extend(brands)
        time.sleep(1)  # 避免触发速率限制

    brand_counts = Counter(all_extracted_brands)

    print(f"\n智能提取完成！发现 {len(brand_counts)} 个独特的候选品牌。")

    with open(CONFIRMED_BRANDS_FILE, 'w', encoding='utf-8') as f:
        f.write("# AI确认的候选品牌列表 (按出现频率排序)\n")
        f.write("# ----------------------------------------\n")
        f.write("# 请基于此列表，在下方脚本的 BRAND_DICTIONARY 中构建您的品牌词典。\n\n")
        for brand, count in brand_counts.most_common():
            f.write(f"{brand} ({count})\n")

    print(f"高度精确的候选品牌列表已保存到 '{CONFIRMED_BRANDS_FILE}'。")
    print("下一步：请打开此文件，并根据其内容完成脚本中的 BRAND_DICTIONARY。")
    return brand_counts


# --- 3. 品牌词典与计分规则 ---

# --- 2. 品牌词典与白名单 ---

# TODO: 步骤一：填充完整的品牌词典
# 这个词典需要包含所有你想识别的品牌，包括国际品牌，以便正确解析文本。
# 键是标准品牌名，值是别名列表。
# ==============================================================================
# BRAND_DICTIONARY: 包含所有需要识别的品牌（中外品牌）及其别名
# ==============================================================================
BRAND_DICTIONARY = {
    # --- 主要中国家电品牌 ---
    "Dreame": ["dreame", "追觅", "追觅科技", "追觅 Dreame"],
    "Xiaomi": ["xiaomi", "小米", "米家", "mijia", "Mijia", "MIJIA", "Mi Store", "小米有品", "华为智选"],
    # 华为智选也可能销售小米生态产品
    "Hisense": ["hisense", "海信"],
    "TCL": ["tcl", "华星光电"],  # 华星光电是TCL的子公司
    "Midea": ["midea", "美的"],
    "Joyoung": ["joyoung", "九阳"],
    "Roborock": ["roborock", "石头科技", "石头", "Q Revo"],  # Q Revo是Roborock的产品系列名，但有时可能被单独提及
    "Laifen": ["laifen", "徕芬", "徕芬科技"],
    "Ecovacs": ["ecovacs", "科沃斯", "ECOVACS"],
    "Bear": ["bear", "小熊", "小熊电器"],
    "COSORI": ["cosori", "可松"],  # COSORI是Vesync旗下的品牌，可松是音译
    "Liven": ["liven", "利仁"],
    "Soocas": ["soocas", "素士", "SOOCAS", "素士 SOOCAS"],
    "Narwal": ["narwal", "云鲸", "云鲸智能"],
    "Skyworth": ["skyworth", "创维"],
    "Viomi": ["viomi", "云米"],
    "Deerma": ["deerma", "德尔玛"],
    "Morphy Richards": ["morphy richards", "摩飞"],  # 摩飞电器原是英国品牌，在中国由新宝股份运营
    "Royalstar": ["royalstar", "荣事达"],
    "SUPOR": ["supor", "苏泊尔"],
    "Proscenic": ["proscenic", "浦桑尼克"],
    "Nathome": ["nathome", "北欧欧慕"],
    "Changhong": ["changhong", "长虹"],
    "XGIMI": ["xgimi", "极米"],
    "JMGO": ["jmgo", "坚果投影"],
    "Dangbei": ["dangbei", "当贝"],
    "Huawei": ["huawei", "华为"],
    "HONOR": ["honor"],
    "Konka": ["konka", "康佳"],
    "Vivo": ["vivo"],
    "Ulike": ["ulike", "优利克"],
    "Moyu": ["moyu", "初语 Moyu", "MoyuCare"],
    "Enchen": ["enchen", "映趣"],
    "SHOWSEE": ["showsee", "昂秀"],
    "Gaabor": ["gaabor", "高梵"],  # Gaabor原是德国品牌，现被中国公司收购和运营

    # --- 国际品牌 (用于识别和过滤) ---
    "Dyson": ["dyson", "戴森"],
    "Ninja": ["ninja"],
    "Samsung": ["samsung", "三星"],
    "Toshiba": ["toshiba", "东芝"],
    "Bissell": ["bissell", "必胜", "Bissell China"],
    "Chefman": ["chefman"],
    "Innsky": ["innsky"],
    "Ultrean": ["ultrean"],
    "Geek Chef": ["geek chef", "极客厨"],
    "GoWISE USA": ["gowise usa"],
    "Gorenje": ["gorenje"],
    "LG": ["lg"],
    "Sony": ["sony", "索尼"],
    "Panasonic": ["panasonic", "松下"],
    "Remington": ["remington"],
    "Philips": ["philips", "飞利浦"],
}

# TODO: 步骤二：定义中国品牌白名单
# 这是最关键的一步！只有在这里列出的品牌，才会进入最终的计分排名。
# 请确保这里只包含中国品牌，并且使用的是 BRAND_DICTIONARY 中的标准键名。
# ==============================================================================
# CHINESE_BRANDS_WHITELIST: 仅包含中国品牌标准名，用于最终计分
# ==============================================================================
CHINESE_BRANDS_WHITELIST = {
    "Dreame",
    "Xiaomi",
    "Hisense",
    "TCL",
    "Midea",
    "Joyoung",
    "Roborock",
    "Laifen",
    "Ecovacs",
    "Bear",
    "COSORI",
    "Liven",
    "Soocas",
    "Narwal",
    "Skyworth",
    "Viomi",
    "Deerma",
    "Morphy Richards",  # 在中国运营，计入
    "Royalstar",
    "SUPOR",
    "Proscenic",
    "Nathome",
    "Changhong",
    "XGIMI",
    "JMGO",
    "Dangbei",
    "Huawei",
    "HONOR",
    "Konka",
    "Vivo",
    "Ulike",
    "Moyu",
    "Enchen",
    "SHOWSEE",
    "Gaabor",  # 已被中国公司收购运营，计入
    "Anker",  # Anker虽然没出现在您的列表中，但很重要，建议保留
    "Eufy",  # Anker旗下品牌，建议保留
    "Tineco",  # Tineco也建议保留
}

# 计分权重
WEIGHTS_V3 = {
    "visibility": 20,  # 品牌可见度
    "mention_rate": 20,  # 引用率 (提及次数)
    "ai_ranking": 20,  # 品牌AI认知排行榜指数 (推荐强度)
    "ref_depth": 15,  # 正文引用率 (引用深度)
    "mind_share": 15,  # 品牌AI认知份额
    "competitiveness": 10,  # 竞争力指数
}


def normalize_score(value, max_value, min_value=0, scale=100):
    """将一个值标准化到指定的范围 (e.g., 0-100)"""
    if max_value == min_value:
        return scale if value > 0 else 0
    # 使用对数缩放，让头部差异更明显，尾部差异不那么刺眼
    log_value = math.log1p(value)
    log_max = math.log1p(max_value)
    return (log_value / log_max) * scale if log_max > 0 else 0


def analyze_single_answer(answer_text: str, references: list, brand_map: dict):
    """分析单个AI回答，提取所有品牌的原始指标"""
    raw_metrics = defaultdict(lambda: {
        "mentioned": 0,
        "first_pos": float('inf'),
        "is_strong": 0,
        "ref_count": 0,
        "mention_count": 0,
    })
    answer_lower = answer_text.lower()

    # 1. 提取基础指标
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

    # 2. 提取推荐强度
    sentences = re.split(r'[。\n]', answer_text)
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for brand, metrics in raw_metrics.items():
            if metrics["mentioned"] and brand.lower() in sentence_lower:
                if any(word in sentence_lower for word in
                       ["首选", "最佳", "强烈推荐", "best", "top pick", "most recommended"]):
                    raw_metrics[brand]["is_strong"] = 1

    # 3. 提取引用深度
    if references:
        for brand, metrics in raw_metrics.items():
            if metrics["mentioned"]:
                for ref in references:
                    # 简化逻辑：如果引用链接中包含品牌名，则认为相关
                    if brand.lower() in ref.lower():
                        raw_metrics[brand]["ref_count"] += 1

    return raw_metrics


# ==============================================================================
# main 函数 (v4.0 - 全局直接分析版)
# ==============================================================================
def main():
    """主分析函数 - 直接分析所有数据，生成唯一的总榜单"""
    print("--- 使用 v4.0 全局直接分析模型 ---")

    # 1. 加载数据
    try:
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 未找到 '{RESULTS_FILE}' 文件。")
        return

    # 2. 全局指标收集：将所有回答视为一个整体
    print("阶段一：正在全局收集中间指标...")
    all_brands_raw_metrics = defaultdict(lambda: {
        "total_mentions": 0,
        "first_pos_sum": 0,
        "strong_recommend_count": 0,
        "total_ref_count": 0,
        "mention_in_answers": 0,  # 在多少个回答中出现过
    })

    total_brand_mentions_across_all = 0

    for item in data:
        answer = item.get("response", {}).get("answer", "")
        references = item.get("response", {}).get("references", [])
        if not answer: continue

        # 对每个回答进行分析
        answer_metrics = analyze_single_answer(answer, references, BRAND_DICTIONARY)

        # 将指标累加到全局的品牌指标中
        for brand, metrics in answer_metrics.items():
            if brand in CHINESE_BRANDS_WHITELIST:
                brand_global_metrics = all_brands_raw_metrics[brand]
                brand_global_metrics["total_mentions"] += metrics["mention_count"]
                if metrics["first_pos"] != float('inf'):
                    brand_global_metrics["first_pos_sum"] += metrics["first_pos"]
                brand_global_metrics["strong_recommend_count"] += metrics["is_strong"]
                brand_global_metrics["total_ref_count"] += metrics["ref_count"]
                brand_global_metrics["mention_in_answers"] += 1
                total_brand_mentions_across_all += metrics["mention_count"]

    # 3. 全局计分：在所有品牌之间进行一次性标准化和计分
    print("阶段二：正在进行全局计分...")
    final_scores = {}

    if all_brands_raw_metrics:
        # 计算全局的最大值和最小值用于标准化
        max_mentions = max(m["total_mentions"] for m in all_brands_raw_metrics.values())
        min_pos_avg = min(m["first_pos_sum"] / m["mention_in_answers"] for m in all_brands_raw_metrics.values() if
                          m["mention_in_answers"] > 0)
        max_strong = max(m["strong_recommend_count"] for m in all_brands_raw_metrics.values())
        max_refs = max(m["total_ref_count"] for m in all_brands_raw_metrics.values())

        for brand, metrics in all_brands_raw_metrics.items():
            scores = {}
            avg_pos = metrics["first_pos_sum"] / metrics["mention_in_answers"] if metrics[
                                                                                      "mention_in_answers"] > 0 else float(
                'inf')

            # 计算六大维度的得分
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

            # 将所有需要的数据存入final_scores
            final_scores[brand] = {
                "品牌指数": total_score,
                "总提及次数": metrics["total_mentions"],
                "出现次数": metrics["mention_in_answers"],
            }

    # 4. 生成最终的唯一报告
    print("阶段三：正在生成最终报告...")
    with open(RANKING_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# 中国出海品牌AI认知指数--家用电器类\n\n")
        f.write(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("| 排名 | 品牌名称 | 品牌指数 | 总提及次数 | 出现次数 |\n")
        f.write("|:---:|:---|:---:|:---:|:---:|\n")

        sorted_brands = sorted(final_scores.items(), key=lambda x: x[1]["品牌指数"], reverse=True)

        for rank, (brand, scores_data) in enumerate(sorted_brands, 1):
            f.write(
                f"| {rank} | {brand} | **{scores_data['品牌指数']:.2f}** | {scores_data['总提及次数']} | {scores_data['出现次数']} |\n")

        f.write("\n")

    print(f"分析完成！全局总榜单已保存到 '{RANKING_OUTPUT_FILE}'。")


if __name__ == "__main__":
    main()
