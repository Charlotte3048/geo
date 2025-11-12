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
WEIGHTS = {
    "mention": 1.0,  # 基础提及分
    "rank": 5.0,  # 顺位排名附加分
    "strength": 3.0  # 推荐强度附加分
}


def calculate_scores(answer: str, brand_map: dict) -> dict:
    """为单篇回答中出现的品牌计分"""
    scores = defaultdict(float)
    answer_lower = answer.lower()

    # 1. 提及分
    mentioned_brands = set()
    for std_brand, aliases in brand_map.items():
        for alias in aliases:
            if alias.lower() in answer_lower:
                scores[std_brand] += WEIGHTS["mention"]
                mentioned_brands.add(std_brand)
                break  # 一个品牌只加一次基础分

    # 2. 顺位排名分和强度分
    # 简化处理：我们检查句子中的排名和推荐词
    sentences = re.split(r'[。\n]', answer)
    rank_order = 1
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for brand in mentioned_brands:
            if brand.lower() in sentence_lower:
                # 强度分
                if any(word in sentence_lower for word in ["首选", "最佳", "强烈推荐", "best", "top pick"]):
                    scores[brand] += WEIGHTS["strength"]

                # 排名分 (简化版：按句子顺序和关键词)
                if any(word in sentence_lower for word in [f"第{rank_order}", f"{rank_order}.", "首先", "第一"]):
                    scores[brand] += WEIGHTS["rank"] * (1 / rank_order)
                    rank_order += 1
    return scores


# --- 4. 主执行逻辑 ---
def main():
    """主分析函数"""
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("错误：请先设置环境变量 'OPENAI_API_KEY'。")
        return

    client = openai.OpenAI()

    # 加载数据
    try:
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 未找到 '{RESULTS_FILE}' 文件。")
        return

    all_answers = [item['response']['answer'] for item in data if 'response' in item and 'answer' in item['response']]

    # --- 运行第一阶段：智能品牌发现 ---
    # 注意：这一步会调用AI，产生费用。如果已生成 `confirmed_brands.txt` 且满意，可以注释掉下面这行。
    # discover_and_confirm_brands(client, all_answers)

    print("\n--- 请在脚本中完成 BRAND_DICTIONARY 的填充，然后再次运行此脚本 ---")
    # 检查词典是否已填充，如果未填充，则提示并退出
    if len(BRAND_DICTIONARY) <= 12:  # 检查是否还是初始状态
        print("提示：BRAND_DICTIONARY 似乎尚未填充。请根据 " + CONFIRMED_BRANDS_FILE + " 的内容进行更新。")
        return

    print("\n--- 阶段二: 量化计分与排名 ---")

    category_scores = defaultdict(lambda: defaultdict(float))

    for item in data:
        category = item.get("category", "Uncategorized")
        answer = item.get("response", {}).get("answer", "")
        if not answer:
            continue

        # answer_scores 包含所有被识别品牌的分数（包括国际品牌）
        answer_scores = calculate_scores(answer, BRAND_DICTIONARY)

        for brand, score in answer_scores.items():
            # **核心过滤步骤**：只有在白名单中的品牌，其分数才会被计入最终结果
            if brand in CHINESE_BRANDS_WHITELIST:
                category_scores[category][brand] += score

        # --- 新增：总榜单计算逻辑 ---
    print("正在计算总榜单...")
    total_scores = defaultdict(float)

    # 1. 计算每个品类的标准化得分
    for category, scores in category_scores.items():
        if not scores:
            continue

        # 找到品类冠军的得分
        max_score = max(scores.values())
        if max_score == 0:
            continue

        # 计算该品类下所有品牌的标准化得分，并累加到总分中
        for brand, score in scores.items():
            normalized_score = (score / max_score) * 100
            total_scores[brand] += normalized_score

        # --- 生成并保存包含总榜单的最终报告 ---
        with open(RANKING_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("# 中国出海品牌GenAI心智占有率排行榜\n\n")
            f.write(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # --- 写入总榜单 ---
            f.write("## ⭐️ 家用电器总榜单 (综合排名) ⭐️\n\n")
            f.write("此榜单通过标准化各品类得分并加权汇总，反映品牌的跨品类综合影响力。\n\n")
            f.write("| 排名 | 品牌 | 综合影响力得分 |\n")
            f.write("|:---:|:---|:---|\n")
            sorted_total_brands = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (brand, score) in enumerate(sorted_total_brands, 1):
                f.write(f"| {rank} | {brand} | {score:.2f} |\n")
            f.write("\n---\n\n")  # 分隔线

            # --- 写入各子品类榜单 ---
            f.write("## 各子品类详细榜单\n\n")
            for category, scores in sorted(category_scores.items()):
                f.write(f"### 品类: {category}\n\n")
                if not scores:
                    f.write("该品类下未发现可计分的中国品牌。\n\n")
                    continue
                f.write("| 排名 | 品牌 | AI心智占有率得分 (原始分) |\n")
                f.write("|:---:|:---|:---|\n")
                sorted_brands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for rank, (brand, score) in enumerate(sorted_brands, 1):
                    f.write(f"| {rank} | {brand} | {score:.2f} |\n")
                f.write("\n")

    print(f"分析完成！仅包含中国品牌的最终排行榜已保存到 '{RANKING_OUTPUT_FILE}'。")


if __name__ == "__main__":
    main()
