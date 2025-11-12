import json
import spacy
from collections import Counter
import re

# --- 配置 ---
RESULTS_FILE = "results.json"
OUTPUT_FILE = "candidate_brands_cleaned.txt"  # 使用新的输出文件名以示区别

# --- 加载语言模型 ---
try:
    nlp_en = spacy.load("en_core_web_sm")
    nlp_zh = spacy.load("zh_core_web_sm")
    print("成功加载 en_core_web_sm 和 zh_core_web_sm 模型。")
except OSError:
    print("错误：请先下载spaCy语言模型...")
    exit()

# --- 优化后的噪音词列表 ---
# 我们将您发现的噪音词以及其他常见噪音词加入这个集合
NOISE_WORDS = {
    # 1. 常见地理位置和组织
    "amazon", "google", "apple", "facebook", "microsoft", "europe", "china", "us",
    "usa", "america", "north america", "ces", "uk", "germany", "france", "eu",

    # 2. 常见非品牌术语 (技术、功能、通用词)
    "robot", "vacuum", "cleaner", "model", "brand", "company", "price", "app",
    "tech", "technology", "consumer", "product", "series", "pro", "ultra", "plus",
    "home", "kitchen", "appliances", "review", "website", "page", "com", "https",
    "ul", "tra", "max", "id", "http", "https", "www",
    "ai", "model", "market", "user", "customer", "service", "models", "brands",
    "design", "performance", "quality", "feature", "features", "system", "systems",
    "function", "functions", "ability", "navigation", "identification", "performance",
    "mopping", "mop", "station", "base", "camera", "laser", "lds", "vslam",

    # 3. 权威媒体和网站
    "the verge", "cnet", "rtings.com", "rtings", "forbes", "techradar", "pcmag", "wired",
    "youtube", "reddit",

    # 4. 中文噪音词
    "机器人", "吸尘器", "品牌", "公司", "型号", "市场", "用户", "性价比", "中国",
    "科技", "有限公司", "产品", "系列", "问题", "答案", "推荐", "区别", "选择",
    "一个", "一些", "几个", "这个", "那个", "什么", "怎么样", "我们", "他们",
    "技术", "智能", "功能", "系统", "识别", "能力", "清洁", "拖布", "导航",
    "表现", "基站", "方面", "体验", "效果", "优势", "核心", "旗舰", "自动",
    "避障", "算法", "版本", "价格", "评测", "媒体", "网站", "网页", "链接",
}


def is_valid_candidate(text: str) -> bool:
    """增强版的候选词有效性检查"""
    text_lower = text.lower().strip()

    # 1. 过滤掉噪音词
    if text_lower in NOISE_WORDS:
        return False
    # 2. 过滤掉太短或太长的词
    if len(text_lower) < 2 or len(text_lower) > 20:
        return False
    # 3. 过滤掉纯数字或主要包含数字的词
    if text_lower.isdigit() or sum(c.isdigit() for c in text_lower) > len(text_lower) / 2:
        return False
    # 4. 过滤掉不含任何字母的词 (允许中文)
    if not any(c.isalpha() for c in text_lower):
        return False
    # 5. 过滤掉包含非法字符的词
    if not re.match(r'^[a-zA-Z0-9\-\.\s\u4e00-\u9fa5]+$', text_lower):
        return False
    # 6. 过滤掉以非字母数字开头的词
    if not text_lower[0].isalnum():
        return False

    return True


def process_entities(doc, candidate_list):
    """从spaCy的doc对象中提取实体和专有名词"""
    # 提取命名实体 (ORG, PRODUCT)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            candidate_list.append(ent.text)
    # 提取专有名词 (PROPN)
    for token in doc:
        if token.pos_ == "PROPN":
            candidate_list.append(token.text)


def main():
    """主执行函数"""
    print(f"正在从 '{RESULTS_FILE}' 文件中加载数据...")
    try:
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 处理 '{RESULTS_FILE}' 文件时出错: {e}")
        return

    all_answers = [item['response']['answer'] for item in data if 'response' in item and 'answer' in item['response']]
    print(f"已加载 {len(all_answers)} 条AI回答。")

    candidate_brands = []
    print("正在使用 spaCy 提取候选品牌词...")
    for answer in all_answers:
        # 统一处理，先用英文模型，再用中文模型补充
        process_entities(nlp_en(answer), candidate_brands)
        process_entities(nlp_zh(answer), candidate_brands)

    print("提取完成。正在进行清洗和频率统计...")

    # --- 这里是修正的部分 ---
    # 使用一个清晰的 for 循环代替复杂的列表推导式
    valid_brands = []
    for brand in candidate_brands:
        cleaned_brand = brand.strip()
        if is_valid_candidate(cleaned_brand):
            valid_brands.append(cleaned_brand)

    # 现在对清洗过的列表进行频率统计
    brand_counts = Counter(valid_brands)
    # --- 修正结束 ---

    print(f"清洗后，发现 {len(brand_counts)} 个独特的候选品牌。")

    # 保存到文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# GenAI 候选品牌列表 (已优化过滤, 按出现频率排序)\n")
        f.write("# --------------------------------------------------\n")
        f.write("# 请审核此列表以构建您的 BRAND_DICTIONARY。\n\n")

        if not brand_counts:
            f.write("未发现有效的候选品牌。")
        else:
            for brand, count in brand_counts.most_common():
                f.write(f"{brand} ({count})\n")

    print(f"结果已成功保存到 '{OUTPUT_FILE}' 文件中。")
    print("\n下一步：请手动审核该文件，以创建或更新您的品牌词典。")


if __name__ == "__main__":
    main()
