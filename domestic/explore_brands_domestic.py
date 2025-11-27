import openai
import json
import os
import re
import argparse
import time
import yaml  # 新增导入 yaml
from collections import Counter
from dotenv import load_dotenv  # 新增导入 dotenv

# ==============================================================================
# 国内榜单品牌探索引擎 (v1.0 - 针对已合并文件)
# 描述: 直接分析已按品类合并的 JSON 文件，并生成品牌词典模板。
# Moonshot-Kimi-K2-Instruct
# export KIMI_API_KEY="sk-06829e4221f84e878e6a3f207d60aa8c"
# export KIMI_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
# cd domestic
# python explore_brands_domestic.py --task nev --results_file results_nev_merged.json
# python explore_brands_domestic.py --task scenic --results_file results_scenic_merged.json
# ==============================================================================

# --- 配置 ---
# 假设 .env 文件在项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(os.path.dirname(BASE_DIR), '.env'))

# 建议使用 Kimi 或 DeepSeek 进行中文品牌提取，因为它们对中文支持较好
DEFAULT_MODEL_NAME = "Moonshot-Kimi-K2-Instruct"
DEFAULT_MODEL_KEY_ENV = "KIMI_API_KEY"
DEFAULT_MODEL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# Kimi 的 Base URL


# --- 核心函数 ---
def get_brands_from_text_with_ai(client: openai.OpenAI, text: str, model: str) -> list:
    """使用指定的AI模型从文本中提取品牌名称"""
    system_prompt = """
    你是一个专业的市场分析师。你的任务是从给定的中文文本中，提取所有清晰的品牌名称、公司名称或景区名称。
    规则:
    1. 只返回品牌名称或景区名称，例如: "比亚迪", "故宫博物院", "蔚来"。
    2. 忽略技术术语 (例如: "三电系统"), 价格 (例如: "15万"), 车型名称 (例如: "Model 3"), 泛指名词 (例如: "SUV", "景区")。
    3. 返回一个 JSON 数组，例如: ["比亚迪", "理想", "故宫博物院"]。如果没有找到任何品牌或景区，返回空数组 []。
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            # 确保模型返回 JSON 格式
            response_format={"type": "json_object"},
        )

        # 尝试解析 JSON 字符串
        content_str = response.choices[0].message.content.strip()

        # 尝试处理可能返回的 {"brands": [...]} 或直接返回 [...]
        try:
            content = json.loads(content_str)
        except json.JSONDecodeError:
            # 如果解析失败，尝试用正则提取 JSON 块
            match = re.search(r'\[.*?\]', content_str, re.DOTALL)
            if match:
                content = json.loads(match.group(0))
            else:
                print(f"  - AI返回内容无法解析为JSON: {content_str[:50]}...")
                return []

        if isinstance(content, dict):
            # 兼容 {"brands": [...]} 格式
            return content.get("brands", [])
        elif isinstance(content, list):
            return content
        return []
    except Exception as e:
        print(f"  - AI extraction error: {e}")
        return []


def generate_brand_dictionary_template(config_file_path: str, task_name: str, brand_counts: Counter):
    """根据品牌列表，生成一个带注释的YAML品牌词典模板"""

    # 确保只包含出现次数大于 1 的品牌，减少噪音
    filtered_brands = {brand: count for brand, count in brand_counts.items() if count > 1}

    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(f"# {task_name.replace('_', ' ').title()} 品类品牌词典模板\n")
        f.write("# ========================================================\n")
        f.write("# 自动生成于: {}\n".format(time.strftime('%Y-%m-%d %H:%M:%S')))
        f.write("# 请仔细审核和完善此文件，特别是品牌别名 (Alias)。\n\n")

        f.write("brand_dictionary:\n")
        for brand, count in Counter(filtered_brands).most_common():
            # 自动生成一个基础模板
            # 键名使用原始提取的品牌名，方便用户修改
            f.write(f"  {brand}: [{brand.lower()}] # (出现 {count} 次)\n")
        f.write("\n  # --- 请在此处手动添加或合并别名 ---\n")
        f.write("  # Example: 蔚来: [蔚来, nio, 蔚来汽车]\n\n")

        f.write("# 步骤二: 定义品牌白名单 (仅使用标准键名)\n")
        f.write("brands_whitelist:\n")
        for brand, count in Counter(filtered_brands).most_common():
            f.write(f"  - {brand}\n")
        f.write("\n  # --- 请在此处审核，只保留最终要分析的品牌/景区 ---\n")

    print(f"\n✅ 品牌词典模板已生成: '{config_file_path}'")
    print("下一步关键操作：请打开并编辑此YAML文件，完成品牌词典和白名单的最终确认！")


def main():
    # --- 1. 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="国内榜单品牌探索引擎。")
    parser.add_argument("--task", required=True, help="定义本次分析任务的唯一名称 (例如: nev, scenic)。")
    parser.add_argument("--results_file", required=True,
                        help="包含所有原始数据的JSON文件 (例如: results_nev_merged.json)。")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="用于品牌提取的LLM模型名称 (例如: kimi)。")
    args = parser.parse_args()

    # --- 2. 设置API客户端 ---
    api_key = os.environ.get(DEFAULT_MODEL_KEY_ENV)
    if not api_key:
        print(f"错误: 请先设置环境变量 '{DEFAULT_MODEL_KEY_ENV}'。")
        return

    # 假设 Kimi 的模型 ID 就是 model 参数
    client = openai.OpenAI(api_key=api_key, base_url=DEFAULT_MODEL_BASE_URL)
    model_id = args.model  # 实际调用时，这里应该使用 Kimi 的模型 ID，但我们简化为使用 args.model

    # --- 3. 加载数据 ---
    print(f"--- 开始探索任务: {args.task} ---")
    results_file_path = os.path.join(BASE_DIR, args.results_file)
    print(f"正在从 '{results_file_path}' 加载数据...")
    try:
        with open(results_file_path, 'r', encoding='utf-8') as f:
            filtered_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 数据文件 '{results_file_path}' 未找到。")
        return

    if not filtered_data:
        print("警告: 数据文件为空。")
        return

    all_answers = [item['response']['answer'] for item in filtered_data if
                   'response' in item and 'answer' in item['response']]
    print(f"筛选完成！共找到 {len(all_answers)} 条相关回答进行分析。")

    # --- 4. 智能品牌提取 ---
    print(f"\n--- 开始智能品牌提取 (使用模型: {model_id}) ---")
    all_extracted_brands = []
    for i, answer in enumerate(all_answers):
        print(f"正在处理回答 {i + 1}/{len(all_answers)}...")
        # 注意：这里使用 Kimi 的模型 ID
        brands = get_brands_from_text_with_ai(client, answer, "Moonshot-Kimi-K2-Instruct")
        if brands:
            all_extracted_brands.extend(brands)
        time.sleep(0.5)  # 轻微延迟以示友好

    brand_counts = Counter(all_extracted_brands)
    print(f"\n提取完成！共发现 {len(brand_counts)} 个独特的候选品牌。")

    # --- 5. 生成隔离的输出文件 ---
    config_template_file = f"brand_dictionary_{args.task}.yaml"

    # 生成配置文件模板
    generate_brand_dictionary_template(config_template_file, args.task, brand_counts)


if __name__ == "__main__":
    main()
