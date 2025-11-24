import openai
import json
import os
import re
import argparse
from collections import Counter
import time

# ==============================================================================
# 最终品牌探索引擎 (v4.0 - 按任务隔离版)
# 描述: 可根据任务名称，筛选性地分析数据，并生成隔离的、带配置模板的输出。
# ==============================================================================
# export OPENROUTER_API_KEY="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"
# export OPENROUTER_API_BASE_URL="https://openrouter.ai/api/v1"
# 示例用法:
# python explore_brands.py --task smart_hardware --category_prefix "智能硬件-"
# python explore_brands.py --task home_appliance --category_prefix "家用电器-"
# ==============================================================================

# --- 配置 ---
# 这些现在是默认值，可以通过命令行参数覆盖
# oversea/results_merged_ha.json
DEFAULT_RESULTS_FILE = "results_merged_ha.json"
DEFAULT_MODEL = "google/gemini-2.5-flash"  # 默认使用高性价比模型


# --- 核心函数 ---
def get_brands_from_text_with_ai(client: openai.OpenAI, text: str, model: str) -> list:
    """使用指定的AI模型从文本中提取品牌名称"""
    system_prompt = """
    You are a professional market analyst. Your task is to extract all clear brand names representing companies or products from the given text.
    Rules:
    1. Return only brand names, e.g., "Roborock", "TCL", "美的".
    2. Ignore technical terms (e.g., "Mini LED"), geographical locations (e.g., "North America"), generic nouns (e.g., "TV", "technology"), and website names (e.g., "Amazon").
    3. If a brand has multiple names (e.g., "石头科技" and "Roborock"), extract them all.
    4. Return as a JSON array, e.g., ["Roborock", "Ecovacs", "美的"]. Return an empty array [] if no brands are found.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = json.loads(response.choices[0].message.content)
        if isinstance(content, dict):
            return content.get("brands", [])
        elif isinstance(content, list):
            return content
        return []
    except Exception as e:
        print(f"  - AI extraction error: {e}")
        return []


def generate_config_template(config_file_path: str, task_name: str, brand_counts: Counter):
    """根据品牌列表，生成一个带注释的YAML配置文件模板"""
    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(f"# {task_name.replace('_', ' ').title()} 品类分析配置文件\n")
        f.write("# ========================================================\n")
        f.write("# 自动生成于: {}\n".format(time.strftime('%Y-%m-%d %H:%M:%S')))
        f.write("# 请仔细审核和完善此文件，特别是 brand_dictionary 和 chinese_brands_whitelist。\n\n")

        f.write(f"task_name: {task_name}\n")
        f.write(f"results_file: {DEFAULT_RESULTS_FILE}\n")
        f.write(f"ranking_output_file: ranking_report_{task_name}.md\n")
        f.write(f"report_title: '# 中国出海品牌GenAI心智占有率 -- {task_name.replace('_', ' ').title()}类'\n\n")

        f.write("weights:\n")
        f.write("  visibility: 20\n")
        f.write("  mention_rate: 20\n")
        f.write("  ai_ranking: 20\n")
        f.write("  ref_depth: 15\n")
        f.write("  mind_share: 15\n")
        f.write("  competitiveness: 10\n\n")

        f.write("# 步骤一: 完善品牌词典 (包含所有中外品牌及其别名)\n")
        f.write("brand_dictionary:\n")
        for brand, count in brand_counts.most_common():
            # 为高频词自动生成一个基础模板
            if count > 1:
                f.write(f"  {brand.capitalize().replace(' ', '')}: [{brand.lower()}] # ({count}次)\n")
        f.write("\n  # --- 请在此处手动添加或合并别名 ---\n")
        f.write("  # Example: Anker: [anker, anker innovations, 安克]\n\n")

        f.write("# 步骤二: 定义中国品牌白名单 (仅使用标准键名)\n")
        f.write("chinese_brands_whitelist:\n")
        for brand, count in brand_counts.most_common():
            if count > 1:
                f.write(f"  - {brand.capitalize().replace(' ', '')}\n")
        f.write("\n  # --- 请在此处审核，只保留中国品牌 ---\n")

    print(f"\n✅ 配置文件模板已生成: '{config_file_path}'")
    print("下一步关键操作：请打开并编辑此YAML文件，完成品牌词典和白名单的最终确认！")


def main():
    # --- 1. 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="最终品牌探索引擎 (v4.0 - 按任务隔离版)。")
    parser.add_argument("--task", required=True, help="定义本次分析任务的唯一名称 (例如: home_appliance)。")
    parser.add_argument("--category_prefix", required=True, help="用于筛选数据的分类名前缀 (例如: '家用电器-')。")
    parser.add_argument("--results_file", default=DEFAULT_RESULTS_FILE, help="包含所有原始数据的JSON文件。")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="用于品牌提取的LLM模型ID。")
    args = parser.parse_args()

    # --- 2. 设置API客户端 ---
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("错误: 请先设置环境变量 'OPENROUTER_API_KEY'。")
        return
    client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # --- 3. 加载并筛选数据 ---
    print(f"--- 开始探索任务: {args.task} ---")
    print(f"正在从 '{args.results_file}' 加载数据，并筛选分类前缀为 '{args.category_prefix}' 的条目...")
    try:
        with open(args.results_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 数据文件 '{args.results_file}' 未找到。")
        return

    filtered_data = [item for item in all_data if item.get('category', '').startswith(args.category_prefix)]

    if not filtered_data:
        print(f"警告: 未找到任何分类以 '{args.category_prefix}' 开头的条目。请检查前缀是否正确。")
        return

    all_answers = [item['response']['answer'] for item in filtered_data if
                   'response' in item and 'answer' in item['response']]
    print(f"筛选完成！共找到 {len(all_answers)} 条相关回答进行分析。")

    # --- 4. 智能品牌提取 ---
    print(f"\n--- 开始智能品牌提取 (使用模型: {args.model}) ---")
    all_extracted_brands = []
    for i, answer in enumerate(all_answers):
        print(f"正在处理回答 {i + 1}/{len(all_answers)}...")
        brands = get_brands_from_text_with_ai(client, answer, args.model)
        if brands:
            all_extracted_brands.extend(brands)
        time.sleep(0.5)  # 轻微延迟以示友好

    brand_counts = Counter(all_extracted_brands)
    print(f"\n提取完成！共发现 {len(brand_counts)} 个独特的候选品牌。")

    # --- 5. 生成隔离的输出文件 ---
    candidate_file = f"candidate_brands_{args.task}.txt"
    config_template_file = f"config_{args.task}.yaml"

    # 生成候选品牌文本文件
    with open(candidate_file, 'w', encoding='utf-8') as f:
        f.write(f"# {args.task.replace('_', ' ').title()} 品类候选品牌列表 (v4.0)\n")
        f.write("# ----------------------------------------\n")
        for brand, count in brand_counts.most_common():
            f.write(f"{brand} ({count})\n")
    print(f"✅ 候选品牌列表已保存到: '{candidate_file}'")

    # 生成配置文件模板
    generate_config_template(config_template_file, args.task, brand_counts)


if __name__ == "__main__":
    main()
