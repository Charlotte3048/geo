import openai
import json
import os
import re
import argparse
from collections import Counter
import time
import glob
from dotenv import load_dotenv

# ==============================================================================
# 品牌探索引擎 : 从 AI 回答中提取品牌名称，生成配置文件模板
# ==============================================================================
# 示例用法:
# 请确认在oversea目录下运行，若不在oversea目录下，请先在控制台中运行如下命令切换目录：
# cd oversea
# python explore_brand_oversea.py --task ha --category_prefix "家用电器"
# python explore_brand_oversea.py --task sh --category_prefix "智能硬件"
# python explore_brand_oversea.py --task ha --category_prefix "家用电器" --results_file results/results_merged_ha_20260114.json
# ==============================================================================

# 假设您的项目根目录是 oversea
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 加载 .env 文件（从根目录加载）
root_dir = os.path.dirname(BASE_DIR)  # 获取父目录（根目录）
load_dotenv(os.path.join(root_dir, '.env'))

# 默认模型
DEFAULT_MODEL = "google/gemini-2.5-flash"

# 预设的品类 Prompt 模板
CATEGORY_PROMPTS = {
    "ha": """
你是一个专业的市场分析师。你的任务是从给定的文本中，提取所有清晰的家用电器品牌名称。
规则:
1. 只返回家用电器品牌名称，例如: "Midea", "Haier", "Samsung", "LG", "Sony", "Panasonic"。
2. 包括但不限于以下品类的品牌: 电视、冰箱、洗衣机、空调、厨房电器、清洁电器、音频设备等。
3. 忽略非品牌的通用词汇，例如: "电视机", "冰箱", "空调", "智能家居" 等品类名称。
4. 忽略零售商/平台名称，例如: "Amazon", "Best Buy", "京东", "天猫"。
5. 同时识别中英文品牌名称，例如: "美的" 和 "Midea" 都应该提取。
6. 返回一个 JSON 对象，格式为: {"brands": ["Brand1", "Brand2", ...]}；如果没有找到任何品牌，返回 {"brands": []}。
""",
    "sh": """
你是一个专业的市场分析师。你的任务是从给定的文本中，提取所有清晰的智能硬件品牌名称。
规则:
1. 只返回智能硬件品牌名称，例如: "DJI", "Huawei", "Xiaomi", "Anker", "Roborock"。
2. 包括但不限于以下品类的品牌: 无人机、智能手表、智能音箱、扫地机器人、充电设备、3D打印机、便携储能等。
3. 忽略非品牌的通用词汇，例如: "无人机", "智能手表", "扫地机器人" 等品类名称。
4. 忽略零售商/平台名称，例如: "Amazon", "Best Buy", "京东", "天猫"。
5. 同时识别中英文品牌名称，例如: "大疆" 和 "DJI" 都应该提取。
6. 返回一个 JSON 对象，格式为: {"brands": ["Brand1", "Brand2", ...]}；如果没有找到任何品牌，返回 {"brands": []}。
"""
}

# 品类名称映射
CATEGORY_NAMES = {
    "ha": "Home Appliance",
    "sh": "Smart Hardware"
}


def find_latest_results_file(task: str) -> str:
    """查找指定任务的最新结果文件"""
    results_dir = os.path.join(BASE_DIR, "results")
    pattern = os.path.join(results_dir, f"results_merged_{task}_*.json")
    files = glob.glob(pattern)

    if not files:
        return None

    # 按文件名排序（日期戳在文件名中），取最新的
    files.sort(reverse=True)
    return files[0]


def get_brands_from_text_with_ai(client: openai.OpenAI, text: str, model: str, system_prompt: str) -> list:
    """使用指定的AI模型从文本中提取品牌名称"""
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


def generate_config_template(config_file_path: str, task_name: str, brand_counts: Counter, results_file: str):
    """根据品牌列表，生成一个带注释的YAML配置文件模板"""
    category_name = CATEGORY_NAMES.get(task_name, task_name.replace('_', ' ').title())

    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(f"# {category_name} 品类分析配置文件\n")
        f.write("# ========================================================\n")
        f.write(f"# 自动生成于: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# 请仔细审核和完善此文件，特别是 brand_dictionary 和 brands_whitelist。\n\n")

        f.write(f"task_name: {task_name}\n")
        f.write(f"results_file: {results_file}\n")
        f.write(f"ranking_output_file: ranking_report_{task_name}.md\n")
        f.write(f"report_title: '# 中国出海品牌AI认知指数 -- {category_name}类'\n\n")

        # 添加权重配置
        f.write("weights:\n")
        f.write("  brand_prominence: 20\n")
        f.write("  share_of_voice: 20\n")
        f.write("  top10_visibility: 20\n")
        f.write("  competitiveness: 20\n")
        f.write("  sentiment_analysis: 20\n\n")

        f.write("\n# 步骤一: 完善品牌词典 (包含所有中外品牌及其别名)\n")
        f.write("brand_dictionary:\n")
        for brand, count in brand_counts.most_common():
            # 为高频词自动生成一个基础模板
            if count > 1:
                # 标准化品牌名称（首字母大写，去除空格）
                std_name = brand.strip().title().replace(' ', '')
                f.write(f"  {std_name}: [{brand.lower()}] # ({count}次)\n")
        f.write("\n  # --- 请在此处手动添加或合并别名 ---\n")
        f.write("  # Example: Anker: [anker, anker innovations, 安克]\n\n")

        f.write("# 步骤二: 定义中国品牌白名单 (仅使用标准键名)\n")
        f.write("brands_whitelist:\n")
        for brand, count in brand_counts.most_common():
            if count > 1:
                std_name = brand.strip().title().replace(' ', '')
                f.write(f"  - {std_name}\n")
        f.write("\n  # --- 请在此处审核，只保留中国品牌 ---\n")

    print(f"\n✅ 配置文件模板已生成: '{config_file_path}'")
    print("下一步关键操作：请打开并编辑此YAML文件，完成品牌词典和白名单的最终确认！")


def main():
    # --- 1. 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="品牌探索引擎 (v5.0)")
    parser.add_argument("--task", required=True, choices=["ha", "sh"],
                        help="任务名称: ha (家用电器) 或 sh (智能硬件)")
    parser.add_argument("--category_prefix", required=True,
                        help="用于筛选数据的分类名前缀 (例如: '家用电器' 或 '智能硬件')")
    parser.add_argument("--results_file", default=None,
                        help="结果文件路径 (默认自动查找 results/ 目录下最新的文件)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="用于品牌提取的LLM模型ID")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"品牌探索引擎 (v5.0)")
    print(f"{'=' * 60}\n")

    # --- 2. 设置API客户端 ---
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ 错误: 请先设置环境变量 'OPENROUTER_API_KEY'。")
        return
    client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # --- 3. 确定结果文件路径 ---
    if args.results_file:
        results_file = args.results_file
    else:
        results_file = find_latest_results_file(args.task)
        if not results_file:
            print(f"❌ 错误: 未找到任务 '{args.task}' 的结果文件。")
            print(f"   请确保 results/ 目录下存在 results_merged_{args.task}_*.json 文件，")
            print(f"   或使用 --results_file 参数指定文件路径。")
            return

    # --- 4. 获取品类对应的 Prompt ---
    system_prompt = CATEGORY_PROMPTS.get(args.task)
    if not system_prompt:
        print(f"❌ 错误: 未找到任务 '{args.task}' 的 Prompt 模板。")
        return

    category_name = CATEGORY_NAMES.get(args.task, args.task)

    print(f"📋 任务: {args.task} ({category_name})")
    print(f"📂 分类前缀: {args.category_prefix}")
    print(f"📄 结果文件: {results_file}")
    print(f"🤖 模型: {args.model}\n")

    # --- 5. 加载并筛选数据 ---
    print(f"正在加载数据...")
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 数据文件 '{results_file}' 未找到。")
        return

    filtered_data = [item for item in all_data if item.get('category', '').startswith(args.category_prefix)]

    if not filtered_data:
        print(f"⚠️ 警告: 未找到任何分类以 '{args.category_prefix}' 开头的条目。")
        print(f"   请检查前缀是否正确。")
        return

    all_answers = [item['response']['answer'] for item in filtered_data if
                   'response' in item and 'answer' in item['response']]
    print(f"✅ 筛选完成！共找到 {len(all_answers)} 条相关回答进行分析。\n")

    # --- 6. 智能品牌提取 ---
    print(f"{'─' * 50}")
    print(f"开始智能品牌提取...")
    print(f"{'─' * 50}")

    all_extracted_brands = []
    for i, answer in enumerate(all_answers):
        print(f"  [{i + 1}/{len(all_answers)}] 正在处理...")
        brands = get_brands_from_text_with_ai(client, answer, args.model, system_prompt)
        if brands:
            all_extracted_brands.extend(brands)
            print(f"      提取到 {len(brands)} 个品牌")
        else:
            print(f"      未提取到品牌")
        time.sleep(0.5)  # 轻微延迟以示友好

    brand_counts = Counter(all_extracted_brands)
    print(f"\n✅ 提取完成！共发现 {len(brand_counts)} 个独特的候选品牌。\n")

    # --- 7. 生成配置文件模板 (保存到 config 目录) ---
    config_dir = os.path.join(BASE_DIR, "config")
    os.makedirs(config_dir, exist_ok=True)

    config_template_file = os.path.join(config_dir, f"config_{args.task}.yaml")

    # 生成配置文件模板
    generate_config_template(config_template_file, args.task, brand_counts, results_file)

    # --- 8. 显示统计信息 ---
    print(f"\n{'=' * 60}")
    print(f"📈 统计信息:")
    print(f"   - 分析回答数: {len(all_answers)}")
    print(f"   - 提取品牌总数: {len(all_extracted_brands)}")
    print(f"   - 独特品牌数: {len(brand_counts)}")
    print(f"   - 配置模板文件: {config_template_file}")

    # 显示 Top 10 品牌
    print(f"\n🏆 Top 10 高频品牌:")
    print(f"{'─' * 40}")
    for i, (brand, count) in enumerate(brand_counts.most_common(10), 1):
        print(f"   {i:2d}. {brand:20s} ({count}次)")

    print(f"\n{'=' * 60}")
    print("✅ 品牌探索完成！")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
