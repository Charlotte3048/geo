import openai
import json
import os
import re
from collections import Counter
import time

# ==============================================================================
# 纯净智能品牌探索脚本 (v3.0 - Pure LLM)
# 描述: 从零开始，仅依赖大语言模型从 results.json 中智能提取候选品牌。
# 确保在运行脚本前设置了以下环境变量，在当前目录的终端中执行：:
# export OPENAI_API_KEY="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"
# export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
# ==============================================================================

# --- 1. 配置 ---
RESULTS_FILE = "results.json"
OUTPUT_FILE = "candidate_brands_pure_llm.txt"

# 用于智能提取的模型 (推荐使用能力强、性价比高的模型)
# google/gemini-pro 是一个非常好的选择
EXTRACTION_MODEL = "google/gemini-2.5-flash"

# 将文本分块的大小 (字符数)
CHUNK_SIZE = 2000


# --- 2. 智能品牌发现 (LLM-Powered) ---
def get_brands_from_chunk_with_ai(client: openai.OpenAI, text_chunk: str) -> list:
    """
    使用强大的AI模型从一小块文本中提取品牌名称。
    """
    system_prompt = """
    你是一个高度精准的品牌名称识别引擎。你的唯一任务是从给定的文本中，抽取出所有明确的、代表公司或产品的【品牌名称】。

    **严格遵守以下规则:**
    1.  **只返回品牌名**: 例如 "Roborock", "TCL", "美的", "Anker"。
    2.  **坚决过滤非品牌词**:
        *   **技术术语**: 忽略 "Mini LED", "QLED", "Android"。
        *   **地理位置**: 忽略 "北美", "欧洲", "China"。
        *   **通用名词**: 忽略 "电视", "技术", "系列", "公司", "性价比"。
        *   **网站与媒体**: 忽略 "Amazon", "Rtings.com", "The Verge"。
    3.  **合并常见别名**: 如果发现 "石头科技" 和 "Roborock" 同时出现，请尽量只返回 "Roborock"。如果发现 "小米" 和 "Mijia"，请尽量只返回 "Xiaomi"。
    4.  **输出格式**: 必须以一个JSON数组的格式返回你发现的品牌名，例如: ["Roborock", "Ecovacs", "Midea"]。如果未发现任何品牌，则返回空数组 []。
    """
    try:
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_chunk}
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
        # 在处理单个块时出错，打印错误但继续
        print(f"    - 智能提取块时发生错误: {e}")
        return []


# --- 3. 主执行函数 ---
def main():
    """主执行函数"""
    # --- 加载原始数据 ---
    print(f"正在从 '{RESULTS_FILE}' 文件中加载数据...")
    try:
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_answers = [item['response']['answer'] for item in data if
                       'response' in item and 'answer' in item['response']]
        print(f"已加载 {len(all_answers)} 条AI回答。")
    except Exception as e:
        print(f"错误: 处理 '{RESULTS_FILE}' 文件时出错: {e}")
        return

    # --- 创建OpenAI客户端 ---
    try:
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        print(f"使用模型 '{EXTRACTION_MODEL}' 进行品牌提取。")
    except TypeError:
        print("错误: 请确保已设置 OPENROUTER_API_KEY 环境变量。")
        return

    # --- 开始智能提取 ---
    all_discovered_brands = []
    print("\n--- 正在使用大语言模型智能提取品牌 ---")

    for i, answer in enumerate(all_answers):
        print(f"正在处理回答 {i + 1}/{len(all_answers)}...")
        if not answer.strip():
            continue

        # 将长回答分块处理
        for j in range(0, len(answer), CHUNK_SIZE):
            chunk = answer[j:j + CHUNK_SIZE]
            print(f"  - 正在处理块 {j // CHUNK_SIZE + 1}...")

            new_brands = get_brands_from_chunk_with_ai(client, chunk)

            if new_brands:
                print(f"    + 发现候选: {new_brands}")
                all_discovered_brands.extend(new_brands)

            time.sleep(1)  # 礼貌地等待，避免速率超限

    # --- 汇总与统计 ---
    print("\n--- 正在汇总、清洗和统计 ---")

    # 对所有发现的品牌进行最终的频率统计
    final_brand_counts = Counter()
    for brand in all_discovered_brands:
        cleaned_brand = brand.strip()
        # 进行一次非常基础的最终清洗
        if len(cleaned_brand) > 1 and not cleaned_brand.isdigit():
            final_brand_counts[cleaned_brand] += 1

    print(f"分析完成！共发现 {len(final_brand_counts)} 个独特的候选品牌。")

    # --- 保存结果 ---
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# 纯净智能品牌探索报告 (v3.0 - Pure LLM)\n")
        f.write("# ----------------------------------------\n")
        f.write("# 请审核此列表，以创建或完善您的 config.yaml 文件。\n\n")

        if not final_brand_counts:
            f.write("未发现有效的候选品牌。")
        else:
            for brand, count in final_brand_counts.most_common():
                f.write(f"{brand} ({count})\n")

    print(f"结果已成功保存到 '{OUTPUT_FILE}' 文件中。")
    print("\n下一步：请手动审核该文件，以创建您的品牌词典和白名单。")


if __name__ == "__main__":
    main()
