import openai
import json
import os
import time
import argparse
import yaml  # 引入PyYAML


# ==============================================================================
# 终极数据采集引擎 (v1.0 - 任务驱动版)
# 描述: 一个由中央配置文件驱动的、按任务隔离的、支持断点续传的数据采集脚本。
# ==============================================================================
# 示例用法:
# python run_analysis.py --task ha_perplexity
# python run_analysis.py --task sh_perplexity
# export OPENROUTER_API_KEY="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"
# export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
# ==============================================================================

# --- API 调用核心函数 ---
def get_ai_response(client: openai.OpenAI, question: str, model: str, retries=3, delay=5):
    """调用AI模型获取回答，包含重试逻辑"""
    for attempt in range(retries):
        try:
            print(f"    - 正在调用模型 '{model}' (尝试 {attempt + 1}/{retries})...")
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=0.1,  # 使用较低的温度以获得更稳定的回答
            )
            # 假设返回的格式包含 answer 和 references
            # 注意：不同模型的返回结构可能不同，这里做一个兼容性假设
            # 真实的解析可能需要根据模型适配
            response_content = completion.choices[0].message.content

            # 这是一个简化的解析，实际可能需要更复杂的逻辑
            # 例如，Perplexity的引用格式可能不同
            references = re.findall(r'https?://[^\s )]+', response_content)

            return {
                "answer": response_content,
                "references": references
            }
        except Exception as e:
            print(f"    - 发生API错误 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"    - 等待 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                print("    - 所有重试均失败。")
                return None


# --- 主执行函数 ---
def main():
    # --- 1. 解析命令行参数 ---
    parser = argparse.ArgumentParser(description="终极数据采集引擎 (v1.0 - 任务驱动版)。")
    parser.add_argument("--task", required=True, help="指定要执行的任务名称 (例如: ha_gpt)。")
    parser.add_argument("--config", default="config.yaml", help="中央配置文件的路径。")
    args = parser.parse_args()

    # --- 2. 加载中央配置文件 ---
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 中央配置文件 '{args.config}' 未找到。")
        return

    # --- 3. 获取当前任务的配置 ---
    task_config = config.get('tasks', {}).get(args.task)
    if not task_config:
        print(f"错误: 在 '{args.config}' 中未找到任务 '{args.task}' 的配置。")
        return

    model_key = task_config.get('model_key')
    category_prefix = task_config.get('category_prefix')
    model_id = config.get('models', {}).get(model_key)
    questions_file_path = config.get('global_settings', {}).get('questions_file')

    if not all([model_key, category_prefix, model_id, questions_file_path]):
        print("错误: 任务配置不完整，缺少 model_key, category_prefix, model_id 或 questions_file。")
        return

    print(f"--- 开始执行采集任务: {args.task} ---")
    print(f"  - 模型: {model_id}")
    print(f"  - 分类前缀: '{category_prefix}'")

    # --- 4. 设置API客户端 ---
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("错误: 请先设置环境变量 'OPENROUTER_API_KEY'。")
        return
    client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # --- 5. 加载并筛选问题 ---
    try:
        with open(questions_file_path, 'r', encoding='utf-8') as f:
            all_questions = json.load(f)
    except FileNotFoundError:
        print(f"错误: 问题文件 '{questions_file_path}' 未找到。")
        return

    questions_to_run = [q for q in all_questions if q.get('category', '').startswith(category_prefix)]
    print(f"已从 '{questions_file_path}' 加载并筛选出 {len(questions_to_run)} 个相关问题。")

    # --- 6. 准备输出文件和断点续传 ---
    output_file = f"results_{args.task}.json"
    processed_results = []
    processed_ids = set()

    if os.path.exists(output_file):
        print(f"发现已存在的输出文件 '{output_file}'，将进行增量采集。")
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                processed_results = json.load(f)
                processed_ids = {item['id'] for item in processed_results}
                print(f"已加载 {len(processed_results)} 条已处理的结果。")
            except json.JSONDecodeError:
                print(f"警告: '{output_file}' 文件内容不是有效的JSON，将创建一个新文件。")
                processed_results = []
                processed_ids = set()

    # --- 7. 执行数据采集循环 ---
    for i, question_data in enumerate(questions_to_run):
        question_id = question_data.get('id')
        question_text = question_data.get('question')

        print(f"\n正在处理问题 {i + 1}/{len(questions_to_run)} (ID: {question_id}): {question_text}")

        if question_id in processed_ids:
            print("  - 此问题已处理，跳过。")
            continue

        response = get_ai_response(client, question_text, model_id)

        if response:
            result_entry = {
                "id": question_id,
                "category": question_data.get('category'),
                "question": question_text,
                "ai_model": model_id,
                "task": args.task,  # 记录任务名
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "response": response
            }
            processed_results.append(result_entry)

            # --- 8. 实时保存 ---
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_results, f, ensure_ascii=False, indent=2)
                print(f"  - 结果已实时保存到 '{output_file}'。")
            except Exception as e:
                print(f"  - 实时保存失败: {e}")
        else:
            print("  - 获取AI回答失败，跳过此问题。")

    print("\n--- 所有问题处理完毕 ---")
    print(f"采集任务 '{args.task}' 完成！最终结果已保存在 '{output_file}'。")


if __name__ == "__main__":
    # 在 re 模块导入时添加，以防万一
    import re

    main()
