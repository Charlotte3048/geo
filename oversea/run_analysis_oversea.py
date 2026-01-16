import openai
import json
import os
import time
import argparse
import yaml
import re
from dotenv import load_dotenv

# ==============================================================================
# 海外数据采集引擎 (v2.0 - 品类驱动版)
# 描述: 按品类运行，支持指定单个模型或运行所有模型，直接输出合并结果文件（带日期戳）
# ==============================================================================
# 示例用法:
# python run_analysis_oversea.py --task ha                    # 运行家用电器品类的所有模型
# python run_analysis_oversea.py --task ha --model gemini     # 只运行家用电器品类的 gemini 模型
# python run_analysis_oversea.py --task sh --model perplexity # 只运行智能硬件品类的 perplexity 模型
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 加载 .env 文件（从根目录加载）
root_dir = os.path.dirname(BASE_DIR)  # 获取父目录（根目录）
load_dotenv(os.path.join(root_dir, '.env'))



def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_questions(questions_path):
    """加载问题 JSON 文件"""
    with open(questions_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ------------ GPT / Gemini 联网搜索调用（通过 OpenRouter :online 后缀） ---------------- #
def get_online_response(client, question: str, model: str, model_key: str, retries=3, delay=10):
    """
    通过 OpenRouter 调用 GPT / Gemini 模型（使用 :online 后缀启用联网搜索）
    """
    URL_PATTERN = r'https?://[^\s )>\]]+'

    # 添加 :online 后缀以启用联网搜索
    if not model.endswith(':online'):
        model = f"{model}:online"

    for attempt in range(retries):
        try:
            print(f"      正在调用 {model_key.upper()} 模型 '{model}' (尝试 {attempt + 1}/{retries})...")

            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=0.1,
            )

            answer = completion.choices[0].message.content or ""

            # 提取 URL 作为引用
            references = []
            urls = re.findall(URL_PATTERN, answer)
            seen_urls = set()

            for url in urls:
                clean_url = url.strip().rstrip(".,;:)]")
                if clean_url and clean_url not in seen_urls:
                    seen_urls.add(clean_url)
                    references.append({
                        "url": clean_url,
                        "title": "",
                        "publisher": "",
                        "snippet": "",
                    })

            return {"answer": answer, "references": references}

        except Exception as e:
            error_msg = str(e)
            print(f"      {model_key.upper()} 调用错误: {error_msg}")

            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                wait_time = delay * (attempt + 1)
                print(f"      配额限制，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            elif attempt < retries - 1:
                print(f"      等待 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                print("      所有重试均失败。")
                return {"answer": "", "references": []}

    return {"answer": "", "references": []}


# ------------ Perplexity 原生联网搜索调用（不需要 :online 后缀） ---------------- #
def get_perplexity_response(client, question: str, model: str, retries=3, delay=5):
    """
    调用 Perplexity 模型（原生支持联网搜索，不需要 :online 后缀）
    """
    URL_PATTERN = r'https?://[^\s)>\]]+'

    for attempt in range(retries):
        try:
            print(f"      正在调用 PERPLEXITY 模型 '{model}' (尝试 {attempt + 1}/{retries})...")
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=0.1,
            )

            response_content = completion.choices[0].message.content or ""

            # 正则 URL 抓取
            urls = re.findall(URL_PATTERN, response_content)
            seen_urls = set()
            references = []

            for url in urls:
                clean_url = url.strip().rstrip(".,;:)]")
                if clean_url and clean_url not in seen_urls:
                    seen_urls.add(clean_url)
                    references.append({
                        "url": clean_url,
                        "title": "",
                        "publisher": "",
                        "snippet": "",
                    })

            return {"answer": response_content, "references": references}

        except Exception as e:
            error_msg = str(e)
            print(f"      PERPLEXITY 调用错误: {error_msg}")

            if "429" in error_msg:
                wait_time = delay * (attempt + 1)
                print(f"      速率限制，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            elif attempt < retries - 1:
                print(f"      等待 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                print("      所有重试均失败。")
                return {"answer": "", "references": []}

    return {"answer": "", "references": []}


def call_model(client, model_key: str, model_name: str, question: str):
    """
    根据模型 key 分派到不同的 API 调用函数
    - GPT / Gemini: 使用 :online 后缀启用联网搜索
    - Perplexity: 原生联网搜索，不需要后缀
    """
    try:
        if "perplexity" in model_key.lower():
            # Perplexity 原生联网搜索
            return get_perplexity_response(client, question, model_name)
        else:
            # GPT / Gemini 使用 :online 后缀
            return get_online_response(client, question, model_name, model_key)
    except Exception as e:
        print(f"      FATAL ERROR for {model_key}: {e}")
        return {"answer": "", "references": []}


def main():
    parser = argparse.ArgumentParser(description="海外数据采集引擎")
    parser.add_argument("--task", required=True, help="任务/品类名称 (如: ha, sh)")
    parser.add_argument("--model", default=None, help="指定单个模型 (如: gemini, gpt, perplexity)，不指定则运行所有模型")
    parser.add_argument("--config", default="config_oversea.yaml", help="配置文件路径")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"海外数据采集引擎")
    print(f"{'=' * 60}\n")

    # 加载 .env 文件
    load_dotenv(os.path.join(BASE_DIR, '.env'))

    # 读取配置文件
    config_path = os.path.join(BASE_DIR, args.config)
    if not os.path.exists(config_path):
        print(f"❌ 错误: 配置文件 '{config_path}' 未找到。")
        return

    config = load_config(config_path)

    # 获取任务配置
    task_cfg = config.get('tasks', {}).get(args.task)
    if not task_cfg:
        print(f"❌ 错误: 未找到任务 '{args.task}' 的配置。")
        print(f"   可用任务: {', '.join(config.get('tasks', {}).keys())}")
        return

    task_name = task_cfg.get("name", args.task)
    category_prefix = task_cfg.get("category_prefix", "")
    questions_file = task_cfg.get("questions_file", f"question/questions_{args.task}.json")

    # 获取模型配置
    all_models = config.get("models", {})
    if args.model:
        # 指定了单个模型
        if args.model not in all_models:
            print(f"❌ 错误: 未找到模型 '{args.model}' 的配置。")
            print(f"   可用模型: {', '.join(all_models.keys())}")
            return
        models_to_run = {args.model: all_models[args.model]}
    else:
        # 运行所有模型
        models_to_run = all_models

    print(f"📋 任务: {task_name} ({args.task})")
    print(f"📂 品类前缀: {category_prefix}")
    print(f"📄 问题文件: {questions_file}")
    print(f"🤖 模型: {', '.join(models_to_run.keys())}")
    print(f"   - GPT/Gemini: 使用 :online 后缀启用联网搜索")
    print(f"   - Perplexity: 原生联网搜索\n")

    # 初始化 OpenRouter 客户端
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ 错误: 请先设置 OPENROUTER_API_KEY 环境变量")
        return

    client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # 加载问题文件
    questions_path = os.path.join(BASE_DIR, questions_file)
    if not os.path.exists(questions_path):
        print(f"❌ 错误: 问题文件 '{questions_path}' 未找到。")
        return

    all_questions = load_questions(questions_path)

    # 按品类前缀筛选问题
    if category_prefix:
        questions_to_run = [q for q in all_questions if q.get('category', '').startswith(category_prefix)]
    else:
        questions_to_run = all_questions

    print(f"📝 匹配到 {len(questions_to_run)} 个问题\n")

    if not questions_to_run:
        print("⚠️ 没有匹配的问题，退出。")
        return

    # 准备输出文件（带日期戳，保存到 results 目录）
    current_date = time.strftime("%Y%m%d")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, f"results_merged_{args.task}_{current_date}.json")

    # 加载已有结果（支持断点续传）
    all_results = []
    processed_keys = set()  # 用于跟踪已处理的 (question_id, model) 组合

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
                for item in all_results:
                    key = (item.get("id"), item.get("ai_model"))
                    processed_keys.add(key)
                print(f"📂 已加载 {len(all_results)} 条历史记录（断点续传模式）\n")
        except json.JSONDecodeError:
            print("⚠️ 输出文件损坏，将重新生成\n")
            all_results = []

    # 开始采集
    total_questions = len(questions_to_run)
    total_models = len(models_to_run)
    total_tasks = total_questions * total_models
    completed = 0

    for model_key, model_name in models_to_run.items():
        print(f"\n{'─' * 50}")
        print(f"🤖 开始采集模型: {model_key} ({model_name})")
        if "perplexity" in model_key.lower():
            print(f"   📡 模式: 原生联网搜索")
        else:
            print(f"   📡 模式: :online 后缀联网搜索")
        print(f"{'─' * 50}")

        for idx, question in enumerate(questions_to_run):
            q_id = question.get("id")
            q_text = question.get("question", question.get("prompt", ""))
            q_category = question.get("category", "")

            completed += 1
            progress = f"[{completed}/{total_tasks}]"

            # 检查是否已处理
            if (q_id, model_name) in processed_keys:
                print(f"  {progress} Q{q_id}: 已存在，跳过")
                continue

            print(f"  {progress} Q{q_id} ({q_category}): {q_text[:50]}...")

            # 调用模型
            response = call_model(client, model_key, model_name, q_text)

            # 构造结果
            result = {
                "id": q_id,
                "category": q_category,
                "question": q_text,
                "ai_model": model_name,
                "model_key": model_key,
                "task": args.task,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "response": response
            }

            all_results.append(result)
            processed_keys.add((q_id, model_name))

            # 实时保存
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            print(f"      ✅ 已保存")

    # 汇总引用
    print(f"\n{'=' * 60}")
    print("📊 采集完成，汇总引用...")

    all_refs = []
    seen_urls = set()

    for item in all_results:
        refs = item.get("response", {}).get("references", [])
        for r in refs:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_refs.append(r)

    refs_dir = os.path.join(BASE_DIR, "references")
    os.makedirs(refs_dir, exist_ok=True)
    refs_file = os.path.join(refs_dir, f"references_{args.task}_{current_date}.json")
    with open(refs_file, "w", encoding="utf-8") as f:
        json.dump(all_refs, f, ensure_ascii=False, indent=2)

    print(f"\n📈 统计信息:")
    print(f"   - 总结果数: {len(all_results)}")
    print(f"   - 总引用数: {len(all_refs)}")
    print(f"   - 结果文件: {output_file}")
    print(f"   - 引用文件: {refs_file}")

    print(f"\n{'=' * 60}")
    print("✅ 全部采集任务完成！")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
