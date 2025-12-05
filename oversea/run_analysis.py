import openai
import json
import os
import time
import argparse
import yaml  # 引入PyYAM
from google import genai  # 导入 Google 官方 SDK
from google.genai.errors import APIError as GeminiAPIError  # 导入 Gemini 的 API 错误
import re


# ==============================================================================
# 终极数据采集引擎 (v1.0 - 任务驱动版)
# 描述: 一个由中央配置文件驱动的、按任务隔离的、支持断点续传的数据采集脚本。
# ==============================================================================
# 示例用法:
# python run_analysis.py --task ha_perplexity
# python run_analysis.py --task sh_perplexity
# python run_analysis.py --task ts_gemini
# python run_analysis.py --task tc_gpt
# export OPENROUTER_API_KEY="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"
# export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
# export GEMINI_API_KEY="AIzaSyA9GNai9wxeAX89Qr9w0qFthseEVV8ROwY"
# ==============================================================================


# ------------ 扩展版 Gemini 引用解析函数 ---------------- #
def get_gemini_response(client, question: str, model: str, retries=3, delay=5):
    URL_PATTERN = r'https?://[^\s)>\]]+'

    for attempt in range(retries):
        try:
            print(f"    - 正在调用 Gemini 模型 '{model}' (尝试 {attempt + 1}/{retries})...")

            response = client.models.generate_content(
                model=model,
                contents=question,
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,
                    tools=[
                        genai.types.Tool(
                            google_search=genai.types.GoogleSearch()
                        )
                    ]
                )
            )

            # ---- 文本处理 ----
            try:
                answer = response.text or ""
            except:
                answer = str(response)

            references = []

            # ---- grounding metadata ----
            try:
                c0 = response.candidates[0]
                meta = getattr(c0, "grounding_metadata", None)
                if meta and hasattr(meta, "grounding_chunks"):
                    for chunk in meta.grounding_chunks:
                        web = chunk.web
                        references.append({
                            "url": getattr(web, "uri", ""),
                            "title": getattr(web, "title", "") or "",
                            "publisher": getattr(web, "site", "") or "",
                            "snippet": getattr(web, "snippet", "") or "",
                        })
            except:
                pass

            # ---- URL fallback ----
            if not references and answer:
                raw_urls = re.findall(URL_PATTERN, answer)
                for url in raw_urls:
                    references.append({
                        "url": url.strip().rstrip(".,;:"),
                        "title": "",
                        "publisher": "",
                        "snippet": "",
                    })

            return {"answer": answer, "references": references}

        except Exception as e:
            print(f"    - Gemini 调用错误: {e}")
            if attempt < retries - 1:
                print(f"    - 等待 {delay} 秒后重试...")
                time.sleep(delay)
            return {"answer": "", "references": []}


# ------------ GPT / Perplexity 通用函数（ OpenRouter API ） ---------------- #
def get_ai_response(client: openai.OpenAI, question: str, model: str, retries=3, delay=5):
    """调用 GPT / Perplexity（OpenAI 兼容 API）"""
    URL_PATTERN = r'https?://[^\s)>\]]+'

    for attempt in range(retries):
        try:
            print(f"    - 正在调用模型 '{model}' (尝试 {attempt + 1}/{retries})...")
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=0.1,
            )

            response_content = completion.choices[0].message.content

            # 正则 URL 抓取
            urls = re.findall(URL_PATTERN, response_content)
            references = [{"url": u.rstrip('.,;:'), "title": ""} for u in urls]

            return {"answer": response_content, "references": references}

        except Exception as e:
            print(f"    - API错误: {e}")
            if attempt < retries - 1:
                print(f"    - 等待 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                print("    - 所有重试均失败。")
                return {"answer": "", "references": []}


# ------------ 主执行函数 ---------------- #
def main():
    parser = argparse.ArgumentParser(description="多模型数据采集引擎。")
    parser.add_argument("--task", required=True, help="任务名称")
    parser.add_argument("--config", default="config.yaml", help="中央配置文件")
    args = parser.parse_args()

    # 读取中央配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    task_cfg = config.get('tasks', {}).get(args.task)
    if not task_cfg:
        print(f"错误: 未找到任务 '{args.task}' 的配置。")
        return

    model_key = task_cfg["model_key"]
    category_prefix = task_cfg["category_prefix"]
    model_name = config["models"][model_key]
    questions_file = config["global_settings"]["questions_file"]

    print(f"--- 开始执行采集任务: {args.task} ---")
    print(f"  - 模型: {model_name}")
    print(f"  - 分类前缀: {category_prefix}")

    # 判断是否 Gemini 任务 & 初始化 API
    is_gemini = "gemini" in args.task.lower()
    if is_gemini:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            print("错误: 请先设置 GEMINI_API_KEY")
            return
        ai_client = genai.Client(api_key=gemini_api_key)
        call_func = get_gemini_response
    else:

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("错误: 请先设置 OPENROUTER_API_KEY")
            return
        client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # 加载并筛选问题
    with open(questions_file, 'r', encoding='utf-8') as f:
        all_questions = json.load(f)

    questions_to_run = [q for q in all_questions if q.get('category', '').startswith(category_prefix)]
    print(f"  → 匹配到 {len(questions_to_run)} 个问题")

    output_file = f"results_{args.task}.json"
    processed_results = []
    processed_ids = set()

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                processed_results = json.load(f)
                processed_ids = {item["id"] for item in processed_results}
                print(f"  → 已加载 {len(processed_results)} 条历史记录（增量模式）")
            except:
                print("  → 输出文件损坏，将重新生成")

    # 采集循环
    for idx, q in enumerate(questions_to_run):
        qid = q["id"]
        qtext = q["question"]
        print(f"\n处理 {idx + 1}/{len(questions_to_run)} (ID {qid}): {qtext}")

        if qid in processed_ids:
            print("  - 已存在，跳过")
            continue

        if is_gemini:
            result = get_gemini_response(ai_client, qtext, model_name)
        else:
            result = get_ai_response(client, qtext, model_name)

        processed_results.append({
            "id": qid,
            "category": q["category"],
            "question": qtext,
            "ai_model": model_name,
            "task": args.task,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "response": result,
        })

        # 实时保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_results, f, ensure_ascii=False, indent=2)
        print("  - 已保存")

    print("\n--- 全部问题处理完成 ---")
    print(f"输出文件: {output_file}")

    # ====== 汇总所有引用 ======
    all_refs = []
    seen = set()

    for item in processed_results:
        refs = item.get("response", {}).get("references", [])
        for r in refs:
            url = r.get("url", "")
            if url and url not in seen:
                seen.add(url)
                all_refs.append(r)

    with open(f"references_{args.task}.json", "w", encoding="utf-8") as f:
        json.dump(all_refs, f, ensure_ascii=False, indent=2)

    print(f"  → 已汇总引用: {len(all_refs)} 条 (保存为 references_{args.task}.json)")


if __name__ == "__main__":
    main()
