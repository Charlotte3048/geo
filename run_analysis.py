import openai
import json
import os
import time
from typing import List, Dict, Any

# --- 1. 配置 (Configuration) ---
# 确保在运行脚本前设置了以下环境变量:
# export OPENAI_API_KEY="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"
# export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# 要调用的具备联网功能的模型 (在OpenRouter上查找 )
# 例如: 'google/gemini-pro' 是一个不错的选择，因为它默认就有很好的联网能力。
# 你也可以选择其他支持工具调用或联网功能的模型。
# MODEL_NAME = "google/gemini-2.5-flash"
MODEL_NAME = "openai/chatgpt-4o-latest"

# 输入的问题文件
QUESTIONS_FILE = "questions.json"

# 输出的结果文件
RESULTS_FILE = "results.json"

# 两次API调用之间的延迟（秒），以避免触发速率限制
REQUEST_DELAY = 2

# API请求的重试次数
MAX_RETRIES = 3
RETRY_DELAY = 5  # 重试前的等待时间（秒）


# --- 2. 准备问题列表 ---
# 我们将问题存储在一个单独的JSON文件中，方便管理。
# questions.json 的格式应该如下:
# [
#   {
#     "id": 1,
#     "category": "家用电器-扫地机器人",
#     "question": "请推荐几个在海外市场最受欢迎的中国扫地机器人品牌及其代表型号。"
#   },
#   {
#     "id": 2,
#     "category": "家用电器-扫地机器人",
#     "question": "石头科技(Roborock)和科沃斯(Ecovacs)的旗舰扫地机器人有什么区别？哪个更值得购买？"
#   }
# ]

def prepare_questions_file():
    """如果问题文件不存在，则创建一个示例文件。"""
    if not os.path.exists(QUESTIONS_FILE):
        print(f"未找到问题文件 '{QUESTIONS_FILE}'。正在创建一个示例文件...")
        sample_questions = [
            {
                "id": 1,
                "category": "家用电器-扫地机器人",
                "question": "请推荐几个在海外市场最受欢迎的中国扫地机器人品牌及其代表型号。"
            },
            {
                "id": 2,
                "category": "家用电器-扫地机器人",
                "question": "石头科技(Roborock)和科沃斯(Ecovacs)的旗舰扫地机器人有什么区别？哪个更值得购买？"
            },
            {
                "id": 3,
                "category": "家用电器-空气炸锅",
                "question": "对于一个在欧洲生活的用户，你会推荐哪个品牌的中国空气炸锅？请说明理由。"
            }
        ]
        with open(QUESTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(sample_questions, f, ensure_ascii=False, indent=2)
        print("示例文件创建成功。您可以根据需要修改此文件。")


# --- 3. 核心API调用函数 ---
def get_ai_response_with_references(client: openai.OpenAI, question: str) -> Dict[str, Any]:
    """
    调用AI模型获取答案，并尝试提取引用来源。
    注意：OpenRouter/OpenAI的联网功能返回格式可能不同。
    对于支持工具调用的模型，它会先返回工具调用请求，然后我们提供结果。
    对于像Gemini Pro这样内置搜索的模型，引用可能直接在内容中或元数据里。
    此处的实现是一个通用模板，需要根据你选择的模型的具体行为进行微调。
    """
    print(f"\n正在处理问题: {question}")

    for attempt in range(MAX_RETRIES):
        try:
            # OpenRouter通过在HTTP头中添加 "X-Title" 来识别你的项目
            # OpenAI库v1.x.x版本后，可以通过 extra_headers 参数来设置
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system",
                     "content": "你是一个严谨的市场分析师。你的回答需要基于最新的网络信息，并尽可能提供你的信息来源网址。请用中文回答。"},
                    {"role": "user", "content": question}
                ],
                # extra_headers={
                #     "HTTP-Referer": "https://github.com/your-repo",  # 推荐填写 ，便于OpenRouter识别项目来源
                #     "X-Title": "GenAI Brand Ranking Project"  # 推荐填写
                # }
            )

            # 提取内容和可能的引用
            # 不同的联网模型返回引用的方式不同。
            # 1. 有些模型会在文本末尾直接列出 [1]... [2]... 并附上链接。
            # 2. 有些模型的API响应体中会包含结构化的 'citations' 或 'sources' 字段。
            # 我们需要解析返回的 content 文本来查找URL。
            content = response.choices[0].message.content

            # 一个简单的URL提取逻辑 (可以使用更复杂的正则表达式)
            import re
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\ ),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                              content)

            print(f"成功获取回答。")
            return {
                "answer": content,
                "references": list(set(urls))  # 去重
            }

        except openai.APIError as e:
            print(f"发生API错误 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
        except openai.RateLimitError as e:
            print(f"触发速率限制 (尝试 {attempt + 1}/{MAX_RETRIES})。将在 {RETRY_DELAY} 秒后重试...")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"发生未知错误 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)

    print("所有重试均失败。跳过此问题。")
    return {
        "answer": "ERROR: Failed to get response after multiple retries.",
        "references": []
    }


# --- 4. 主执行逻辑 ---
def main():
    """主函数，用于执行整个流程。"""
    print("--- 开始执行GenAI品牌心智占有率数据收集脚本 ---")

    # 检查API密钥和URL是否设置
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        print("错误：请先设置环境变量 'OPENAI_API_KEY' 和 'OPENAI_BASE_URL'。")
        print("例如: export OPENAI_API_KEY='sk-or-...'")
        print("      export OPENAI_BASE_URL='https://openrouter.ai/api/v1'")
        return

    # 初始化OpenAI客户端
    client = openai.OpenAI()

    # 准备问题文件
    prepare_questions_file()

    # 读取问题
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # 检查是否有已存在的结果，以便断点续传
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    else:
        results = []

    processed_ids = {item['id'] for item in results}
    print(f"已找到 {len(results)} 条旧结果。将跳过已处理的问题。")

    # 循环处理问题
    for item in questions:
        if item['id'] in processed_ids:
            print(f"问题ID {item['id']} 已处理，跳过。")
            continue

        response_data = get_ai_response_with_references(client, item['question'])

        # 组合结果
        result_item = {
            "id": item['id'],
            "category": item['category'],
            "question": item['question'],
            "ai_model": MODEL_NAME,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "response": response_data
        }
        results.append(result_item)

        # 实时保存结果到文件，防止中途失败丢失数据
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"结果已保存到 '{RESULTS_FILE}'。")

        # 等待一段时间，避免请求过于频繁
        time.sleep(REQUEST_DELAY)

    print("\n--- 所有问题处理完毕 ---")
    print(f"最终结果已全部保存在 '{RESULTS_FILE}' 文件中。")


if __name__ == "__main__":
    main()
