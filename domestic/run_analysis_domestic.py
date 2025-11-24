import os
import json
import yaml
import argparse
import time
from openai import OpenAI, APIError
from dotenv import load_dotenv
from collections import defaultdict
from SparkApi import SparkSyncClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError

# --- 导入特殊模型 SDK ---
# 腾讯云 SDK
from tencentcloud.common import credential
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models as hunyuan_models

# 科大讯飞星火 SDK (请根据实际库名调整，这里假设是 spark_ai_python)
# 由于星火的调用通常涉及异步和签名，这里提供一个基于常见同步模式的框架
# 实际使用时，可能需要参考官方文档调整
# from spark_ai_python import SparkClient # 假设的导入，如果不对请修改

# 假设您的项目根目录是 GEO
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_questions(questions_path):
    """加载问题 JSON 文件"""
    with open(questions_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# --- 1. OpenAI 兼容模型调用 ---
@retry(
    # 仅在遇到 RateLimitError 时重试
    retry=retry_if_exception_type(RateLimitError),
    # 最多重试 5 次
    stop=stop_after_attempt(5),
    # 等待时间：2^x 秒，x 从 1 开始，即 2s, 4s, 8s, 16s, 32s
    wait=wait_exponential(multiplier=1, min=2, max=60),
    # 关键修改：移除 .sleep() 后的括号
    before_sleep=lambda retry_state: print(
        f"  -> Rate limit hit for {retry_state.args[0]['name']}. Retrying in {retry_state.next_action.sleep} seconds..."),
    reraise=True  # 重新抛出非重试异常
)
def call_openai_compatible_api(model_config, question):
    """
    统一调用 OpenAI 兼容接口的模型 API (包括 DashScope, Kimi, 豆包等)
    """
    api_key = os.getenv(model_config['api_key_env'])

    if not api_key:
        print(
            f"  -> Error: API Key for {model_config['name']} not found in environment variables ({model_config['api_key_env']}). Skipping.")
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=model_config['base_url']
    )

    model_name = model_config['model']

    messages = [
        {"role": "system", "content": "你是一个专业的市场分析师，请根据用户的问题提供详细、客观的分析和回答。"},
        {"role": "user", "content": question['prompt']}
    ]

    # --- 关键修改：判断是否需要强制流式调用 ---
    # 智谱模型需要强制流式调用
    is_streaming_required = (model_config['name'] == "智谱 GLM")

    try:
        print(f"  -> Calling {model_config['name']} ({model_name})...")

        response_stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            stream=is_streaming_required  # 智谱模型设置为 True
        )

        # --- 关键修改：处理流式和非流式结果 ---
        if is_streaming_required:
            # 聚合流式结果
            answer_parts = []
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    answer_parts.append(chunk.choices[0].delta.content)
            answer = "".join(answer_parts)
        else:
            # 非流式结果 (其他模型)
            answer = response_stream.choices[0].message.content

        result = {
            "category": question['category'],
            "question_id": question['id'],
            "model": model_config['name'],
            "response": {
                "answer": answer,
                "references": []
            }
        }
        return result


    except APIError as e:
        # 如果是速率限制错误，抛出 RateLimitError
        if e.status_code == 429:
            # 关键修改：从原始错误对象中获取 body，并正确构造 RateLimitError
            # e.response.text 包含原始的 JSON 响应体
            raise RateLimitError(
                message=f"Rate limit hit for {model_config['name']}",
                response=e.response,
                body=e.response.text  # 传递 body 参数
            )

        # 其他 API 错误，正常处理
        print(f"  -> API Error for {model_config['name']}: {e}")
        return None
        return None
    except Exception as e:
        print(f"  -> An unexpected error occurred for {model_config['name']}: {e}")
        return None


# --- 3. 科大讯飞星火模型调用 ---
def call_spark_api(model_config, question):
    """
    调用科大讯飞星火 API (使用封装后的同步客户端)
    """
    try:
        api_key = os.getenv(model_config['api_key_env'])
        api_secret = os.getenv(model_config['secret_key_env'])
        app_id = os.getenv(model_config['app_id_env'])
        # 注意这里，取决于你 config 里 base_url 是不是写在 env 里
        spark_url = os.getenv(model_config['base_url'])  # 如果写的是 ENV 名
        # 或者如果 config 里直接就是 URL，就用：
        # spark_url = model_config['base_url']

        if not api_key or not api_secret or not app_id or not spark_url:
            print(f"  -> Error: Key/Secret/AppID/URL for {model_config['name']} not found.")
            return None

        client = SparkSyncClient()
        domain = model_config['model']  # 比如 generalv3.5

        print(f"  -> Calling {model_config['name']} ({domain})...")

        answer = client.chat(
            appid=app_id,
            api_key=api_key,
            api_secret=api_secret,
            Spark_url=spark_url,
            domain=domain,
            question=question['prompt']
        )

        # ⭐ 关键：像其他模型一样构造统一的返回结构
        result = {
            "category": question['category'],
            "question_id": question['id'],
            "model": model_config['name'],
            "response": {
                "answer": answer,
                "references": []
            }
        }
        return result

    except Exception as e:
        print(f"  -> API Error for {model_config['name']}: {e}")
        return None


# --- 核心分派函数 ---
def call_model(model_key, model_config, question):
    """
    根据模型 key 分派到不同的 API 调用函数，并捕获重试后的最终异常
    """
    try:
        if model_key == 'spark':
            return call_spark_api(model_config, question)
        else:
            # 默认使用 OpenAI 兼容接口 (包括 hunyuan)
            return call_openai_compatible_api(model_config, question)
    except Exception as e:
        # 捕获所有异常，包括 tenacity 重试后的最终异常
        print(f"  -> FATAL ERROR for {model_config['name']} on QID {question['id']}: {e}")
        print("  -> Skipping this model/question pair.")
        return None  # 返回 None，让主循环继续


# ... (保持其他函数和导入不变)

def main():
    parser = argparse.ArgumentParser(description="Run domestic brand analysis data collection.")
    parser.add_argument('--config', type=str, default='config_domestic.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    config_path = os.path.join(BASE_DIR, args.config)

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    # --- 加载 .env 文件 ---
    load_dotenv(os.path.join(os.path.dirname(BASE_DIR), '.env'))
    # ---------------------------

    config = load_config(config_path)

    # 确保结果目录存在
    results_dir = os.path.join(BASE_DIR, config['paths']['results_dir'])
    os.makedirs(results_dir, exist_ok=True)

    # 加载问题
    questions_path = os.path.join(BASE_DIR, config['paths']['questions_file'])
    if not os.path.exists(questions_path):
        print(f"Error: Questions file not found at {questions_path}. Please create it.")
        return
    questions = load_questions(questions_path)

    all_results = []  # 存储所有模型和所有品类的结果

    for model_key, model_config in config['models'].items():
        print(f"--- Starting data collection for Model: {model_config['name']} ---")

        model_output_path = os.path.join(results_dir, f"results_{model_key}.json")
        model_results = []

        # --- 关键优化：检查并加载已存在的结果 ---
        existing_q_ids = set()
        if os.path.exists(model_output_path):
            try:
                with open(model_output_path, 'r', encoding='utf-8') as f:
                    model_results = json.load(f)
                existing_q_ids = {r['question_id'] for r in model_results}
                print(f"  -> Found {len(model_results)} existing results. Skipping Question IDs: {existing_q_ids}")
            except json.JSONDecodeError:
                print(f"  -> Warning: Could not read existing results file {model_output_path}. Overwriting.")
                model_results = []
        # ----------------------------------------

        newly_collected_results = []

        for question in questions:
            q_id = question['id']

            if q_id in existing_q_ids:
                print(f"  -> Question ID: {q_id} ({question['category']}) already exists. Skipping.")
                continue

            print(f"  -> Question ID: {q_id} ({question['category']})")

            result = call_model(model_key, model_config, question)

            if result:
                model_results.append(result)
                newly_collected_results.append(result)

            time.sleep(5)  # 避免API调用过于频繁

        # 如果有新采集的结果，则保存
        if newly_collected_results:
            # 保存单个模型的原始结果 (覆盖旧文件，包含新旧结果)
            with open(model_output_path, 'w', encoding='utf-8') as f:
                json.dump(model_results, f, ensure_ascii=False, indent=4)
            print(
                f"--- Saved {len(newly_collected_results)} new results for {model_config['name']} to {model_output_path} ---")
        else:
            print(f"--- No new results collected for {model_config['name']}. ---")

        all_results.extend(model_results)  # 将该模型的所有结果（新旧）加入总列表

    # --- 按品类分开保存合并结果 (保持不变) ---
    results_by_category = defaultdict(list)
    for result in all_results:
        category = result['category']
        results_by_category[category].append(result)

    print("\n*** Data collection finished. Saving merged results by category... ***")

    for category, results in results_by_category.items():
        if category == "新能源汽车":
            file_suffix = "nev"
        elif category == "5A级景区":
            file_suffix = "scenic"
        else:
            file_suffix = category.replace(" ", "_")

        merged_output_path = os.path.join(BASE_DIR, f"results_{file_suffix}_merged.json")

        with open(merged_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"-> Saved merged results for '{category}' to {merged_output_path}")

    print("\n*** All data processing complete. ***")


if __name__ == "__main__":
    main()
