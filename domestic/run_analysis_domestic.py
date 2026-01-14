import os
import json
import yaml
import argparse
from dotenv import load_dotenv
from collections import defaultdict
from SparkApi import SparkSyncClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError
import requests
import qianfan

# python run_analysis_domestic.py --task snack
# python run_analysis_domestic.py --task city

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
    except RateLimitError as e:
        # 速率限制错误，直接重新抛出让 tenacity 重试
        raise

    except APIError as e:
        # API 错误（有状态码）
        if hasattr(e, 'status_code') and e.status_code == 429:
            # 如果是429但没被识别为RateLimitError，手动抛出
            raise RateLimitError(
                message=f"Rate limit hit for {model_config['name']}",
                response=e.response,
                body=e.response.text if hasattr(e, 'response') else None
            )
        print(f"  -> API Error for {model_config['name']}: {e}")
        return None

    except Exception as e:
        # 其他所有异常（包括 APIConnectionError）
        print(f"  -> Connection/Unexpected error for {model_config['name']}: {e}")
        return None


from openai import OpenAI
import json


# --- 3. 科大讯飞星火模型调用 ---
def call_spark_api(model_config, question):
    """
    调用科大讯飞星火 API (使用封装后的同步客户端)
    """
    try:
        api_key = os.getenv(model_config['api_key_env'])
        api_secret = os.getenv(model_config['secret_key_env'])
        app_id = os.getenv(model_config['app_id_env'])

        # ⭐ 关键修改：直接使用配置中的 URL，而不是从环境变量读取
        spark_url = model_config['base_url']  # 直接使用配置中的 URL

        if not api_key or not api_secret or not app_id or not spark_url:
            print(f"  -> Error: Key/Secret/AppID/URL for {model_config['name']} not found.")
            return None

        client = SparkSyncClient()
        domain = model_config['model']  # 比如 spark-x1.5

        print(f"  -> Calling {model_config['name']} ({domain})...")

        answer = client.chat(
            appid=app_id,
            api_key=api_key,
            api_secret=api_secret,
            Spark_url=spark_url,
            domain=domain,
            question=question['prompt']
        )

        # 关键：像其他模型一样构造统一的返回结构
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


def call_spark_api(model_config, question):
    """
    调用科大讯飞星火 API (使用封装后的同步客户端)
    """
    try:
        api_key = os.getenv(model_config['api_key_env'])
        api_secret = os.getenv(model_config['secret_key_env'])
        app_id = os.getenv(model_config['app_id_env'])

        # ⭐ 关键修改：直接使用配置中的 URL，而不是从环境变量读取
        spark_url = model_config['base_url']  # 直接使用配置中的 URL

        if not api_key or not api_secret or not app_id or not spark_url:
            print(f"  -> Error: Key/Secret/AppID/URL for {model_config['name']} not found.")
            return None

        client = SparkSyncClient()
        domain = model_config['model']  # 比如 spark-x1.5

        print(f"  -> Calling {model_config['name']} ({domain})...")

        answer = client.chat(
            appid=app_id,
            api_key=api_key,
            api_secret=api_secret,
            Spark_url=spark_url,
            domain=domain,
            question=question['prompt']
        )

        # 关键：像其他模型一样构造统一的返回结构
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


def call_model(model_key, model_config, question):
    """
    根据模型 key 分派到不同的 API 调用函数
    """
    try:
        if model_key == 'spark':
            return call_spark_api(model_config, question)
        else:
            # 默认使用 OpenAI 兼容接口
            return call_openai_compatible_api(model_config, question)
    except Exception as e:
        print(f"  -> FATAL ERROR for {model_config['name']} on QID {question['id']}: {e}")
        return None


# ... (保持其他函数和导入不变)

def main():
    parser = argparse.ArgumentParser(description="Run domestic brand analysis data collection.")
    parser.add_argument('--config', type=str, default='config_domestic.yaml', help='Path to the configuration file.')
    # --- 新增 --task 参数 ---
    parser.add_argument('--task', type=str, help='Specify the task/category to run (e.g., snack, phone).')
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

    # --- 动态加载问题文件 ---
    if args.task:
        # 如果指定了 task，则加载 questions_{task}.json
        questions_file_name = f"question/questions_{args.task}.json"
        print(f"-> Running task: {args.task}. Loading questions from {questions_file_name}")
    else:
        # 否则加载默认的 questions_file
        questions_file_name = config['paths']['questions_file']
        print(f"-> Running default task. Loading questions from {questions_file_name}")

    questions_path = os.path.join(BASE_DIR, questions_file_name)

    if not os.path.exists(questions_path):
        print(f"Error: Questions file not found at {questions_path}. Please create it.")
        return
    questions = load_questions(questions_path)

    all_results = []  # 存储所有模型和所有品类的结果

    for model_key, model_config in config['models'].items():
        print(f"--- Starting data collection for Model: {model_config['name']} ---")

        model_output_path = os.path.join(results_dir, f"results_{model_key}.json")
        model_results = []

        # --- 关键修改：检查并加载已存在的结果 ---
        # 仍然加载旧结果，但不会用于跳过
        if os.path.exists(model_output_path):
            try:
                with open(model_output_path, 'r', encoding='utf-8') as f:
                    model_results = json.load(f)
                print(f"  -> Found {len(model_results)} existing results. Will re-run all questions and append.")
            except json.JSONDecodeError:
                print(f"  -> Warning: Could not read existing results file {model_output_path}. Starting fresh.")
                model_results = []
        # ----------------------------------------

        newly_collected_results = []

        for question in questions:
            q_id = question['id']

            # 移除跳过逻辑，强制运行
            print(f"  -> Question ID: {q_id} ({question['category']})")

            result = call_model(model_key, model_config, question)

            if result:
                model_results.append(result)
                newly_collected_results.append(result)

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
        elif category == "智能手机":
            file_suffix = "phone"
        elif category == "餐饮美食":
            file_suffix = "food"
        elif category == "奢侈品":
            file_suffix = "luxury"
        elif category == "美妆护肤":
            file_suffix = "beauty"
        elif category == "零食饮料":
            file_suffix = "snack"
        elif category == "中国旅游城市":
            file_suffix = "city"
        else:
            file_suffix = "other"

        merged_output_path = os.path.join(BASE_DIR, f"merged_results/results_{file_suffix}_merged.json")

        with open(merged_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"-> Saved merged results for '{category}' to {merged_output_path}")

    print("\n*** All data processing complete. ***")


if __name__ == "__main__":
    main()
