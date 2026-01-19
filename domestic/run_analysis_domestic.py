import os
import json
import yaml
import argparse
from dotenv import load_dotenv
from collections import defaultdict
from SparkApi import SparkSyncClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, RateLimitError, APIError
import dashscope
from dashscope import Generation
from http import HTTPStatus

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


# ============================================================
# 1. DashScope 原生模式调用（阿里云百炼，支持联网搜索）
# ============================================================
def call_dashscope_api(model_config, question):
    """
    使用 DashScope 原生 SDK 调用模型（支持联网搜索）
    适用于：通义千问、DeepSeek、Kimi、智谱GLM（通过阿里云百炼）
    """
    api_key = os.getenv(model_config['api_key_env'])

    if not api_key:
        print(
            f"  -> Error: API Key for {model_config['name']} not found in environment variables ({model_config['api_key_env']}). Skipping.")
        return None

    dashscope.api_key = api_key

    model_name = model_config['model']
    enable_search = model_config.get('enable_search', True)

    messages = [
        {"role": "system", "content": "你是一个专业的市场分析师，请根据用户的问题提供详细、客观的分析和回答。"},
        {"role": "user", "content": question['prompt']}
    ]

    try:
        search_status = "开启" if enable_search else "关闭"
        print(f"  -> Calling {model_config['name']} ({model_name}) [联网搜索: {search_status}]...")

        call_params = {
            "model": model_name,
            "messages": messages,
            "result_format": "message",
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        if enable_search:
            call_params["enable_search"] = True
            call_params["search_options"] = {
                "search_strategy": "standard",
                "enable_source": True
            }

        response = Generation.call(**call_params)

        if response.status_code == HTTPStatus.OK:
            answer = response.output.choices[0].message.content

            references = []
            if enable_search and hasattr(response.output, 'search_info') and response.output.search_info:
                search_results = response.output.search_info.get("search_results", [])
                for ref in search_results:
                    references.append({
                        "index": ref.get('index', ''),
                        "title": ref.get('title', ''),
                        "url": ref.get('url', '')
                    })
                if references:
                    print(f"     -> 获取到 {len(references)} 条搜索引用")

            result = {
                "category": question['category'],
                "question_id": question['id'],
                "model": model_config['name'],
                "response": {
                    "answer": answer,
                    "references": references
                }
            }
            return result
        else:
            print(f"  -> API Error for {model_config['name']}: {response.code} - {response.message}")
            return None

    except Exception as e:
        print(f"  -> Error for {model_config['name']}: {e}")
        return None


# ============================================================
# 2. 豆包（火山引擎）联网搜索调用
# ============================================================
def call_doubao_api(model_config, question):
    """
    调用豆包 API（火山引擎），支持联网搜索
    使用 responses.create() 端点
    """
    api_key = os.getenv(model_config['api_key_env'])

    if not api_key:
        print(f"  -> Error: API Key for {model_config['name']} not found. Skipping.")
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=model_config['base_url']
    )

    model_name = model_config['model']
    enable_search = model_config.get('enable_search', True)

    try:
        search_status = "开启" if enable_search else "关闭"
        print(f"  -> Calling {model_config['name']} ({model_name}) [联网搜索: {search_status}]...")

        messages = [
            {"role": "system", "content": "你是一个专业的市场分析师，请根据用户的问题提供详细、客观的分析和回答。"},
            {"role": "user", "content": question['prompt']}
        ]

        if enable_search:
            # 使用 responses.create() 端点进行联网搜索
            tools = [{"type": "web_search"}]

            response = client.responses.create(
                model=model_name,
                input=messages,
                tools=tools,
            )

            # 解析响应
            answer = ""
            references = []

            # responses.create() 返回的结构不同，需要遍历 output
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'type'):
                        if item.type == 'message' and hasattr(item, 'content'):
                            # 提取文本内容
                            for content_item in item.content:
                                if hasattr(content_item, 'type') and content_item.type == 'text':
                                    answer += content_item.text
                        elif item.type == 'web_search_call' and hasattr(item, 'search_results'):
                            # 提取搜索结果
                            for ref in item.search_results:
                                references.append({
                                    "title": getattr(ref, 'title', ''),
                                    "url": getattr(ref, 'url', ''),
                                    "snippet": getattr(ref, 'snippet', '')
                                })

            if references:
                print(f"     -> 获取到 {len(references)} 条搜索引用")
        else:
            # 不使用联网搜索，使用标准 chat.completions
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
            answer = response.choices[0].message.content
            references = []

        result = {
            "category": question['category'],
            "question_id": question['id'],
            "model": model_config['name'],
            "response": {
                "answer": answer,
                "references": references
            }
        }
        return result

    except Exception as e:
        print(f"  -> Error for {model_config['name']}: {e}")
        return None


# ============================================================
# 3. 360智脑 联网搜索调用
# ============================================================
def call_zhinao_api(model_config, question):
    """
    调用360智脑 API，支持联网搜索
    """
    api_key = os.getenv(model_config['api_key_env'])

    if not api_key:
        print(f"  -> Error: API Key for {model_config['name']} not found. Skipping.")
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=model_config['base_url']
    )

    model_name = model_config['model']
    enable_search = model_config.get('enable_search', True)

    try:
        search_status = "开启" if enable_search else "关闭"
        print(f"  -> Calling {model_config['name']} ({model_name}) [联网搜索: {search_status}]...")

        messages = [
            {"role": "system", "content": "你是一个专业的市场分析师，请根据用户的问题提供详细、客观的分析和回答。"},
            {"role": "user", "content": question['prompt']}
        ]

        # 360智脑通过 extra_body 参数启用联网搜索
        extra_params = {}
        if enable_search:
            extra_params["extra_body"] = {
                "enable_web_search": True
            }

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            **extra_params
        )

        answer = response.choices[0].message.content
        references = []

        # 尝试提取搜索引用（如果有）
        if hasattr(response, 'web_search') and response.web_search:
            for ref in response.web_search:
                references.append({
                    "title": ref.get('title', ''),
                    "url": ref.get('url', ''),
                    "snippet": ref.get('snippet', '')
                })
            if references:
                print(f"     -> 获取到 {len(references)} 条搜索引用")

        result = {
            "category": question['category'],
            "question_id": question['id'],
            "model": model_config['name'],
            "response": {
                "answer": answer,
                "references": references
            }
        }
        return result

    except Exception as e:
        print(f"  -> Error for {model_config['name']}: {e}")
        return None


# ============================================================
# 4. 腾讯混元 联网搜索调用
# ============================================================
def call_hunyuan_api(model_config, question):
    """
    调用腾讯混元 API，支持联网搜索
    """
    api_key = os.getenv(model_config['api_key_env'])

    if not api_key:
        print(f"  -> Error: API Key for {model_config['name']} not found. Skipping.")
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=model_config['base_url']
    )

    model_name = model_config['model']
    enable_search = model_config.get('enable_search', True)

    try:
        search_status = "开启" if enable_search else "关闭"
        print(f"  -> Calling {model_config['name']} ({model_name}) [联网搜索: {search_status}]...")

        messages = [
            {"role": "system", "content": "你是一个专业的市场分析师，请根据用户的问题提供详细、客观的分析和回答。"},
            {"role": "user", "content": question['prompt']}
        ]

        # 腾讯混元通过 extra_body 参数启用联网搜索
        extra_params = {}
        if enable_search:
            extra_params["extra_body"] = {
                "enable_enhancement": True
            }

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            **extra_params
        )

        answer = response.choices[0].message.content
        references = []

        # 尝试提取搜索引用（如果有）
        if hasattr(response, 'search_info') and response.search_info:
            for ref in response.search_info.get('search_results', []):
                references.append({
                    "title": ref.get('title', ''),
                    "url": ref.get('url', ''),
                    "snippet": ref.get('snippet', '')
                })
            if references:
                print(f"     -> 获取到 {len(references)} 条搜索引用")

        result = {
            "category": question['category'],
            "question_id": question['id'],
            "model": model_config['name'],
            "response": {
                "answer": answer,
                "references": references
            }
        }
        return result

    except Exception as e:
        print(f"  -> Error for {model_config['name']}: {e}")
        return None


# ============================================================
# 5. 通用 OpenAI 兼容接口调用（无联网搜索）
# ============================================================
@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=lambda retry_state: print(
        f"  -> Rate limit hit. Retrying in {retry_state.next_action.sleep} seconds..."),
    reraise=True
)
def call_openai_compatible_api(model_config, question):
    """
    通用 OpenAI 兼容接口调用（不支持联网搜索的模型）
    """
    api_key = os.getenv(model_config['api_key_env'])

    if not api_key:
        print(f"  -> Error: API Key for {model_config['name']} not found. Skipping.")
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

    try:
        print(f"  -> Calling {model_config['name']} ({model_name})...")

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )

        answer = response.choices[0].message.content

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
        raise

    except APIError as e:
        if hasattr(e, 'status_code') and e.status_code == 429:
            raise RateLimitError(
                message=f"Rate limit hit for {model_config['name']}",
                response=e.response,
                body=e.response.text if hasattr(e, 'response') else None
            )
        print(f"  -> API Error for {model_config['name']}: {e}")
        return None

    except Exception as e:
        print(f"  -> Error for {model_config['name']}: {e}")
        return None


# ============================================================
# 6. 科大讯飞星火模型调用
# ============================================================
def call_spark_api(model_config, question):
    """
    调用科大讯飞星火 API (使用封装后的同步客户端)
    """
    try:
        api_key = os.getenv(model_config['api_key_env'])
        api_secret = os.getenv(model_config['secret_key_env'])
        app_id = os.getenv(model_config['app_id_env'])
        spark_url = model_config['base_url']

        if not api_key or not api_secret or not app_id or not spark_url:
            print(f"  -> Error: Key/Secret/AppID/URL for {model_config['name']} not found.")
            return None

        client = SparkSyncClient()
        domain = model_config['model']

        print(f"  -> Calling {model_config['name']} ({domain})...")

        answer = client.chat(
            appid=app_id,
            api_key=api_key,
            api_secret=api_secret,
            Spark_url=spark_url,
            domain=domain,
            question=question['prompt']
        )

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


# ============================================================
# 7. 模型调用分派器
# ============================================================
def call_model(model_key, model_config, question):
    """
    根据模型配置分派到不同的 API 调用函数
    """
    try:
        api_type = model_config.get('api_type', 'openai')

        if api_type == 'dashscope':
            return call_dashscope_api(model_config, question)
        elif api_type == 'doubao':
            return call_doubao_api(model_config, question)
        elif api_type == 'zhinao':
            return call_zhinao_api(model_config, question)
        elif api_type == 'hunyuan':
            return call_hunyuan_api(model_config, question)
        elif api_type == 'spark':
            return call_spark_api(model_config, question)
        else:
            # 默认使用通用 OpenAI 兼容接口
            return call_openai_compatible_api(model_config, question)
    except Exception as e:
        print(f"  -> FATAL ERROR for {model_config['name']} on QID {question['id']}: {e}")
        return None


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Run domestic brand analysis data collection.")
    parser.add_argument('--config', type=str, default='config_domestic.yaml', help='Path to the configuration file.')
    parser.add_argument('--task', type=str, help='Specify the task/category to run (e.g., snack, phone).')
    args = parser.parse_args()

    config_path = os.path.join(BASE_DIR, args.config)

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    # 加载 .env 文件
    load_dotenv(os.path.join(os.path.dirname(BASE_DIR), '.env'))

    config = load_config(config_path)

    # 确保结果目录存在
    results_dir = os.path.join(BASE_DIR, config['paths']['results_dir'])
    os.makedirs(results_dir, exist_ok=True)

    # 动态加载问题文件
    if args.task:
        questions_file_name = f"question/questions_{args.task}.json"
        print(f"-> Running task: {args.task}. Loading questions from {questions_file_name}")
    else:
        questions_file_name = config['paths']['questions_file']
        print(f"-> Running default task. Loading questions from {questions_file_name}")

    questions_path = os.path.join(BASE_DIR, questions_file_name)

    if not os.path.exists(questions_path):
        print(f"Error: Questions file not found at {questions_path}. Please create it.")
        return
    questions = load_questions(questions_path)

    all_results = []

    for model_key, model_config in config['models'].items():
        print(f"\n{'=' * 60}")
        print(f"Starting data collection for Model: {model_config['name']}")
        print(f"{'=' * 60}")

        model_output_path = os.path.join(results_dir, f"results_{model_key}.json")
        model_results = []

        if os.path.exists(model_output_path):
            try:
                with open(model_output_path, 'r', encoding='utf-8') as f:
                    model_results = json.load(f)
                print(f"  -> Found {len(model_results)} existing results. Will re-run all questions and append.")
            except json.JSONDecodeError:
                print(f"  -> Warning: Could not read existing results file {model_output_path}. Starting fresh.")
                model_results = []

        newly_collected_results = []

        for question in questions:
            q_id = question['id']
            print(f"  -> Question ID: {q_id} ({question['category']})")

            result = call_model(model_key, model_config, question)

            if result:
                model_results.append(result)
                newly_collected_results.append(result)

        if newly_collected_results:
            with open(model_output_path, 'w', encoding='utf-8') as f:
                json.dump(model_results, f, ensure_ascii=False, indent=4)
            print(
                f"--- Saved {len(newly_collected_results)} new results for {model_config['name']} to {model_output_path} ---")
        else:
            print(f"--- No new results collected for {model_config['name']}. ---")

        all_results.extend(model_results)

    # 按品类分开保存合并结果
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
