# agent/tools/collection_tool.py

import subprocess
import os
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool


class CollectionInput(BaseModel):
    category: str = Field(..., description="英文分类 key，如 snack")
    config_path: str = Field(..., description="Config Agent 生成的 YAML 配置路径")


def run_collection_process(category: str, config_path: str) -> str:
    """
    通过 subprocess 调用 run_analysis_domestic.py
    返回结果 JSON 文件路径
    """
    config_path = os.path.abspath(config_path)
    script_path = "domestic/run_analysis_domestic.py"

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"找不到 pipeline 脚本：{script_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config 文件不存在：{config_path}")

    # 调用采集脚本
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    process = subprocess.Popen(
        ["python", script_path, "--config", config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # 实时打印 pipeline 的输出
    print("[CollectionAgent] 开始采集…")
    for line in process.stdout:
        print("[Pipeline]", line.strip())

    process.wait()  # 等待进程结束

    # 根据你的 pipeline ，输出 JSON 位置为：
    result_file = f"domestic/merged_results/results_{category}_merged.json"

    if not os.path.exists(result_file):
        raise FileNotFoundError(f"采集未生成结果文件：{result_file}")

    return result_file


collection_tool = StructuredTool.from_function(
    name="run_collection_process",
    func=run_collection_process,
    args_schema=CollectionInput,
    description="执行模型采集任务（调用 run_analysis_domestic.py）"
)
