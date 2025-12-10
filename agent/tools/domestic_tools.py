# agent/tools/domestic_tools.py

from langchain.tools import tool
from agent.pipelines.domestic_pipeline import run_domestic_pipeline_raw

@tool
def run_domestic_pipeline(category: str) -> str:
    """
    根据英文 category key（如 snack/beauty/...）运行国内 GEO pipeline，
    返回 merged results 的 JSON 文件路径。
    """
    return run_domestic_pipeline_raw(category)
