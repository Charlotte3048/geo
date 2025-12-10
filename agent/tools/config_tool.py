# agent/tools/config_tool.py

import os
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool


# ===== 输入 Schema =====
class ConfigInput(BaseModel):
    category: str = Field(..., description="英文分类 key，如 snack")
    questions_path: str = Field(..., description="过滤后的问题 JSON 文件路径")


# ===== YAML 配置文件生成函数 =====
def generate_runtime_config(category: str, questions_path: str) -> str:
    """
    根据 category + questions_path 生成运行时 YAML 文件。
    返回 runtime 配置文件路径。
    """

    # 1. 文件存在性检查（根据你的 2B 选择）
    if not os.path.exists(questions_path):
        raise FileNotFoundError(
            f"[ConfigTool] questions_path 不存在：{questions_path}\n"
            f"请检查 FilterAgent 是否正确生成文件。"
        )

    # 2. 调用原 pipeline 的配置生成函数
    from agent.pipelines.domestic_pipeline import build_domestic_runtime_config

    runtime_path = build_domestic_runtime_config(category, questions_path)

    # 3. 返回 runtime 配置路径
    return runtime_path


# ===== 封装为 StructuredTool =====
config_tool = StructuredTool.from_function(
    name="generate_runtime_config",
    func=generate_runtime_config,
    args_schema=ConfigInput,
    description="根据 category 和 questions_path 生成国内 GEO pipeline 的 YAML 配置文件"
)
