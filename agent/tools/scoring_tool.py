# agent/tools/scoring_tool.py
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from agent.pipelines.scoring_pipeline import run_scoring_pipeline

class ScoringInput(BaseModel):
    category: str = Field(..., description="品类英文key，例如 snack")
    results_path: str = Field(..., description="merged_results json 路径")

def scoring_func(category: str, results_path: str) -> str:
    return run_scoring_pipeline(category, results_path)

scoring_tool = StructuredTool.from_function(
    name="score_results",
    func=scoring_func,
    args_schema=ScoringInput,
    description="对 merged_results 进行品牌评分，输出 scores_{category}.json"
)
