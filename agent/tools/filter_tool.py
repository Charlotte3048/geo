from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from agent.pipelines.domestic_pipeline import filter_questions_domestic

class FilterInput(BaseModel):
    category: str = Field(..., description="英文分类 key，如 snack / beauty / food")

def filter_tool_func(category: str):
    return filter_questions_domestic(category)

filter_tool = StructuredTool.from_function(
    name="filter_questions",
    func=filter_tool_func,
    args_schema=FilterInput,
    description="根据分类过滤 questions_domestic.json，生成 temp/questions_xxx.json"
)
