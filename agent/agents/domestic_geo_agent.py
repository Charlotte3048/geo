import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agent.tools.domestic_tools import run_domestic_pipeline

# python -c "from agent.agents.domestic_geo_agent import executor_run; print(executor_run('帮我跑一下国内零食饮料的排行榜'))"

# load .env
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---- TOOL SCHEMA ----
class DomesticPipelineInput(BaseModel):
    category: str = Field(..., description="英文分类 key，如 snack / beauty / food")

def call_domestic_pipeline(category: str):
    return run_domestic_pipeline.run(category)

run_domestic_pipeline_tool = StructuredTool.from_function(
    name="run_domestic_pipeline",
    func=call_domestic_pipeline,
    args_schema=DomesticPipelineInput,
    description="根据英文分类 key 调用国内 GEO pipeline"
)

tools = [run_domestic_pipeline_tool]

# ---- PROMPT ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是 GEO 国内 Agent，会自动把用户中文品类映射为英文 category。"),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}")
])


# ---- LLM ----
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

# ---- 真正的 FUNCTION-CALLING AGENT ----
agent = create_tool_calling_agent(llm, tools, prompt)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def executor_run(text: str):
    return executor.invoke({"input": text})
