# agent/agents/scoring_agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from agent.tools.scoring_tool import scoring_tool
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

llm = ChatOpenAI(
    model="openai/gpt-4o",  # 必须支持 tool calling
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是评分Agent。你只负责调用工具对结果文件打分并返回输出路径。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, [scoring_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[scoring_tool], verbose=True)


def run_scoring_agent(category: str, results_path: str):
    return executor.invoke({"input": {"category": category, "results_path": results_path}})
