# agent/agents/collection_agent.py

import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from agent.tools.collection_tool import collection_tool

llm = ChatOpenAI(
    model="openai/gpt-4o",  # 必须支持 tool calling
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

collection_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是 Collection Agent，你的任务是根据 category 和 config_path "
     "调用工具 run_collection_process 来执行模型采集。\n"
     "你只能调用工具，不允许生成自己的输出内容。\n"
     "不要解释，只需调用工具并返回工具输出。"),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

collection_agent = create_tool_calling_agent(
    llm,
    [collection_tool],
    collection_prompt
)

collection_agent_executor = AgentExecutor(
    agent=collection_agent,
    tools=[collection_tool],
    verbose=True
)


def run_collection_agent(category: str, config_path: str):
    return collection_agent_executor.invoke({
        "input": {
            "category": category,
            "config_path": config_path
        }
    })
