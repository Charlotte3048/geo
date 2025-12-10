import os
from dotenv import load_dotenv

# ==========================================================
# ① 在任何 import 其他模块之前加载 .env
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

# ==========================================================
# ② 下面才开始 import Agent / Tools / LLM
# ==========================================================
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from agent.tools.filter_tool import filter_tool

# ==========================================================
# ③ 初始化 LLM（此时 getenv 必须已经生效）
# ==========================================================
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

# ==========================================================
# ④ 构建 Agent
# ==========================================================

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是 Filter Agent，你的唯一任务是根据用户输入的 category 调用工具 filter_questions。\n"
     "⚠️ 你只能调用工具，不允许生成任何自然语言。\n"
     "⚠️ 工具返回的内容（路径字符串）必须直接作为你的最终回答。\n"
     "⚠️ 不要添加任何解释、前缀、总结、Markdown 标记。\n"
     "最终输出只能是一个纯路径，例如：temp/questions_snack.json"
    ),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}")
])



agent = create_tool_calling_agent(llm, [filter_tool], prompt)

filter_agent_executor = AgentExecutor(
    agent=agent,
    tools=[filter_tool],
    verbose=True
)


def run_filter_agent(text: str):
    return filter_agent_executor.invoke({"input": text})
