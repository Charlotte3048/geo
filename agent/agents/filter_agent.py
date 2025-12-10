import os
from dotenv import load_dotenv

# ==========================================================
# ① 在任何 import 其他模块之前加载 .env
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)
print("[DEBUG] Loaded OPENROUTER_API_KEY =", os.getenv("OPENROUTER_API_KEY"))

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
     "你是 Filter Agent。任务：根据用户输入找出对应的**唯一英文 category key**。"
     "你的调用规则："
     "1. 你只能返回 EXACT ONE category。"
     "2. 不能自动拆分类（如 snack→ snack+beverage）。"
     "3. 如果用户提到的是“零食饮料”，你必须映射为 SNACK（snack）。"
     "4. category 的取值必须来自：['snack','beauty','food','city','luxury','nev','phone','scenic']"
     "严格遵守，不得联想不存在的品类。"
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
