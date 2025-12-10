# agent/agents/config_agent.py

import os
from dotenv import load_dotenv

# ====== ① 加载环境变量（必须在其他 import 之前执行） ======
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)
print("[DEBUG] Loaded OPENROUTER_API_KEY in ConfigAgent =", os.getenv("OPENROUTER_API_KEY"))

# ====== ② 支持 Tool Calling 的模型（必须使用 langchain_openai） ======
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# ====== ③ 导入 ConfigTool ======
from agent.tools.config_tool import config_tool


# ====== ④ 初始化 LLM ======
llm = ChatOpenAI(
    model="openai/gpt-4o",        # 或 gpt-4o，需要工具调用能力
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)


# ====== ⑤ Config Agent Prompt ======
config_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是 Config Agent。你的任务是根据 category 和 questions_path 调用工具 "
     "`generate_runtime_config` 来生成运行时 YAML 配置文件。\n"
     "你只能调用工具，不允许自己创建 YAML。\n"
     "工具的输出（路径）必须直接作为你的最终回答。\n"
     "不要输出任何解释性的文字。"),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}")
])


# ====== ⑥ 创建 Agent（绑定工具） ======
config_agent = create_tool_calling_agent(
    llm,
    [config_tool],
    config_prompt
)


# ====== ⑦ 创建 AgentExecutor ======
config_agent_executor = AgentExecutor(
    agent=config_agent,
    tools=[config_tool],
    verbose=True
)


# ====== ⑧ 对外暴露接口（Planner 调用此函数） ======
def run_config_agent(category: str, questions_path: str):
    """
    输入：category（英文分类 key）
         questions_path（过滤后的问题 JSON 文件路径）
    输出：runtime YAML config 文件路径
    """
    return config_agent_executor.invoke({
        "input": {
            "category": category,
            "questions_path": questions_path
        }
    })
