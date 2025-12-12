# agent/tools/collection_tool.py

import subprocess
import os
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool


class CollectionInput(BaseModel):
    category: str = Field(..., description="英文分类 key，如 snack")
    config_path: str = Field(..., description="Config Agent 生成的 YAML 配置路径")


def render_progress_bar(done: int, total: int, width: int = 20) -> str:
    """
    根据当前完成数量渲染一个文本进度条，例如：
    [██████░░░░░░░░░░]  9/30 (30%)
    """
    if total <= 0:
        total = 1
    ratio = done / total
    ratio = max(0.0, min(1.0, ratio))
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    percent = int(ratio * 100)
    return f"[{bar}] {done}/{total} ({percent}%)"


def run_collection_process(category: str, config_path: str) -> str:
    """
    使用 subprocess 调用 run_analysis_domestic.py
    返回生成的 merged_results JSON 路径
    """
    import os
    import subprocess

    # 绝对路径，防止 nested domestic
    config_path = os.path.abspath(config_path)
    script_path = os.path.abspath("domestic/run_analysis_domestic.py")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config 文件不存在：{config_path}")

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Pipeline 脚本不存在：{script_path}")

    # ====== 这里开始：带文本进度条的实时采集输出 ======
    TOTAL_QUESTIONS = 30  # 你现在每个品类都是 30 道题

    print(f"[CollectionAgent] 开始采集类别 {category}，共约 {TOTAL_QUESTIONS} 道题…")

    process = subprocess.Popen(
        ["python", "-u", script_path, "--config", config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # 行缓冲
    )

    done = 0
    current_model = None

    # 实时读取 stdout
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue

        # 打印原始日志（可选）
        # print("[Pipeline]", line)

        # 识别“正在跑哪个模型”
        if "Starting data collection for Model:" in line:
            # 例如：--- Starting data collection for Model: DeepSeek ---
            current_model = line.split("Model:")[-1].strip("- ").strip()
            print(f"\n[Model] 正在采集：{current_model}")

        # 识别“当前问题进度”（例如：-> Question ID: 161）
        if "Question ID:" in line:
            done += 1
            bar = render_progress_bar(done, TOTAL_QUESTIONS)
            # 用 \r 回到行首覆盖 + end="" 避免换行
            print(f"\r[Progress] {bar}", end="", flush=True)

    process.wait()
    print()  # 换行，避免最后一行进度条挤在一起

    # 打印 stderr 方便排查
    stderr = process.stderr.read()
    if stderr:
        print("[CollectionAgent stderr]")
        print(stderr)

    # ====== 结果文件路径（保持你现在的逻辑） ======
    result_file = os.path.abspath(
        f"domestic/merged_results/results_{category}_merged.json"
    )

    if not os.path.exists(result_file):
        raise FileNotFoundError(f"未找到结果文件：{result_file}")

    print(f"\n[CollectionAgent] 采集完成，结果文件：{result_file}")
    return result_file


collection_tool = StructuredTool.from_function(
    name="run_collection_process",
    func=run_collection_process,
    args_schema=CollectionInput,
    description="执行模型采集任务（调用 run_analysis_domestic.py）"
)
