# agent/pipelines/domestic_pipeline.py

import os
import json
import yaml
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DOMESTIC_DIR = os.path.join(BASE_DIR, "domestic")

CATEGORY_META = {
    "beauty": {"cn": "美妆护肤"},
    "city": {"cn": "中国旅游城市"},
    "food": {"cn": "餐饮美食"},
    "luxury": {"cn": "奢侈品"},
    "nev": {"cn": "新能源汽车"},
    "phone": {"cn": "智能手机"},
    "scenic": {"cn": "5A级景区"},
    "snack": {"cn": "零食饮料"},
}


def filter_questions_domestic(category_key: str,
                              questions_all_path: str = os.path.join(DOMESTIC_DIR, "questions_domestic.json")
                              ) -> str:
    cn_cat = CATEGORY_META[category_key]["cn"]

    with open(questions_all_path, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    subset = [q for q in all_questions if q["category"] == cn_cat]

    os.makedirs("temp", exist_ok=True)
    out_path = os.path.join("temp", f"questions_{category_key}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"[filter] {cn_cat} → {len(subset)} 条 → {out_path}")
    return out_path


TEMPLATE_CONFIG_PATH = os.path.join(DOMESTIC_DIR, "config_domestic.yaml")


def build_domestic_runtime_config(category_key: str,
                                  questions_subset_path: str) -> str:
    with open(TEMPLATE_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"]["questions_file"] = os.path.abspath(questions_subset_path)

    results_dir = os.path.join("domestic", "outputs", "results", category_key)
    os.makedirs(results_dir, exist_ok=True)
    cfg["paths"]["results_dir"] = results_dir

    runtime_dir = os.path.join("domestic", "config_runtime")
    os.makedirs(runtime_dir, exist_ok=True)

    runtime_cfg_path = os.path.join(runtime_dir, f"config_{category_key}.yaml")

    with open(runtime_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    print(f"[config] runtime config → {runtime_cfg_path}")
    return runtime_cfg_path


def run_domestic_collection(runtime_cfg_path: str, category_key: str) -> str:
    """
    调 run_analysis_domestic.py 并返回 merged 结果路径
    """
    # 注意：传入相对路径，而不是 domestic/config_runtime/xxx
    cfg_rel = os.path.relpath(runtime_cfg_path, "domestic")

    analysis_script = os.path.join(DOMESTIC_DIR, "run_analysis_domestic.py")
    cmd = ["python", analysis_script, "--config", cfg_rel]
    print(f"[run_analysis] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    merged_path = f"results_{category_key}_merged.json"
    return merged_path


def run_domestic_pipeline_raw(category_key: str) -> str:
    subset_path = filter_questions_domestic(category_key)
    runtime_cfg_path = build_domestic_runtime_config(category_key, subset_path)

    result_path = run_domestic_collection(runtime_cfg_path, category_key)

    print(f"[pipeline finished] merged result → {result_path}")
    return result_path
