# merge_results.py
import json
import os
import glob

# ==============================================================================
# 结果合并工具 (v1.0)
# 描述: 一个用于合并多个 results-*.json 文件的辅助脚本。
# ==============================================================================
# 用法: python merge_results.py
# ==============================================================================

# --- 配置 ---
# 使用 glob 模式匹配所有需要合并的文件
# 我们假设旧的GPT结果在 'results.json'，新结果在 'results_ha_*.json'
FILES_TO_MERGE = glob.glob("results_ha_*.json") + ["results.json"]

# 定义合并后输出的文件名
MERGED_OUTPUT_FILE = "results_merged_ha.json"


def main():
    """主执行函数"""
    print("--- 开始合并结果文件 ---")

    merged_data = []
    seen_ids = set()  # 用于去重，防止意外重复添加

    print(f"准备合并以下文件: {FILES_TO_MERGE}")

    for file_path in FILES_TO_MERGE:
        if not os.path.exists(file_path):
            print(f"警告: 文件 '{file_path}' 不存在，已跳过。")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 确保data是列表格式
                if not isinstance(data, list):
                    print(f"警告: 文件 '{file_path}' 的内容不是一个JSON列表，已跳过。")
                    continue

                count = 0
                for item in data:
                    # 确保每个item都有一个唯一的标识符，这里我们用 (id, task) 作为联合主键
                    item_id = item.get('id')
                    task_name = item.get('task', 'unknown')  # 兼容旧的 results.json 可能没有task字段
                    unique_key = (item_id, task_name)

                    if unique_key not in seen_ids:
                        merged_data.append(item)
                        seen_ids.add(unique_key)
                        count += 1

                print(f"  - 已从 '{file_path}' 加载并合并 {count} 条记录。")

        except json.JSONDecodeError:
            print(f"警告: 文件 '{file_path}' 不是有效的JSON格式，已跳过。")
        except Exception as e:
            print(f"处理文件 '{file_path}' 时发生未知错误: {e}")

    # 保存合并后的数据
    try:
        with open(MERGED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

        print("\n--- 合并完成！---")
        print(f"总共 {len(merged_data)} 条记录已成功合并并保存到 '{MERGED_OUTPUT_FILE}'。")
        print("现在，您可以使用这个合并后的文件进行下一步的品牌探索和最终分析。")

    except Exception as e:
        print(f"\n保存合并文件时发生错误: {e}")


if __name__ == "__main__":
    main()
