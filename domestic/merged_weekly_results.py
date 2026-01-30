"""
合并merged_results目录下的JSON文件，按照周和品类为单位进行合并
输出到domestic/weekly_results文件夹下
支持增量合并，避免重复处理已合并的文件
"""

import json
import os
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path


def parse_filename(filename):
    """
    解析文件名，提取品类和时间戳
    格式: results_{category}_merged_{timestamp}.json
    或: results_{category}_merged.json

    返回: (category, timestamp) 或 (category, None)
    """
    # 移除.json后缀
    name = filename.replace('.json', '')
    parts = name.split('_')

    # 查找merged的位置
    if 'merged' not in parts:
        return None, None

    merged_idx = parts.index('merged')

    # 提取品类 (results和merged之间的部分)
    if merged_idx < 2:
        return None, None

    category = '_'.join(parts[1:merged_idx])

    # 提取时间戳 (merged之后的部分)
    timestamp = None
    if merged_idx + 1 < len(parts):
        timestamp_str = parts[merged_idx + 1]
        # 验证时间戳格式 YYYYMMDD
        if len(timestamp_str) == 8 and timestamp_str.isdigit():
            timestamp = timestamp_str

    return category, timestamp


def get_week_number(date_str):
    """
    根据日期字符串(YYYYMMDD)获取周数
    返回格式: YYYY-WXX (例如: 2026-W03)
    """
    if not date_str or len(date_str) != 8:
        return None

    try:
        date = datetime.strptime(date_str, '%Y%m%d')
        # 获取ISO周数
        year, week, _ = date.isocalendar()
        return f"{year}-W{week:02d}"
    except ValueError:
        return None


def get_existing_weekly_files(output_dir):
    """
    获取输出目录中已存在的周报文件
    返回: {(week, category): filepath}
    """
    existing_files = {}

    if not os.path.exists(output_dir):
        return existing_files

    for filename in os.listdir(output_dir):
        if not filename.endswith('.json'):
            continue

        # 解析输出文件名格式: results_{category}_weekly_{week}.json
        if '_weekly_' not in filename:
            continue

        name = filename.replace('.json', '')
        parts = name.split('_weekly_')

        if len(parts) == 2:
            # 提取品类 (results_之后的部分)
            category_part = parts[0]
            if category_part.startswith('results_'):
                category = category_part[8:]  # 移除 'results_' 前缀
                week = parts[1]
                existing_files[(week, category)] = os.path.join(output_dir, filename)

    return existing_files


def merge_weekly_results(input_dir, output_dir, force=False, verbose=True):
    """
    合并JSON文件，按周和品类分组

    Args:
        input_dir: 输入目录路径 (merged_results)
        output_dir: 输出目录路径 (domestic/weekly_results)
        force: 是否强制重新合并已存在的文件
        verbose: 是否显示详细日志
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 获取已存在的周报文件
    existing_files = get_existing_weekly_files(output_dir)

    if existing_files and not force:
        print(f"发现 {len(existing_files)} 个已存在的周报文件")
        print("增量模式: 将跳过已合并的周次和品类")
        print("如需重新合并所有文件，请使用 --force 参数")
        print("-" * 60)

    # 用于存储按周和品类分组的数据
    # 结构: {week: {category: [data1, data2, ...]}}
    weekly_data = defaultdict(lambda: defaultdict(list))

    # 用于存储没有时间戳的文件
    no_timestamp_data = defaultdict(list)

    # 统计信息
    stats = {
        'processed': 0,
        'skipped': 0,
        'new_files': 0,
        'updated_files': 0,
        'no_timestamp': 0
    }

    # 遍历输入目录中的所有JSON文件
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(input_dir, filename)

        # 解析文件名
        category, timestamp = parse_filename(filename)

        if not category:
            if verbose:
                print(f"⚠ 警告: 无法解析文件名 {filename}, 跳过")
            continue

        # 读取JSON数据
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 确保数据是列表格式
            if not isinstance(data, list):
                if verbose:
                    print(f"⚠ 警告: {filename} 不是列表格式, 跳过")
                continue

        except json.JSONDecodeError as e:
            print(f"✗ 错误: 无法解析JSON文件 {filename}: {e}")
            continue
        except Exception as e:
            print(f"✗ 错误: 读取文件 {filename} 失败: {e}")
            continue

        # 根据是否有时间戳进行分组
        if timestamp:
            week = get_week_number(timestamp)
            if week:
                # 检查是否已存在且不强制覆盖
                if not force and (week, category) in existing_files:
                    stats['skipped'] += 1
                    if verbose:
                        print(f"⊘ 跳过: {filename} -> 周 {week}, 品类 {category} (已存在)")
                    continue

                weekly_data[week][category].extend(data)
                stats['processed'] += 1
                if verbose:
                    print(f"✓ 处理: {filename} -> 周 {week}, 品类 {category}, 数据条数 {len(data)}")
            else:
                if verbose:
                    print(f"⚠ 警告: 无法解析时间戳 {timestamp} 在文件 {filename}")
                no_timestamp_data[category].extend(data)
                stats['no_timestamp'] += 1
        else:
            # 没有时间戳的文件单独处理
            no_timestamp_data[category].extend(data)
            stats['no_timestamp'] += 1
            if verbose:
                print(f"✓ 处理: {filename} -> 品类 {category} (无时间戳), 数据条数 {len(data)}")

    print("\n" + "=" * 60)
    print("开始保存合并结果...")
    print("=" * 60)

    # 保存按周分组的数据
    for week, categories in sorted(weekly_data.items()):
        for category, data in categories.items():
            output_filename = f"results_{category}_weekly_{week}.json"
            output_path = os.path.join(output_dir, output_filename)

            # 判断是新文件还是更新文件
            is_new = not os.path.exists(output_path)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            if is_new:
                stats['new_files'] += 1
                print(f"✓ 新建: {output_filename}, 共 {len(data)} 条数据")
            else:
                stats['updated_files'] += 1
                print(f"✓ 更新: {output_filename}, 共 {len(data)} 条数据")

    # 保存没有时间戳的数据
    if no_timestamp_data:
        print("\n处理无时间戳文件...")
        for category, data in no_timestamp_data.items():
            output_filename = f"results_{category}_no_timestamp.json"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"✓ 保存: {output_filename}, 共 {len(data)} 条数据 (无时间戳)")

    # 打印统计信息
    print("\n" + "=" * 60)
    print("合并完成! 统计信息:")
    print("=" * 60)
    print(f"处理文件数: {stats['processed']}")
    print(f"跳过文件数: {stats['skipped']} (已存在)")
    print(f"新建文件数: {stats['new_files']}")
    print(f"更新文件数: {stats['updated_files']}")
    print(f"无时间戳文件数: {stats['no_timestamp']}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='按周和品类合并merged_results目录下的JSON文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 增量合并 (默认，跳过已存在的文件)
  python3 merge_weekly_results.py

  # 强制重新合并所有文件
  python3 merge_weekly_results.py --force

  # 静默模式 (仅显示摘要)
  python3 merge_weekly_results.py --quiet

  # 指定自定义路径
  python3 merge_weekly_results.py --input /path/to/merged_results --output /path/to/weekly_results
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='输入目录路径 (默认: domestic/merged_results)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出目录路径 (默认: domestic/weekly_results)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='强制重新合并已存在的文件 (默认: 跳过已存在的文件)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，仅显示摘要信息'
    )

    args = parser.parse_args()

    # 默认路径配置
    script_dir = Path(__file__).parent
    input_dir = args.input if args.input else script_dir  / "merged_results"
    output_dir = args.output if args.output else script_dir  / "weekly_results"

    # 转换为Path对象
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"✗ 错误: 输入目录不存在: {input_dir}")
        print("请确保目录结构为: domestic/merged_results/")
        return

    print("=" * 60)
    print("周报合并工具")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"合并模式: {'强制覆盖' if args.force else '增量合并'}")
    print(f"日志模式: {'静默' if args.quiet else '详细'}")
    print("=" * 60 + "\n")

    # 执行合并
    merge_weekly_results(
        str(input_dir),
        str(output_dir),
        force=args.force,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
