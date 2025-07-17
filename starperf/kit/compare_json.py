import json
import os
from random import random
import random


def compare_json_files(file1_path, file2_path):
    """
    比较两个JSON文件并计算变化项的百分比

    Args:
        file1_path: 第一个JSON文件路径
        file2_path: 第二个JSON文件路径

    Returns:
        dict: 包含比较结果的字典
    """
    try:
        # 读取JSON文件
        with open(file1_path, 'r', encoding='utf-8') as f1:
            json1 = json.load(f1)

        with open(file2_path, 'r', encoding='utf-8') as f2:
            json2 = json.load(f2)

        # 获取所有键
        keys1 = set(json1.keys())
        keys2 = set(json2.keys())
        all_keys = keys1.union(keys2)

        # 检查两个文件是否有相同的键
        if len(keys1) != len(keys2):
            print(f"警告: 两个文件的项数不同 - 文件1: {len(keys1)}项, 文件2: {len(keys2)}项")

        total_items = len(all_keys)
        changed_items = 0
        unchanged_items = 0
        details = []

        print(f"开始比较 {total_items} 项数据...\n")

        # 比较每一项
        for key in sorted(all_keys, key=lambda x: int(x) if x.isdigit() else x):
            value1 = json1.get(key)
            value2 = json2.get(key)

            # 如果某个键在其中一个文件中不存在，算作变化
            if value1 is None or value2 is None:
                changed_items += 1
                #print(f"项 \"{key}\": 变化 (其中一个文件中不存在该键)")
                details.append({
                    'key': key,
                    'status': 'changed',
                    'reason': '其中一个文件中不存在该键'
                })
                continue

            # 比较数组
            if value1 != value2:
                changed_items += 1
                #print(f"项 \"{key}\": 变化")
                #print(f"  文件1: {value1}")
                #print(f"  文件2: {value2}")
                details.append({
                    'key': key,
                    'status': 'changed',
                    'value1': value1,
                    'value2': value2
                })
            else:
                unchanged_items += 1
                #print(f"项 \"{key}\": 无变化")
                details.append({
                    'key': key,
                    'status': 'unchanged'
                })

        # 计算百分比
        change_percentage = (changed_items / total_items) * 100
        unchanged_percentage = (unchanged_items / total_items) * 100

        # 输出结果
        print('\n' + '=' * 50)
        print('比较结果')
        print('=' * 50)
        print(f"总项数: {total_items}")
        print(f"变化项数: {changed_items}")
        print(f"未变化项数: {unchanged_items}")
        print(f"变化百分比: {change_percentage:.2f}%")
        print(f"未变化百分比: {unchanged_percentage:.2f}%")

        return {
            'total_items': total_items,
            'changed_items': changed_items,
            'unchanged_items': unchanged_items,
            'change_percentage': round(change_percentage, 2),
            'unchanged_percentage': round(unchanged_percentage, 2),
            'details': details
        }

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式错误 - {e}")
        return None
    except Exception as e:
        print(f"错误: {e}")
        return None


def main():
    """主函数"""
    print("JSON文件比较工具")
    print("=" * 50)

    change = []
    # 获取文件路径
    for num in range(1, 5730):
        num_one = random.randint(0, 3000)
        num_two = num_one + 1
        file1_path = f'neighbors_slot_{num_one}.json'
        file2_path = f'neighbors_slot_{num_two}.json'

        # 检查文件是否存在
        if not os.path.exists(file1_path):
            print(f"错误: 文件 '{file1_path}' 不存在")
            return

        if not os.path.exists(file2_path):
            print(f"错误: 文件 '{file2_path}' 不存在")
            return

        print(f"\n开始比较文件:")
        print(f"文件1: {file1_path}")
        print(f"文件2: {file2_path}")
        print("-" * 50)

        # 执行比较
        result = compare_json_files(file1_path, file2_path)
        change.append(result.get('changed_items'))
        if result:
            print("\n比较完成！")
        else:
            print("\n比较失败，请检查文件路径和格式。")

    total_change = sum(change)
    len_change = len(change)
    change_percentage = total_change / len_change
    print(f"平均改变项数：{change_percentage}")





if __name__ == "__main__":
    main()