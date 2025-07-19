import json


def check_array_dimensions(data):
    """检查sat_positions_per_slot数组的维度和结构"""
    dimensions = []

    # 递归检查维度
    def get_dim(arr, level=0):
        if not isinstance(arr, list):
            return
        # 添加当前维度的长度
        if level >= len(dimensions):
            dimensions.append(len(arr))
        else:
            # 验证同一维度的长度是否一致
            if dimensions[level] != len(arr):
                print(f"警告: 第{level + 1}维长度不一致: {dimensions[level]} vs {len(arr)}")
        # 递归检查子元素
        if arr and isinstance(arr[0], list):
            get_dim(arr[0], level + 1)

    get_dim(data)
    return dimensions


def validate_satellite_data(file_path):
    """验证包含经纬度和高度的卫星位置数据结构"""
    try:
        # 读取JSON文件
        with open(file_path, 'r') as f:
            data = json.load(f)

        # 检查是否包含所需的键
        if 'sat_positions_per_slot' not in data:
            print("错误: JSON数据中缺少'sat_positions_per_slot'键")
            return

        positions = data['sat_positions_per_slot']
        dimensions = check_array_dimensions(positions)

        print(f"数组维度: {len(dimensions)}维")
        print(f"各维度大小: {dimensions}")

        # 验证是否为三维数组
        if len(dimensions) == 3:
            print("结构验证通过: 符合三维数组要求 [时间点][卫星][经纬度+高度]")
        else:
            print(f"结构警告: 期望三维数组，但得到{len(dimensions)}维")

        # 检查最内层元素是否为[经度, 纬度, 高度]
        if len(dimensions) >= 2:
            sample = positions[0][0] if positions and positions[0] else []
            if len(sample) == 3 and all(isinstance(x, (int, float)) for x in sample):
                print("验证通过: 最内层元素为[经度, 纬度, 高度]格式")
            else:
                print(f"结构警告: 最内层元素格式应为[经度, 纬度, 高度]，当前长度: {len(sample)}")

    except Exception as e:
        print(f"处理文件时出错: {e}")


if __name__ == "__main__":
    # 使用示例（替换为实际文件路径）
    file_path = "../data/satellite_positions.json"
    validate_satellite_data(file_path)