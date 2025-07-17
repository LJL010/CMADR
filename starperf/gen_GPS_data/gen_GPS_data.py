import requests
import json
from datetime import datetime, timedelta
import starperf.kit.get_satellite_position as GET_SATELLITE_POSITION


class Satellite:
    def __init__(self, tle_line1, tle_line2, name):
        self.tle_2le = [tle_line1, tle_line2]
        self.name = name


class SatelliteShell:
    def __init__(self, satellites):
        self.satellites = satellites
        # GPS卫星平均轨道周期约11小时58分钟，单位：秒
        self.orbit_cycle = 11 * 3600 + 58 * 60


def fetch_gps_tle_data(url):
    """爬取指定URL的GPS卫星TLE数据"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"数据爬取失败: {str(e)}")
        return ""


def parse_tle_data(tle_text):
    """解析TLE文本数据为卫星对象列表"""
    satellites = []
    lines = tle_text.strip().split('\n')

    # 处理每组TLE数据（每3行为一组：卫星名称、第一行TLE、第二行TLE）
    i = 0
    while i < len(lines):
        # 跳过空行
        if not lines[i].strip():
            i += 1
            continue

        # 检查是否有足够的行来构成一个完整的TLE集合
        if i + 2 >= len(lines):
            break

        # 获取卫星名称（第一行）
        if lines[i].strip() and not lines[i].startswith('1 ') and not lines[i].startswith('2 '):
            satellite_name = lines[i].strip()
            i += 1
        else:
            satellite_name = "Unknown Satellite"

        # 检查接下来的两行是否是有效的TLE数据
        if i + 1 < len(lines) and lines[i].startswith('1 ') and lines[i + 1].startswith('2 '):
            tle_line1 = lines[i].strip()
            tle_line2 = lines[i + 1].strip()

            # 验证TLE数据格式的基本正确性
            if len(tle_line1) >= 69 and len(tle_line2) >= 69:
                try:
                    # 提取NORAD ID进行验证
                    norad_id_line1 = tle_line1[2:7].strip()
                    norad_id_line2 = tle_line2[2:7].strip()

                    if norad_id_line1 == norad_id_line2:
                        satellites.append(Satellite(tle_line1, tle_line2, satellite_name))
                        print(f"成功解析卫星: {satellite_name} (NORAD ID: {norad_id_line1})")
                    else:
                        print(f"警告: 卫星 {satellite_name} 的TLE数据NORAD ID不匹配")
                except Exception as e:
                    print(f"解析卫星 {satellite_name} 时出错: {str(e)}")
            else:
                print(f"警告: 卫星 {satellite_name} 的TLE数据长度不正确")
        else:
            print(f"警告: 无法找到卫星 {satellite_name} 的完整TLE数据")

        i += 2

    return satellites


def parse_tle_data_alternative(tle_text):
    """
    备选的TLE解析方法，适用于没有卫星名称的情况
    """
    satellites = []
    lines = [line.strip() for line in tle_text.strip().split('\n') if line.strip()]

    # 处理每两行为一组的TLE数据
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith('1 ') and lines[i + 1].startswith('2 '):
            tle_line1 = lines[i]
            tle_line2 = lines[i + 1]

            # 验证TLE数据格式
            if len(tle_line1) >= 69 and len(tle_line2) >= 69:
                try:
                    # 提取NORAD ID
                    norad_id = tle_line1[2:7].strip()
                    satellite_name = f"GPS Satellite {norad_id}"

                    # 验证两行的NORAD ID是否匹配
                    if tle_line1[2:7] == tle_line2[2:7]:
                        satellites.append(Satellite(tle_line1, tle_line2, satellite_name))
                        print(f"成功解析卫星: {satellite_name}")
                    else:
                        print(f"警告: TLE数据NORAD ID不匹配 (行 {i + 1} 和 {i + 2})")
                except Exception as e:
                    print(f"解析TLE数据时出错 (行 {i + 1}): {str(e)}")
            else:
                print(f"警告: TLE数据长度不正确 (行 {i + 1})")

    return satellites


def save_longitude_latitude_altitude_data(current_time, orbit_period, shell, output_file="satellite_positions.json"):
    """生成并保存卫星轨道周期内的位置数据"""
    # 生成时间点列表（每12秒一个点，直到轨道周期结束）
    orbit_period = orbit_period
    moments = []
    current_time = current_time
    end_time = current_time + 2*timedelta(seconds=orbit_period)
    while current_time <= end_time:
        moments.append(current_time)
        current_time += timedelta(seconds=2)

    # 初始化结果结构：[时间点][卫星][经纬度]
    sat_positions_per_slot = []
    for moment in moments:
        satellite_positions = []
        for satellite in shell.satellites:
            TLE_2LE = [satellite.tle_2le[0], satellite.tle_2le[1]]

            year, month, day = moment.year, moment.month, moment.day
            hour, minute, second = moment.hour, moment.minute, moment.second

            try:
                position = GET_SATELLITE_POSITION.get_satellite_position(
                    TLE_2LE, year, month, day, hour, minute, second
                )
                longitude = position[0][0]
                latitude = position[0][1]
                altitude = position[0][2]
                # 添加到当前时间点的卫星位置列表
                satellite_positions.append([longitude, latitude, altitude])
            except Exception as e:
                print(f"计算卫星 {satellite.name} 位置时出错: {str(e)}")
                # 如果计算失败，添加空值或默认值
                satellite_positions.append([None, None, None])

        # 将当前时间点的所有卫星位置添加到结果中
        sat_positions_per_slot.append(satellite_positions)

    # 构建完整JSON结构
    result = {
        "sat_positions_per_slot_GPS": sat_positions_per_slot
    }

    # 保存到JSON文件
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"卫星位置数据已保存到 {output_file}")
    print(f"共处理了 {len(shell.satellites)} 颗卫星，{len(moments)} 个时间点")


def main(current_time, orbit_period):
    # 更新URL为TLE格式数据源
    gps_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=GPS-OPS&FORMAT=TLE"

    # 爬取TLE数据
    tle_text = fetch_gps_tle_data(gps_url)
    if not tle_text:
        print("无数据可处理，程序退出")
        return

    print("开始解析TLE数据...")
    print(f"原始数据长度: {len(tle_text)} 字符")

    # 解析TLE数据
    satellites = parse_tle_data(tle_text)

    # 如果第一种方法没有解析到卫星，尝试备选方法
    if not satellites:
        print("尝试备选解析方法...")
        satellites = parse_tle_data_alternative(tle_text)

    if not satellites:
        print("无法解析任何卫星数据，程序退出")
        return

    print(f"成功解析 {len(satellites)} 颗卫星")

    # 创建卫星壳层对象
    shell = SatelliteShell(satellites)

    # 生成并保存位置数据
    save_longitude_latitude_altitude_data(current_time, orbit_period, shell, "gps_satellite_positions.json")


if __name__ == "__main__":
    main()