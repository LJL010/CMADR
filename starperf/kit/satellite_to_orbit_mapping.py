'''

Author : yunanhou

Date : 2023/12/02

Function : The orbits within a shell are derived based on a clustering algorithm and the satellites are assigned to
           these orbits.

           Specifically, before executing this script, the corresponding relationship between satellites and shells has
           been established, but the relationship between satellites and orbit points has not yet been established. The
           function of this script is to establish the corresponding relationship between satellites and orbit.

           The main function of the script needs to pass in a shell class object, and then cluster the raan of each
           satellite in the shell object to obtain several orbits, and then assign all satellites in the shell to these
           orbits.

'''
import jenkspy
import starperf.entity.orbit as ORBIT
import matplotlib.pyplot as plt

import threading

# 创建全局锁（适用于多线程）
_global_lock = threading.Lock()
# Parameter :
# shells : a collection of shell objects that have established corresponding relationships
# Return Value :
# after the function is executed, the mapping relationship between satellite, orbit, and shell has been established
# without any return value.
def satellite_to_orbit_mapping(shells):
    # 预定义的轨道数量列表
    predefined_orbit_counts = [1, 6, 5, 21, 30, 20, 30, 20, 20, 17, 2, 20]

    # 把交互的raans提前统计好，在下面直接使用
    with _global_lock:
        try:
            for shell_index, sh in enumerate(shells):
                # 检查是否有足够的预定义轨道数量
                if shell_index >= len(predefined_orbit_counts):
                    raise ValueError(f"没有足够的预定义轨道数量，shell索引 {shell_index} 超出范围")

                # 获取当前shell对应的预定义轨道数量
                orbits_number = predefined_orbit_counts[shell_index]

                # extract the raan of all satellites in sh
                raans = []
                for sat in sh.satellites:
                    raans.append(sat.tle_json["RA_OF_ASC_NODE"])
                raans = sorted(raans)

                # 可选：仍然显示图表以便可视化
                plt.plot(raans)
                plt.ylabel('RAANS')
                plt.title(f'Shell {shell_index} - Using {orbits_number} orbits')
                plt.savefig(f'shell_{shell_index}_raans.png')  # 保存图表而不是显示
                plt.close()  # 关闭图表以释放资源

                breaks = jenkspy.jenks_breaks(values=raans, n_classes=orbits_number)
                orbit_raans = [(breaks[i], breaks[i + 1]) for i in range(len(breaks) - 1)]
                for ra_index, ra in enumerate(orbit_raans):
                    lower_bound = ra[0]
                    upper_bound = ra[1]
                    orbit = ORBIT.orbit(shell=sh, raan_lower_bound=lower_bound, raan_upper_bound=upper_bound)
                    for sat in sh.satellites:
                        if ra_index > 0:
                            if sat.tle_json["RA_OF_ASC_NODE"] > lower_bound and sat.tle_json[
                                "RA_OF_ASC_NODE"] <= upper_bound:
                                sat.orbit = orbit
                                orbit.satellites.append(sat)
                        if ra_index == 0:
                            if sat.tle_json["RA_OF_ASC_NODE"] >= lower_bound and sat.tle_json[
                                "RA_OF_ASC_NODE"] <= upper_bound:
                                sat.orbit = orbit
                                orbit.satellites.append(sat)

                    sh.orbits.append(orbit)
        except Exception as e:
            raise e  # 可选：决定是否继续抛出异常
        finally:
            # 关闭所有图表（释放资源）
            plt.close('all')
