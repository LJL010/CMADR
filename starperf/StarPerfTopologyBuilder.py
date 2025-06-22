import numpy as np
from typing import Dict, List, Set


class StarPerfTopologyBuilder:
    def __init__(
            self,
            sat_positions: List[np.ndarray],  # 卫星位置列表（三维坐标）
            gs_positions: List[np.ndarray],  # 地面站位置列表（三维坐标）
            # 这里太死了，需要变成活的
            num_orbits: int,  # 壳层中的轨道数
            num_sat_per_orbit: int,  # 每条轨道的卫星数
            inclination: float,  # 轨道倾角（度）
            minimum_elevation: float = 25.0  # 最小仰角（度，默认值参考 StarPerf 案例）
    ):
        self.sat_positions = sat_positions
        self.gs_positions = gs_positions
        self.num_satellites = len(sat_positions)
        self.num_ground_stations = len(gs_positions)
        self.num_orbits = num_orbits
        self.num_sat_per_orbit = num_sat_per_orbit
        self.inclination = inclination
        self.minimum_elevation = np.radians(minimum_elevation)  # 转换为弧度

    def _build_satellite_isl(self) -> Dict[int, Set[int]]:
        """构建卫星间 ISL 连接（同轨道前后、同壳层左右）"""
        neighbors = {i: set() for i in range(self.num_satellites)}

        # 按轨道分组：假设卫星按轨道顺序排列（orbit_0: 0~n-1, orbit_1: n~2n-1, ...）
        orbit_groups = [
            list(range(oid * self.num_sat_per_orbit, (oid + 1) * self.num_sat_per_orbit))
            for oid in range(self.num_orbits)
        ]

        # 同轨道内：前后卫星连接（环形）
        for oid, orbit in enumerate(orbit_groups):
            for idx in range(len(orbit)):
                sat_id = orbit[idx]
                prev_sat_id = orbit[(idx - 1) % len(orbit)]  # 前一颗卫星
                next_sat_id = orbit[(idx + 1) % len(orbit)]  # 后一颗卫星
                neighbors[sat_id].add(prev_sat_id)
                neighbors[sat_id].add(next_sat_id)

        # 同壳层内：相邻轨道的左右卫星连接（简化逻辑，假设轨道为极轨道或 Walker 星座）
        for oid in range(self.num_orbits):
            for idx in range(self.num_sat_per_orbit):
                sat_id = oid * self.num_sat_per_orbit + idx
                current_pos = self.sat_positions[sat_id]

                # 寻找相邻轨道（左轨道和右轨道，需处理边界情况）
                for neighbor_oid in [oid - 1, oid + 1]:
                    if neighbor_oid < 0 or neighbor_oid >= self.num_orbits:
                        continue
                    neighbor_orbit = orbit_groups[neighbor_oid]

                    # 计算当前卫星在相邻轨道的最近卫星（简化为按位置索引偏移）
                    # 假设相邻轨道卫星相位差为 1（Walker 星座规则）
                    neighbor_idx = (idx + 1) % self.num_sat_per_orbit
                    neighbor_sat_id = neighbor_oid * self.num_sat_per_orbit + neighbor_idx
                    neighbors[sat_id].add(neighbor_sat_id)

        return neighbors

    def _build_satellite_ground_links(self) -> Dict[int, Set[int]]:
        """构建卫星与地面站的连接（基于仰角和可见性）"""
        ground_neighbors = {i: set() for i in range(self.num_satellites + self.num_ground_stations)}

        for gs_id, gs_pos in enumerate(self.gs_positions):
            gs_ground_idx = self.num_satellites + gs_id  # 地面站在邻居表中的索引
            for sat_id, sat_pos in enumerate(self.sat_positions):
                # 计算卫星相对于地面站的仰角
                dx = sat_pos[0] - gs_pos[0]
                dy = sat_pos[1] - gs_pos[1]
                dz = sat_pos[2] - gs_pos[2]
                distance = np.linalg.norm([dx, dy, dz])
                if distance == 0:
                    continue

                # 仰角公式：elevation = arcsin(dz / distance)
                elevation = np.arcsin(dz / distance)
                if elevation >= self.minimum_elevation:
                    ground_neighbors[sat_id].add(gs_ground_idx)
                    ground_neighbors[gs_ground_idx].add(sat_id)

        return ground_neighbors

    def _build_neighbors(self) -> Dict[int, List[int]]:
        """整合卫星间和卫星-地面站连接，返回邻居关系"""
        # 初始化邻居表
        neighbors = {i: set() for i in range(self.num_satellites + self.num_ground_stations)}

        # 添加卫星间 ISL
        sat_isl = self._build_satellite_isl()
        for sat_id, peers in sat_isl.items():
            neighbors[sat_id].update(peers)

        # 添加卫星-地面站连接
        sat_gs_links = self._build_satellite_ground_links()
        for node_id, peers in sat_gs_links.items():
            neighbors[node_id].update(peers)

        # 转换为排序后的列表
        return {k: sorted(list(v)) for k, v in neighbors.items()}

if __name__ == "__main__":
    # 假设卫星位置和地面站位置为三维坐标数组（单位：km）
    sat_positions = [np.array([x, y, z]) for x, y, z in satellite_coordinates]
    gs_positions = [np.array([lon, lat, 0]) for lon, lat in ground_station_coordinates]

    # 初始化拓扑构建器（参数需与 StarPerf 配置一致）
    topology_builder = StarPerfTopologyBuilder(
        sat_positions=sat_positions,
        gs_positions=gs_positions,
        num_orbits=72,  # 轨道数（如 Starlink 的 XML 配置）
        num_sat_per_orbit=22,  # 每条轨道卫星数
        inclination=53.0  # 轨道倾角
    )

    # 构建邻居关系
    neighbors = topology_builder._build_neighbors()