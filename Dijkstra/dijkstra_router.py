import heapq
import numpy as np
from typing import Dict, List, Tuple, Optional


class DijkstraRouter:
    """
    基于Dijkstra算法的路由器，用作强化学习路由的baseline
    """

    def __init__(self, num_satellites: int, num_ground_stations: int):
        self.num_satellites = num_satellites
        self.num_ground_stations = num_ground_stations
        self.n_nodes = num_satellites + num_ground_stations

    def _calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """计算两个位置之间的距离"""
        if len(pos1) == 3 and len(pos2) == 3:
            # 如果是3D坐标，使用球面距离公式
            return self._haversine_distance(pos1, pos2)
        else:
            # 如果是2D坐标，使用欧几里得距离
            return np.linalg.norm(pos1 - pos2)

    def _haversine_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """计算球面距离（用于3D坐标）"""
        from math import radians, cos, sin, asin, sqrt

        lon1, lat1, alt1 = pos1
        lon2, lat2, alt2 = pos2

        # 转换为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

        # 平均高度
        avg_alt = (alt1 + alt2) / 2
        # 地球半径 + 卫星高度
        r = 6371.0 + avg_alt

        distance = 2 * asin(sqrt(a)) * r * 1000  # 转换为米
        return distance / 1000  # 转换为公里

    def _build_graph(self, sat_positions: List[np.ndarray],
                     gs_positions: List[np.ndarray],
                     neighbors: Dict[int, List[int]]) -> Dict[int, List[Tuple[int, float]]]:
        """
        构建带权重的图
        返回格式: {node_id: [(neighbor_id, weight), ...]}
        """
        graph = {i: [] for i in range(self.n_nodes)}

        for node_id, neighbor_list in neighbors.items():
            if node_id < self.num_satellites:
                current_pos = sat_positions[node_id]
            else:
                current_pos = gs_positions[node_id - self.num_satellites]

            for neighbor_id in neighbor_list:
                if neighbor_id < self.num_satellites:
                    neighbor_pos = sat_positions[neighbor_id]
                else:
                    neighbor_pos = gs_positions[neighbor_id - self.num_satellites]

                # 计算权重（可以是距离、延迟、能耗等的组合）
                distance = self._calculate_distance(current_pos, neighbor_pos)

                # 简单的权重计算：距离 + 固定传输成本
                weight = distance + 10.0  # 10.0是固定的传输成本

                graph[node_id].append((neighbor_id, weight))

        return graph

    def dijkstra(self, graph: Dict[int, List[Tuple[int, float]]],
                 start: int, end: int) -> Tuple[List[int], float]:
        """
        Dijkstra算法寻找最短路径
        返回: (路径, 总距离)
        """
        # 初始化距离和前驱节点
        distances = {node: float('inf') for node in range(self.n_nodes)}
        distances[start] = 0
        previous = {node: None for node in range(self.n_nodes)}

        # 优先队列：(距离, 节点)
        pq = [(0, start)]
        visited = set()

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)

            # 如果到达目标节点，提前结束
            if current_node == end:
                break

            # 更新邻居节点的距离
            for neighbor, weight in graph.get(current_node, []):
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current_node
                        heapq.heappush(pq, (new_dist, neighbor))

        # 重建路径
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]

        if path[-1] != start:
            return [], float('inf')  # 无法到达

        path.reverse()
        return path, distances[end]

    def get_next_hop(self, current_node: int, destination: int,
                     sat_positions: List[np.ndarray],
                     gs_positions: List[np.ndarray],
                     neighbors: Dict[int, List[int]]) -> Optional[int]:
        """
        获取从当前节点到目的地的下一跳节点
        """
        # 目的地是地面站ID，需要转换为节点ID
        dest_node = self.num_satellites + destination

        # 构建图
        graph = self._build_graph(sat_positions, gs_positions, neighbors)

        # 计算最短路径
        path, total_dist = self.dijkstra(graph, current_node, dest_node)

        if len(path) <= 1:
            return None  # 无法到达或已经在目的地

        # 返回下一跳节点
        next_hop = path[1]

        # 验证下一跳是否在邻居列表中
        current_neighbors = neighbors.get(current_node, [])
        if next_hop in current_neighbors:
            return next_hop
        else:
            # 如果计算出的下一跳不在邻居中，选择最近的邻居
            if not current_neighbors:
                return None

            # 选择离目的地最近的邻居
            if current_node < self.num_satellites:
                current_pos = sat_positions[current_node]
            else:
                current_pos = gs_positions[current_node - self.num_satellites]

            dest_pos = gs_positions[destination]

            best_neighbor = None
            best_dist = float('inf')

            for neighbor in current_neighbors:
                if neighbor < self.num_satellites:
                    neighbor_pos = sat_positions[neighbor]
                else:
                    neighbor_pos = gs_positions[neighbor - self.num_satellites]

                dist_to_dest = self._calculate_distance(neighbor_pos, dest_pos)
                if dist_to_dest < best_dist:
                    best_dist = dist_to_dest
                    best_neighbor = neighbor

            return best_neighbor

    def select_actions(self, env, neighbors: Dict[int, List[int]]) -> List[int]:
        """
        为所有节点选择动作（类似于MultiAgentSystem中的select_actions）
        """
        actions = []

        for node_id in range(self.n_nodes):
            # 获取当前节点的buffer
            if node_id < self.num_satellites:
                node = env.satellites[node_id]
            else:
                node = env.ground_stations[node_id - self.num_satellites]

            if node['buffer']:
                # 取buffer中的第一个包
                packet = node['buffer'][0]
                destination = packet['dst']

                # 计算下一跳
                next_hop = self.get_next_hop(
                    node_id, destination,
                    env.sat_positions, env.gs_positions,
                    neighbors
                )

                if next_hop is not None:
                    # 找到next_hop在邻居列表中的索引
                    current_neighbors = neighbors.get(node_id, [])
                    if next_hop in current_neighbors:
                        action = current_neighbors.index(next_hop)
                    else:
                        action = 0  # 默认选择第一个邻居
                else:
                    action = 0  # 没有找到下一跳，选择第一个邻居
            else:
                action = 0  # 没有包要转发

            actions.append(action)

        return actions