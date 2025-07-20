import numpy as np
import random
from math import radians, cos, sin, asin, sqrt
import os
import json

class ISTNEnv:
    def __init__(
            self,
            num_satellites,
            num_ground_stations,
            max_buffer=100,
            max_energy=1.0,
            max_time=478,
            seed=0,
            sat_positions=None,
            gs_positions=None,
            queries=None,
            sat_positions_per_slot=None,
            conn_threshold_min=500,
            conn_threshold_max=2000,
            time_slot=0,
    ):
        """ISTN 环境

        当 ``sat_positions`` 或 ``gs_positions`` 提供时，将使用给定的位置数据，否
        则随机生成。 ``queries`` 用于在 ``reset`` 时初始化地面站 buffer，方便在训
        练和预测阶段保持一致的数据输入。
        """
        self.time_slot = time_slot
        self.num_satellites = num_satellites
        self.num_ground_stations = num_ground_stations
        self.max_buffer = max_buffer
        self.max_energy = max_energy
        self.max_time = max_time
        self.random = random.Random(seed)
        self.n_agents = num_satellites + num_ground_stations

        self.sat_positions_per_slot = sat_positions_per_slot

        if sat_positions_per_slot is not None:
            self.sat_positions = [np.array(p) for p in sat_positions_per_slot[0]]
        else:
            self.sat_positions = sat_positions or [
                np.array([random.uniform(0, 100), random.uniform(0, 100)])
                for _ in range(self.num_satellites)
            ]
        self.gs_positions = gs_positions or [
            np.array([random.uniform(0, 100), random.uniform(0, 100)])
            for _ in range(self.num_ground_stations)
        ]

        self.conn_threshold_min = conn_threshold_min
        self.conn_threshold_max = conn_threshold_max
        # sort queries by time slot; each query should have {'src','dst','time'}
        self.queries = sorted(queries or [], key=lambda q: q.get('time', 0))
        self.query_index = 0  # pointer to next query to release

        # 邻接表在每个time slot动态传入
        self.neighbors = None

        # === 修正：重新计算obs_dim以匹配新的get_obs函数 ===
        # 卫星观测维度：自身(3) + 4个邻居(3*4) + 到目的地距离(1) = 16
        satellite_obs_dim = 3 + 3 * 4 + 1  # 16维

        # 地面站观测维度：自身(1) + 所有卫星信息(2*num_satellites) + 到目的地距离(1)
        gs_obs_dim = 1 + 2 * num_satellites + 1

        # 取最大值作为统一的观测维度（为了网络兼容性）
        self.obs_dim = max(satellite_obs_dim, gs_obs_dim)

        print(f"观测维度计算:")
        print(f"- 卫星观测维度: {satellite_obs_dim}")
        print(f"- 地面站观测维度: {gs_obs_dim}")
        print(f"- 统一观测维度: {self.obs_dim}")

        # 动作维度：最大可能的邻居数
        # 卫星：4个ISL邻居 + 可能的地面站连接
        # 地面站：可能连接的卫星数
        max_satellite_actions = 4 + num_ground_stations
        max_gs_actions = num_satellites
        self.action_dim = max(max_satellite_actions, max_gs_actions)

    # 1.把真实的数据搞下来
    # 2.用n_nearest.py中的函数替换掉下面的函数
    # def _build_neighbors(self):
    #     """Build neighbors based on current node positions."""
    #     neighbors = {i: set() for i in range(self.n_agents)}
    #     # satellite-satellite links
    #     isl_num = 4
    #     for i in range(self.num_satellites):
    #         for j in range(self.num_satellites):
    #             if len(neighbors[i]) >= isl_num:
    #                 break
    #             if i == j:
    #                 continue
    #             dist = self.distance_two_satellites(self.sat_positions[i], self.sat_positions[j])
    #             if self.conn_threshold_min <= dist <= self.conn_threshold_max:
    #                 neighbors[i].add(j)

    #     # satellite-ground links
    #     for gs in range(self.num_ground_stations):
    #         gs_pos = self.gs_positions[gs]
    #         for sat in range(self.num_satellites):
    #             dist = self.distance_two_satellites(self.sat_positions[sat], gs_pos)
    #             if dist <= 700:
    #                 neighbors[sat].add(self.num_satellites + gs)
    #                 neighbors[self.num_satellites + gs].add(sat)
    #     # convert sets to sorted lists
    #     return {k: sorted(list(v)) for k, v in neighbors.items()}

    def _build_neighbors(self):
        """Build neighbors based on current node positions."""
        filename = f"neighbors_data/neighbors_slot_{self.time_slot}.json"
        try:
            # 从文件读取缓存数据
            with open(filename, 'r') as f:
                neighbors = json.load(f)
            # 将列表转换回集合（如果需要）
            integer_key_dict = {int(k): v for k, v in neighbors.items()}
            return integer_key_dict
        except Exception as e:
            print(f"读取缓存失败 (slot={self.time_slot}): {e}")


    def distance_two_satellites(self, satellite1, satellite2):
        longitude1 = satellite1[0]
        latitude1 = satellite1[1]
        longitude2 = satellite2[0]
        latitude2 = satellite2[1]
        # The altitude is the average altitude of the two satellites, in kilometers
        altitude = 1.0 * (satellite1[2] + satellite2[2]) / 2
        longitude1, latitude1, longitude2, latitude2 = map(radians,
                                                           [float(longitude1), float(latitude1), float(longitude2),
                                                            float(latitude2)])  # 经纬度转换成弧度
        dlon = longitude2 - longitude1
        dlat = latitude2 - latitude1
        a = sin(dlat / 2) ** 2 + cos(latitude1) * cos(latitude2) * sin(dlon / 2) ** 2
        # The average radius of the earth is 6371km, and the satellite orbit altitude is 6371km.
        distance = 2 * asin(sqrt(a)) * (6371.0 + altitude) * 1000
        # Convert the result to kilometers with three decimal places.
        distance = np.round(distance / 1000, 3)
        return distance

    def initialize_satellites(self):
        satellites = {}
        for i in range(self.num_satellites):
            satellites[i] = {
                'energy': self.random.uniform(0.5, self.max_energy),
                'buffer': [],  # buffer现在存储包对象
                'latency': self.random.uniform(1, 10)
            }
        return satellites

    def initialize_ground_stations(self):
        ground_stations = {}
        for i in range(self.num_ground_stations):
            ground_stations[i] = {
                'energy': self.random.uniform(0.5, self.max_energy),
                'buffer': [],
                'latency': self.random.uniform(1, 10)
            }
        return ground_stations

    def _create_packet(self):
        # 创建包：带目的地（地面站ID），初始来源地面站随机
        src = self.random.randint(0, self.num_ground_stations - 1)
        dst = self.random.randint(0, self.num_ground_stations - 1)
        while dst == src:
            dst = self.random.randint(0, self.num_ground_stations - 1)
        return {'dst': dst, 'hop': 0, 'src': src, 'path': [], 'start_time': self.time_slot}

    def _create_packet_from_query(self, src, dst):
        """根据给定的查询生成数据包"""
        return {
            'dst': dst,
            'hop': 0,
            'src': src,
            'path': [],
            'start_time': self.time_slot,
        }

    def _release_queries(self):
        """Release queries scheduled for the current time slot."""
        while self.query_index < len(self.queries) and \
                self.queries[self.query_index].get('time', 0) == self.time_slot:
            q = self.queries[self.query_index]
            gs = self.ground_stations[q['src']]
            if len(gs['buffer']) < self.max_buffer:
                pkt = self._create_packet_from_query(q['src'], q['dst'])
                gs['buffer'].append(pkt)
            self.query_index += 1

    def get_obs(self, neighbors):
        """
        修正版本：确保所有agent的观测维度完全一致
        """
        obs = []

        for i in range(self.n_agents):
            if i < self.num_satellites:
                # === 卫星观测 ===
                sat = self.satellites[i]
                state = [sat['energy'], len(sat['buffer']), sat['latency']]

                # 四个邻居的状态（固定4个）
                current_neighbors = neighbors.get(i, [])
                for j in range(4):
                    if j < len(current_neighbors):
                        nb = current_neighbors[j]
                        if nb < self.num_satellites:
                            nb_node = self.satellites[nb]
                            state.extend([nb_node['energy'], len(nb_node['buffer']), nb_node['latency']])
                        else:
                            nb_node = self.ground_stations[nb - self.num_satellites]
                            state.extend([nb_node['energy'], len(nb_node['buffer']), nb_node['latency']])
                    else:
                        state.extend([0.0, 0.0, 0.0])

                # 到目的地的距离
                buf = sat['buffer']
                if buf:
                    dst_gs = buf[0]['dst']
                    sat_pos = self.sat_positions[i]
                    dst_pos = self.gs_positions[dst_gs]
                    dist = np.linalg.norm(np.array(sat_pos) - np.array(dst_pos))
                else:
                    dist = 0.0
                state.append(dist)

            else:
                # === 地面站观测 ===
                gs_idx = i - self.num_satellites
                gs = self.ground_stations[gs_idx]
                state = [gs['energy']]

                # 所有卫星的状态（无论是否连接）
                current_neighbors = neighbors.get(i, [])
                connected_sats = set(nb for nb in current_neighbors if nb < self.num_satellites)

                for sat_idx in range(self.num_satellites):
                    if sat_idx in connected_sats:
                        sat_node = self.satellites[sat_idx]
                        state.extend([sat_node['energy'], len(sat_node['buffer'])])
                    else:
                        state.extend([0.0, 0.0])

                # 到目的地的距离
                buf = gs['buffer']
                if buf:
                    dst_gs = buf[0]['dst']
                    gs_pos = self.gs_positions[gs_idx]
                    dst_pos = self.gs_positions[dst_gs]
                    dist = np.linalg.norm(np.array(gs_pos) - np.array(dst_pos))
                else:
                    dist = 0.0
                state.append(dist)

            # === 关键修正：确保所有观测维度完全一致 ===
            # 填充或截断到统一维度
            while len(state) < self.obs_dim:
                state.append(0.0)
            if len(state) > self.obs_dim:
                state = state[:self.obs_dim]

            obs.append(np.array(state, dtype=np.float32))

        return obs

    def step(self, actions, neighbors):
        """
        修正版本：严格按照论文定义计算奖励和成本
        """
        # === 按照论文定义初始化成本 ===
        cost_energy = np.zeros(self.n_agents)  # 每个agent的能耗
        cost_loss = 0  # 全局丢包数量
        delivered_packets = []
        transit_packets = []
        rewards = np.zeros(self.n_agents)

        for idx, action_idx in enumerate(actions):
            # 确定当前节点
            if idx < self.num_satellites:
                node = self.satellites[idx]
                current_pos = self.sat_positions[idx]
            else:
                gs_idx = idx - self.num_satellites
                if gs_idx < len(self.ground_stations):
                    node = self.ground_stations[gs_idx]
                    current_pos = self.gs_positions[gs_idx]
                else:
                    print(f"Error: Invalid agent index {idx}")
                    continue

            if node['buffer']:
                pkt = node['buffer'][0]
                dst_gs = pkt['dst']
                dst_pos = self.gs_positions[dst_gs]

                # 检查动作有效性
                current_neighbors = neighbors.get(idx, [])
                if not current_neighbors:
                    continue

                # 确保动作索引在有效范围内
                if action_idx >= len(current_neighbors):
                    target_neighbor = current_neighbors[-1]
                else:
                    target_neighbor = current_neighbors[action_idx]

                # 确定目标节点
                if target_neighbor < self.num_satellites:
                    tgt_node = self.satellites[target_neighbor]
                    target_pos = self.sat_positions[target_neighbor]
                else:
                    tgt_gs_idx = target_neighbor - self.num_satellites
                    if tgt_gs_idx < len(self.ground_stations):
                        tgt_node = self.ground_stations[tgt_gs_idx]
                        target_pos = self.gs_positions[tgt_gs_idx]
                    else:
                        continue

                # === 按照论文计算传输速率奖励 ===
                # 计算包向目的地移动的距离
                current_to_dst = np.linalg.norm(np.array(current_pos) - np.array(dst_pos))
                target_to_dst = np.linalg.norm(np.array(target_pos) - np.array(dst_pos))

                # 传输速率 = 距离改善 / 时间单位 (假设每个时隙为1个时间单位)
                distance_improvement = max(0, current_to_dst - target_to_dst)
                transmission_rate = distance_improvement  # 论文中的平均传输速率

                # 缓冲是否满
                if len(tgt_node['buffer']) < self.max_buffer:
                    # 成功转发
                    pkt = node['buffer'].pop(0)
                    pkt['hop'] += 1
                    pkt['path'].append(target_neighbor)
                    tgt_node['buffer'].append(pkt)

                    # 若目标是地面站且正好为目的地，则交付
                    if (target_neighbor >= self.num_satellites) and ((target_neighbor - self.num_satellites) == dst_gs):
                        delivered_packets.append(pkt)
                        tgt_node['buffer'].pop()  # 交付出队
                        # 成功交付额外奖励
                        rewards[idx] += transmission_rate + 10.0  # 基础传输速率 + 交付奖励
                    else:
                        # 转发奖励
                        rewards[idx] += transmission_rate
                        transit_packets.append(pkt)

                    # === 按照论文计算能耗成本 ===
                    node['energy'] -= 0.01
                    cost_energy[idx] += 0.01
                else:
                    # === 按照论文计算丢包成本 ===
                    node['buffer'].pop(0)
                    cost_loss += 1  # 论文中的丢包计数
                    # 丢包惩罚
                    rewards[idx] -= 5.0

            else:
                # 空闲时轻微惩罚
                rewards[idx] -= 0.01

        # === 论文中的全局奖励成分 ===
        # 计算所有包的平均传输速率
        if delivered_packets or transit_packets:
            # 全局传输效率奖励
            global_efficiency = len(delivered_packets) * 2.0
            rewards += global_efficiency / self.n_agents

        self.time_slot += 1
        self._release_queries()

        # 更新卫星位置
        if self.sat_positions_per_slot is not None:
            slot_idx = min(self.time_slot, len(self.sat_positions_per_slot) - 1)
            self.sat_positions = [np.array(p) for p in self.sat_positions_per_slot[slot_idx]]

        done = self.time_slot >= self.max_time

        obs = self.get_obs(neighbors)

        # === 按照论文定义的信息统计 ===
        info = {
            'delivered_packets': len(delivered_packets),
            'packets_in_transit': len(transit_packets),
            'total_cost_loss': cost_loss,
            'delays': [120 * (self.time_slot - pkt['start_time']) for pkt in delivered_packets],
            'avg_hops': np.mean([pkt['hop'] for pkt in delivered_packets]) if delivered_packets else 0,
            'avg_delay': np.mean(
                [self.time_slot - pkt['start_time'] for pkt in delivered_packets]) if delivered_packets else 0,
        }

        # === 按照论文定义返回成本 ===
        costs = {
            'energy': cost_energy,  # 每个agent的能耗成本
            'loss': cost_loss  # 全局丢包成本
        }

        return obs, rewards, done, costs, info


    def reset(self):
        self.satellites = self.initialize_satellites()
        self.ground_stations = self.initialize_ground_stations()
        self.time_slot = 0
        self.query_index = 0

        if self.sat_positions_per_slot is not None:
            self.sat_positions = [np.array(p) for p in self.sat_positions_per_slot[0]]

        self.neighbors = self._build_neighbors()
        
        # 初始化buffer：根据当前time_slot释放查询或随机生成
        self._release_queries()
        if not self.queries:
            for i in range(self.num_ground_stations):
                pkt = self._create_packet()
                self.ground_stations[i]['buffer'].append(pkt)

        return self.get_obs(self.neighbors)
