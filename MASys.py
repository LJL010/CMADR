# MASys.py - 修复概率问题

import torch
import torch.nn as nn
from torch.nn import functional as F
from AC import ActorNet, CriticNet
import numpy as np


class MultiAgentSystem:
    def __init__(self, n_agents, n_nodes, obs_dim, action_dim, hidden_dim, device='cpu'):
        self.n_agents = n_agents
        self.device = device
        self.actors = [ActorNet(obs_dim, hidden_dim, action_dim).to(device) for _ in range(n_agents)]
        self.critics = [CriticNet(obs_dim, hidden_dim).to(device) for _ in range(n_agents)]
        self.global_critic = CriticNet(obs_dim * n_nodes, hidden_dim).to(device)

        self.optim_actors = [torch.optim.Adam(actor.parameters(), lr=1e-3) for actor in self.actors]
        self.optim_critics = [torch.optim.Adam(critic.parameters(), lr=1e-3) for critic in self.critics]
        self.optim_global_critic = torch.optim.Adam(self.global_critic.parameters(), lr=1e-3)

    def select_actions(self, obs_n, neighbors=None):
        """
        修复概率问题的动作选择函数
        """
        actions = []

        for i in range(self.n_agents):
            obs = torch.tensor(obs_n[i], dtype=torch.float32, device=self.device).unsqueeze(0)

            # 获取原始概率
            probs = self.actors[i](obs)

            if neighbors and i in neighbors:
                current_neighbors = neighbors[i]
                num_neighbors = len(current_neighbors)

                if num_neighbors > 0:
                    # === 问题1：限制动作空间到实际邻居数量 ===
                    valid_probs = probs[0][:num_neighbors]

                    # === 问题2：处理概率为0或NaN的情况 ===
                    # 检查是否有有效概率
                    if torch.isnan(valid_probs).any() or torch.isinf(valid_probs).any():
                        print(f"Agent {i}: 检测到NaN或Inf概率，使用均匀分布")
                        valid_probs = torch.ones(num_neighbors, device=self.device) / num_neighbors
                    elif valid_probs.sum() <= 1e-10:
                        print(f"Agent {i}: 概率和接近0，使用均匀分布")
                        valid_probs = torch.ones(num_neighbors, device=self.device) / num_neighbors
                    else:
                        # === 问题3：数值稳定性 ===
                        # 确保概率最小值，避免数值下溢
                        valid_probs = torch.clamp(valid_probs, min=1e-8)
                        # 重新归一化
                        valid_probs = valid_probs / valid_probs.sum()

                    # === 问题4：安全的采样 ===
                    try:
                        m = torch.distributions.Categorical(valid_probs)
                        action = m.sample().item()

                        # 额外检查：确保动作在有效范围内
                        if action >= num_neighbors:
                            action = num_neighbors - 1
                            print(f"Agent {i}: 采样动作超出范围，调整为 {action}")

                    except Exception as e:
                        print(f"Agent {i}: 采样失败 {e}，使用随机动作")
                        action = np.random.randint(0, num_neighbors)

                else:
                    # 没有邻居的情况
                    action = 0
                    #print(f"Agent {i}: 没有邻居，使用默认动作")
            else:
                # === 问题5：没有邻居信息时的处理 ===
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    print(f"Agent {i}: 检测到NaN或Inf概率（无邻居信息），使用随机动作")
                    action = np.random.randint(0, probs.shape[1])
                elif probs[0].sum() <= 1e-10:
                    print(f"Agent {i}: 概率和接近0（无邻居信息），使用随机动作")
                    action = np.random.randint(0, probs.shape[1])
                else:
                    # 正常采样
                    probs_clamped = torch.clamp(probs[0], min=1e-8)
                    probs_clamped = probs_clamped / probs_clamped.sum()

                    try:
                        m = torch.distributions.Categorical(probs_clamped)
                        action = m.sample().item()
                    except Exception as e:
                        print(f"Agent {i}: 采样失败 {e}，使用随机动作")
                        action = np.random.randint(0, probs.shape[1])

            actions.append(action)

        return actions

    def select_actions_with_exploration(self, obs_n, neighbors=None, epsilon=0.1):
        """
        带探索的动作选择（训练时使用）
        """
        actions = []

        for i in range(self.n_agents):
            # === 问题6：添加探索机制 ===
            if np.random.random() < epsilon:
                # 探索：随机选择动作
                if neighbors and i in neighbors:
                    current_neighbors = neighbors[i]
                    if current_neighbors:
                        action = np.random.randint(0, len(current_neighbors))
                    else:
                        action = 0
                else:
                    action = np.random.randint(0, self.actors[i].fc2.out_features)
                actions.append(action)
            else:
                # 利用：使用策略网络
                obs = torch.tensor(obs_n[i], dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():  # 推理时不需要梯度
                    probs = self.actors[i](obs)

                if neighbors and i in neighbors:
                    current_neighbors = neighbors[i]
                    num_neighbors = len(current_neighbors)

                    if num_neighbors > 0:
                        valid_probs = probs[0][:num_neighbors]

                        # 数值稳定性处理
                        if torch.isnan(valid_probs).any() or valid_probs.sum() <= 1e-10:
                            valid_probs = torch.ones(num_neighbors, device=self.device) / num_neighbors
                        else:
                            valid_probs = torch.clamp(valid_probs, min=1e-8)
                            valid_probs = valid_probs / valid_probs.sum()

                        try:
                            m = torch.distributions.Categorical(valid_probs)
                            action = m.sample().item()
                        except:
                            action = np.random.randint(0, num_neighbors)
                    else:
                        action = 0
                else:
                    # 没有邻居信息时的处理
                    probs_clamped = torch.clamp(probs[0], min=1e-8)
                    probs_clamped = probs_clamped / probs_clamped.sum()

                    try:
                        m = torch.distributions.Categorical(probs_clamped)
                        action = m.sample().item()
                    except:
                        action = np.random.randint(0, probs.shape[1])

                actions.append(action)

        return actions

    # 其他方法保持不变...