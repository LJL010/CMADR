# MASys.py - 修复概率问题并添加save/load方法

import torch
import torch.nn as nn
from torch.nn import functional as F
from AC import ActorNet, CriticNet
import numpy as np
import os


class MultiAgentSystem:
    def __init__(self, n_agents, n_nodes, obs_dim, action_dim, hidden_dim, device='cpu'):
        self.n_agents = n_agents
        self.n_nodes = n_nodes
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
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
                    # print(f"Agent {i}: 没有邻居，使用默认动作")
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
                    action = np.random.randint(0, self.action_dim)
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

    def save(self, model_dir):
        """
        保存所有模型参数和优化器状态
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # 保存模型配置信息
        config = {
            'n_agents': self.n_agents,
            'n_nodes': self.n_nodes,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'device': str(self.device)
        }

        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'w') as f:
            import json
            json.dump(config, f, indent=2)

        # 保存每个actor网络
        for i, actor in enumerate(self.actors):
            actor_path = os.path.join(model_dir, f'actor_{i}.pth')
            torch.save(actor.state_dict(), actor_path)

        # 保存每个critic网络
        for i, critic in enumerate(self.critics):
            critic_path = os.path.join(model_dir, f'critic_{i}.pth')
            torch.save(critic.state_dict(), critic_path)

        # 保存全局critic网络
        global_critic_path = os.path.join(model_dir, 'global_critic.pth')
        torch.save(self.global_critic.state_dict(), global_critic_path)

        # 保存优化器状态
        optimizer_states = {
            'optim_actors': [opt.state_dict() for opt in self.optim_actors],
            'optim_critics': [opt.state_dict() for opt in self.optim_critics],
            'optim_global_critic': self.optim_global_critic.state_dict()
        }
        optimizer_path = os.path.join(model_dir, 'optimizers.pth')
        torch.save(optimizer_states, optimizer_path)

        print(f"模型已保存到: {model_dir}")

    def load(self, model_dir):
        """
        加载所有模型参数和优化器状态
        """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")

        # 检查配置文件
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                import json
                saved_config = json.load(f)

            # 验证配置是否匹配
            if (saved_config['n_agents'] != self.n_agents or
                    saved_config['obs_dim'] != self.obs_dim or
                    saved_config['action_dim'] != self.action_dim):
                print("警告: 保存的模型配置与当前配置不匹配！")
                print(f"保存的配置: {saved_config}")
                print(f"当前配置: n_agents={self.n_agents}, obs_dim={self.obs_dim}, action_dim={self.action_dim}")

        # 加载每个actor网络
        for i, actor in enumerate(self.actors):
            actor_path = os.path.join(model_dir, f'actor_{i}.pth')
            if os.path.exists(actor_path):
                actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            else:
                print(f"警告: 找不到actor_{i}.pth")

        # 加载每个critic网络
        for i, critic in enumerate(self.critics):
            critic_path = os.path.join(model_dir, f'critic_{i}.pth')
            if os.path.exists(critic_path):
                critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            else:
                print(f"警告: 找不到critic_{i}.pth")

        # 加载全局critic网络
        global_critic_path = os.path.join(model_dir, 'global_critic.pth')
        if os.path.exists(global_critic_path):
            self.global_critic.load_state_dict(torch.load(global_critic_path, map_location=self.device))
        else:
            print("警告: 找不到global_critic.pth")

        # 加载优化器状态（可选）
        optimizer_path = os.path.join(model_dir, 'optimizers.pth')
        if os.path.exists(optimizer_path):
            try:
                optimizer_states = torch.load(optimizer_path, map_location=self.device)

                # 加载actor优化器状态
                for i, opt_state in enumerate(optimizer_states['optim_actors']):
                    self.optim_actors[i].load_state_dict(opt_state)

                # 加载critic优化器状态
                for i, opt_state in enumerate(optimizer_states['optim_critics']):
                    self.optim_critics[i].load_state_dict(opt_state)

                # 加载全局critic优化器状态
                self.optim_global_critic.load_state_dict(optimizer_states['optim_global_critic'])

            except Exception as e:
                print(f"加载优化器状态时出错: {e}")
        else:
            print("警告: 找不到optimizers.pth，将使用默认优化器状态")

        # 确保所有模型都在正确的设备上
        for actor in self.actors:
            actor.to(self.device)
        for critic in self.critics:
            critic.to(self.device)
        self.global_critic.to(self.device)

        print(f"模型已从 {model_dir} 加载成功")

    def set_train_mode(self):
        """设置所有网络为训练模式"""
        for actor in self.actors:
            actor.train()
        for critic in self.critics:
            critic.train()
        self.global_critic.train()

    def set_eval_mode(self):
        """设置所有网络为评估模式"""
        for actor in self.actors:
            actor.eval()
        for critic in self.critics:
            critic.eval()
        self.global_critic.eval()

    def get_model_info(self):
        """获取模型信息"""
        total_params = 0

        for i, actor in enumerate(self.actors):
            actor_params = sum(p.numel() for p in actor.parameters())
            total_params += actor_params

        for i, critic in enumerate(self.critics):
            critic_params = sum(p.numel() for p in critic.parameters())
            total_params += critic_params

        global_critic_params = sum(p.numel() for p in self.global_critic.parameters())
        total_params += global_critic_params

        return {
            'total_parameters': total_params,
            'n_agents': self.n_agents,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'device': str(self.device)
        }