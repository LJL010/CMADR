# MASys.py - 内存优化版本：保持原逻辑，优化内存使用

import torch
import torch.nn as nn
from torch.nn import functional as F
from AC import ActorNet, CriticNet
import numpy as np
import os
import json
import gc


class MultiAgentSystem:
    def __init__(self, n_agents, n_nodes, obs_dim, action_dim, hidden_dim, device='cpu'):
        self.n_agents = n_agents
        self.n_nodes = n_nodes
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Actor和Local Critic网络（每个agent一个）
        self.actors = [ActorNet(obs_dim, hidden_dim, action_dim).to(device) for _ in range(n_agents)]
        self.critics = [CriticNet(obs_dim, hidden_dim).to(device) for _ in range(n_agents)]

        # === 分离的全局critic网络 ===
        self.global_reward_critic = CriticNet(obs_dim * n_nodes, hidden_dim).to(device)
        self.global_cost_critic = CriticNet(obs_dim * n_nodes, hidden_dim).to(device)

        # 优化器
        self.optim_actors = [torch.optim.Adam(actor.parameters(), lr=1e-3) for actor in self.actors]
        self.optim_critics = [torch.optim.Adam(critic.parameters(), lr=1e-3) for critic in self.critics]

        # === 分别为两个global critic创建优化器 ===
        self.optim_global_reward_critic = torch.optim.Adam(self.global_reward_critic.parameters(), lr=1e-3)
        self.optim_global_cost_critic = torch.optim.Adam(self.global_cost_critic.parameters(), lr=1e-3)

    def select_actions(self, obs_n, neighbors=None):
        """
        内存优化的动作选择函数 - 保持原逻辑
        """
        actions = []

        for i in range(self.n_agents):
            # 内存优化：使用numpy直接转换，避免多次tensor创建
            obs_tensor = torch.from_numpy(obs_n[i].astype(np.float32)).unsqueeze(0).to(self.device)

            # 使用no_grad减少内存占用
            with torch.no_grad():
                # 获取原始概率
                probs = self.actors[i](obs_tensor)

                if neighbors and i in neighbors:
                    current_neighbors = neighbors[i]
                    num_neighbors = len(current_neighbors)

                    if num_neighbors > 0:
                        # 限制动作空间到实际邻居数量
                        valid_probs = probs[0][:num_neighbors]

                        # 处理概率为0或NaN的情况
                        if torch.isnan(valid_probs).any() or torch.isinf(valid_probs).any():
                            # 内存优化：直接使用numpy创建
                            action = np.random.randint(0, num_neighbors)
                        elif valid_probs.sum() <= 1e-10:
                            action = np.random.randint(0, num_neighbors)
                        else:
                            # 数值稳定性
                            valid_probs = torch.clamp(valid_probs, min=1e-8)
                            valid_probs = valid_probs / valid_probs.sum()

                            # 安全的采样
                            try:
                                m = torch.distributions.Categorical(valid_probs)
                                action = m.sample().item()

                                if action >= num_neighbors:
                                    action = num_neighbors - 1
                            except Exception:
                                action = np.random.randint(0, num_neighbors)
                    else:
                        action = 0
                else:
                    # 没有邻居信息时的处理
                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        action = np.random.randint(0, probs.shape[1])
                    elif probs[0].sum() <= 1e-10:
                        action = np.random.randint(0, probs.shape[1])
                    else:
                        probs_clamped = torch.clamp(probs[0], min=1e-8)
                        probs_clamped = probs_clamped / probs_clamped.sum()

                        try:
                            m = torch.distributions.Categorical(probs_clamped)
                            action = m.sample().item()
                        except Exception:
                            action = np.random.randint(0, probs.shape[1])

                actions.append(action)

            # 内存优化：立即删除tensor
            del obs_tensor

        return actions

    def get_model_info(self):
        """获取模型信息 - 用于调试和保存"""
        try:
            actor_params = sum(p.numel() for p in self.actors[0].parameters())
            critic_params = sum(p.numel() for p in self.critics[0].parameters())
            global_reward_params = sum(p.numel() for p in self.global_reward_critic.parameters())
            global_cost_params = sum(p.numel() for p in self.global_cost_critic.parameters())

            total_params = (actor_params + critic_params) * self.n_agents + global_reward_params + global_cost_params

            return {
                'n_agents': self.n_agents,
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'device': str(self.device),
                'total_parameters': total_params
            }
        except Exception as e:
            print(f"获取模型信息时出错: {e}")
            return {
                'n_agents': self.n_agents,
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'device': str(self.device),
                'total_parameters': 0
            }

    def save(self, model_dir):
        """保存所有模型参数"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        try:
            # 保存模型配置信息
            config = self.get_model_info()
            config_path = os.path.join(model_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # 保存每个actor网络
            for i, actor in enumerate(self.actors):
                actor_path = os.path.join(model_dir, f'actor_{i}.pth')
                torch.save(actor.state_dict(), actor_path)

            # 保存每个critic网络
            for i, critic in enumerate(self.critics):
                critic_path = os.path.join(model_dir, f'critic_{i}.pth')
                torch.save(critic.state_dict(), critic_path)

            # === 保存两个global critic网络 ===
            global_reward_critic_path = os.path.join(model_dir, 'global_reward_critic.pth')
            torch.save(self.global_reward_critic.state_dict(), global_reward_critic_path)

            global_cost_critic_path = os.path.join(model_dir, 'global_cost_critic.pth')
            torch.save(self.global_cost_critic.state_dict(), global_cost_critic_path)

            # 保存优化器状态
            optimizer_states = {
                'optim_actors': [opt.state_dict() for opt in self.optim_actors],
                'optim_critics': [opt.state_dict() for opt in self.optim_critics],
                'optim_global_reward_critic': self.optim_global_reward_critic.state_dict(),
                'optim_global_cost_critic': self.optim_global_cost_critic.state_dict()
            }
            optimizer_path = os.path.join(model_dir, 'optimizers.pth')
            torch.save(optimizer_states, optimizer_path)

            print(f"模型已保存到: {model_dir}")

            # 内存优化：保存后清理
            del optimizer_states
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        except Exception as e:
            print(f"保存模型时出错: {e}")

    def load(self, model_dir):
        """加载所有模型参数"""
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")

        try:
            # 检查配置文件
            config_path = os.path.join(model_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)

                # 验证配置是否匹配
                current_config = self.get_model_info()
                for key in ['n_agents', 'obs_dim', 'action_dim']:
                    if saved_config.get(key) != current_config.get(key):
                        print(f"警告: 配置不匹配 {key}: 保存={saved_config.get(key)}, 当前={current_config.get(key)}")

            # 加载每个actor网络
            for i, actor in enumerate(self.actors):
                actor_path = os.path.join(model_dir, f'actor_{i}.pth')
                if os.path.exists(actor_path):
                    actor.load_state_dict(torch.load(actor_path, map_location=self.device))

            # 加载每个critic网络
            for i, critic in enumerate(self.critics):
                critic_path = os.path.join(model_dir, f'critic_{i}.pth')
                if os.path.exists(critic_path):
                    critic.load_state_dict(torch.load(critic_path, map_location=self.device))

            # === 加载两个global critic网络 ===
            global_reward_critic_path = os.path.join(model_dir, 'global_reward_critic.pth')
            if os.path.exists(global_reward_critic_path):
                self.global_reward_critic.load_state_dict(
                    torch.load(global_reward_critic_path, map_location=self.device))

            global_cost_critic_path = os.path.join(model_dir, 'global_cost_critic.pth')
            if os.path.exists(global_cost_critic_path):
                self.global_cost_critic.load_state_dict(torch.load(global_cost_critic_path, map_location=self.device))

            # 加载优化器状态（可选）
            optimizer_path = os.path.join(model_dir, 'optimizers.pth')
            if os.path.exists(optimizer_path):
                try:
                    optimizer_states = torch.load(optimizer_path, map_location=self.device)

                    for i, opt_state in enumerate(optimizer_states.get('optim_actors', [])):
                        if i < len(self.optim_actors):
                            self.optim_actors[i].load_state_dict(opt_state)

                    for i, opt_state in enumerate(optimizer_states.get('optim_critics', [])):
                        if i < len(self.optim_critics):
                            self.optim_critics[i].load_state_dict(opt_state)

                    if 'optim_global_reward_critic' in optimizer_states:
                        self.optim_global_reward_critic.load_state_dict(optimizer_states['optim_global_reward_critic'])

                    if 'optim_global_cost_critic' in optimizer_states:
                        self.optim_global_cost_critic.load_state_dict(optimizer_states['optim_global_cost_critic'])

                    # 内存优化：立即删除
                    del optimizer_states

                except Exception as e:
                    print(f"加载优化器状态时出错: {e}")

            # 确保所有模型都在正确的设备上
            for actor in self.actors:
                actor.to(self.device)
            for critic in self.critics:
                critic.to(self.device)
            self.global_reward_critic.to(self.device)
            self.global_cost_critic.to(self.device)

            print(f"模型已从 {model_dir} 加载成功")

            # 内存优化：加载后清理
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise e

    def set_train_mode(self):
        """设置所有网络为训练模式"""
        for actor in self.actors:
            actor.train()
        for critic in self.critics:
            critic.train()
        self.global_reward_critic.train()
        self.global_cost_critic.train()

    def set_eval_mode(self):
        """设置所有网络为评估模式"""
        for actor in self.actors:
            actor.eval()
        for critic in self.critics:
            critic.eval()
        self.global_reward_critic.eval()
        self.global_cost_critic.eval()