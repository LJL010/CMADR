# LM.py - 修正版本：严格按照论文实现拉格朗日方法

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from datetime import datetime


class LagrangeMultiplier:
    """单约束拉格朗日乘子"""

    def __init__(self, init_value=1.0, lr=0.01, min_val=1e-3, max_val=100.0, device='cpu'):
        self.value = torch.tensor([init_value], dtype=torch.float32, requires_grad=True, device=device)
        self.lr = lr
        self.min_val = min_val
        self.max_val = max_val
        self.device = device

    def update(self, cost_violation):
        """根据约束违反程度更新拉格朗日乘子"""
        if not isinstance(cost_violation, torch.Tensor):
            cost_violation = torch.tensor(cost_violation, dtype=torch.float32, device=self.device)

        grad = cost_violation.detach()
        with torch.no_grad():
            self.value += self.lr * grad
            self.value.clamp_(self.min_val, self.max_val)
        self.value.requires_grad = True

    def __call__(self):
        return self.value


def train_cmadr(env, mac, num_episodes=500, gamma=0.98, cost_limits=None, device='cpu', batch_size=50):
    """
    按照论文CMADR算法实现的训练函数
    """
    n_agents = mac.n_agents
    cost_limits = cost_limits or {'energy': 0.5, 'loss': 5}

    # 创建日志目录
    log_dir = "training_logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_log_dir = os.path.join(log_dir, f"training_{timestamp}")
    os.makedirs(current_log_dir, exist_ok=True)

    # === 按照论文初始化拉格朗日乘子 ===
    # λC: 全局丢包率约束 (公式12)
    lagrange_global_loss = LagrangeMultiplier(init_value=1.0, lr=0.01, device=device)

    # λj: 地面站能耗约束 (公式13)
    lagrange_gs_energy = [LagrangeMultiplier(init_value=1.0, lr=0.01, device=device)
                          for _ in range(env.num_ground_stations)]

    # λi: 卫星能耗约束 (公式14)
    lagrange_sat_energy = [LagrangeMultiplier(init_value=1.0, lr=0.01, device=device)
                           for _ in range(env.num_satellites)]

    for ep in range(num_episodes):
        episode_log_file = os.path.join(current_log_dir, f"episode_{ep:04d}.log")

        obs = env.reset()
        done = False
        ep_reward = 0
        ep_global_cost = 0  # 全局丢包成本
        ep_gs_costs = [0] * env.num_ground_stations  # 地面站能耗成本
        ep_sat_costs = [0] * env.num_satellites  # 卫星能耗成本

        # 存储episode数据
        obs_list = []
        actions_list = []
        rewards_list = []
        global_obs_list = []
        global_cost_list = []  # 全局丢包成本
        gs_cost_lists = [[] for _ in range(env.num_ground_stations)]  # 地面站能耗成本
        sat_cost_lists = [[] for _ in range(env.num_satellites)]  # 卫星能耗成本

        with open(episode_log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Episode {ep} 开始 ===\n")

        step_count = 0
        while not done:
            neighbors = env._build_neighbors()
            actions = mac.select_actions(obs, neighbors)
            next_obs, rewards, done, costs, info = env.step(actions, neighbors)

            # === 按照论文定义收集成本数据 ===
            # 全局丢包成本 (对应公式12的JC(π))
            global_cost = costs['loss'] if np.isscalar(costs['loss']) else np.sum(costs['loss'])

            # 地面站能耗成本 (对应公式13的Jj(π))
            gs_costs = []
            for j in range(env.num_ground_stations):
                if j < len(costs['energy']) - env.num_satellites:
                    gs_cost = costs['energy'][env.num_satellites + j] if hasattr(costs['energy'], '__len__') else 0
                else:
                    gs_cost = 0
                gs_costs.append(gs_cost)

            # 卫星能耗成本 (对应公式14的Ji(π))
            sat_costs = []
            for i in range(env.num_satellites):
                if i < len(costs['energy']) if hasattr(costs['energy'], '__len__') else 1:
                    sat_cost = costs['energy'][i] if hasattr(costs['energy'], '__len__') else costs['energy']
                else:
                    sat_cost = 0
                sat_costs.append(sat_cost)

            # 存储数据
            global_obs = np.concatenate(obs, axis=0)
            obs_list.append(obs.copy())
            actions_list.append(actions.copy())
            rewards_list.append(rewards.copy())
            global_obs_list.append(global_obs.copy())
            global_cost_list.append(global_cost)

            for j in range(env.num_ground_stations):
                gs_cost_lists[j].append(gs_costs[j])
            for i in range(env.num_satellites):
                sat_cost_lists[i].append(sat_costs[i])

            obs = next_obs
            step_count += 1
            ep_reward += np.sum(rewards)
            ep_global_cost += global_cost

            for j in range(env.num_ground_stations):
                ep_gs_costs[j] += gs_costs[j]
            for i in range(env.num_satellites):
                ep_sat_costs[i] += sat_costs[i]

            # 日志记录
            log_message = f"Step {step_count}: global_cost={global_cost:.3f}, info={info}\n"
            with open(episode_log_file, 'a', encoding='utf-8') as f:
                f.write(log_message)

        T = len(obs_list)
        if T == 0:
            continue

        # === 按照论文计算cost-to-go (公式对应) ===
        def compute_cost_to_go(cost_list, gamma):
            cost_to_go = []
            running = 0.0
            for t in reversed(range(T)):
                is_done = (t == T - 1)
                running = cost_list[t] + gamma * running * (0.0 if is_done else 1.0)
                cost_to_go.insert(0, running)
            return cost_to_go

        # 计算各种cost-to-go
        global_cost_to_go = compute_cost_to_go(global_cost_list, gamma)
        gs_cost_to_gos = [compute_cost_to_go(gs_cost_lists[j], gamma) for j in range(env.num_ground_stations)]
        sat_cost_to_gos = [compute_cost_to_go(sat_cost_lists[i], gamma) for i in range(env.num_satellites)]

        # === 分批处理训练 ===
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            B = end - start

            # 准备批次数据
            obs_batch = torch.from_numpy(np.array(obs_list[start:end], dtype=np.float32)).to(device)
            act_batch = torch.from_numpy(np.array(actions_list[start:end], dtype=np.int64)).to(device)
            rew_batch = torch.from_numpy(np.array(rewards_list[start:end], dtype=np.float32)).to(device)
            global_obs_batch = torch.from_numpy(np.array(global_obs_list[start:end], dtype=np.float32)).to(device)

            done_batch = torch.zeros(B, dtype=torch.float32, device=device)
            if end == T:
                done_batch[-1] = 1.0

            # cost-to-go张量
            batch_global_cost_to_go = torch.from_numpy(np.array(global_cost_to_go[start:end], dtype=np.float32)).to(
                device)

            # === 按照论文公式训练global critics ===
            with torch.no_grad():
                global_reward_values = mac.global_reward_critic(global_obs_batch).squeeze(-1)
                global_cost_values = mac.global_cost_critic(global_obs_batch).squeeze(-1)

            # === 训练每个agent (按照论文公式21-22) ===
            for agent_idx in range(n_agents):
                agent = mac.actors[agent_idx]
                critic = mac.critics[agent_idx]
                optimizer_a = mac.optim_actors[agent_idx]
                optimizer_c = mac.optim_critics[agent_idx]

                # 提取agent数据
                obs_agent = obs_batch[:, agent_idx, :]
                act_agent = act_batch[:, agent_idx]
                rew_agent = rew_batch[:, agent_idx]

                # 计算值函数和优势
                values = critic(obs_agent).squeeze(-1)

                with torch.no_grad():
                    if B > 1:
                        next_val = torch.cat([values[1:], values[-1:]])
                    else:
                        next_val = values

                # TD目标和优势 (基于reward)
                td_target = rew_agent + gamma * next_val * (1 - done_batch)

                if B > 1:
                    advantage = td_target[:-1] - values[:-1]

                    # === Actor损失 (按照论文公式21) ===
                    logits = agent(obs_agent[:-1])

                    if torch.all(logits >= 0) and torch.allclose(logits.sum(dim=-1),
                                                                 torch.ones(logits.shape[0], device=device), atol=1e-6):
                        probs = logits
                    else:
                        probs = F.softmax(logits, dim=-1)

                    selected_probs = probs.gather(1, act_agent[:-1].unsqueeze(-1)).squeeze(-1)
                    logp = torch.log(selected_probs + 1e-8)

                    # 计算拉格朗日损失项
                    lagrange_loss = 0.0

                    # 全局丢包约束项
                    lagrange_loss += lagrange_global_loss() * batch_global_cost_to_go[:-1].mean()

                    # 个体能耗约束项
                    if agent_idx < env.num_satellites:
                        # 卫星约束
                        sat_cost_to_go = torch.from_numpy(
                            np.array(sat_cost_to_gos[agent_idx][start:end - 1], dtype=np.float32)).to(device)
                        lagrange_loss += lagrange_sat_energy[agent_idx]() * sat_cost_to_go.mean()
                    else:
                        # 地面站约束
                        gs_idx = agent_idx - env.num_satellites
                        if gs_idx < env.num_ground_stations:
                            gs_cost_to_go = torch.from_numpy(
                                np.array(gs_cost_to_gos[gs_idx][start:end - 1], dtype=np.float32)).to(device)
                            lagrange_loss += lagrange_gs_energy[gs_idx]() * gs_cost_to_go.mean()

                    # Actor总损失 (论文公式21)
                    actor_loss = -torch.mean(logp * advantage.detach()) + lagrange_loss

                    # Critic损失
                    critic_loss = F.mse_loss(values[:-1], td_target[:-1].detach())
                else:
                    actor_loss = torch.tensor(0.0, device=device)
                    critic_loss = F.mse_loss(values, td_target.detach())

                # 优化
                optimizer_a.zero_grad()
                actor_loss.backward()
                optimizer_a.step()

                optimizer_c.zero_grad()
                critic_loss.backward()
                optimizer_c.step()

            # === 训练global critics (按照论文公式28-30) ===
            # Global reward critic (预测累积奖励)
            global_reward_values = mac.global_reward_critic(global_obs_batch).squeeze(-1)
            reward_target = torch.from_numpy(
                np.array([np.sum(rewards_list[t]) for t in range(start, end)], dtype=np.float32)).to(device)

            global_reward_loss = F.mse_loss(global_reward_values, reward_target.detach())

            # Global cost critic (预测累积丢包成本)
            global_cost_values = mac.global_cost_critic(global_obs_batch).squeeze(-1)
            global_cost_loss = F.mse_loss(global_cost_values, batch_global_cost_to_go.detach())

            # 优化global critics
            mac.optim_global_reward_critic.zero_grad()
            global_reward_loss.backward()
            mac.optim_global_reward_critic.step()

            mac.optim_global_cost_critic.zero_grad()
            global_cost_loss.backward()
            mac.optim_global_cost_critic.step()

        # === 更新拉格朗日乘子 (按照论文公式23-27) ===
        # 全局丢包率约束违反程度
        avg_global_cost = sum(global_cost_to_go) / T
        global_violation = avg_global_cost - cost_limits['loss']
        lagrange_global_loss.update(torch.tensor(global_violation, dtype=torch.float32, device=device))

        # 地面站能耗约束违反程度
        for j in range(env.num_ground_stations):
            avg_gs_cost = sum(gs_cost_to_gos[j]) / T
            gs_violation = avg_gs_cost - cost_limits['energy']
            lagrange_gs_energy[j].update(torch.tensor(gs_violation, dtype=torch.float32, device=device))

        # 卫星能耗约束违反程度
        for i in range(env.num_satellites):
            avg_sat_cost = sum(sat_cost_to_gos[i]) / T
            sat_violation = avg_sat_cost - cost_limits['energy']
            lagrange_sat_energy[i].update(torch.tensor(sat_violation, dtype=torch.float32, device=device))

        # 清理内存
        del (obs_list, actions_list, rewards_list, global_obs_list, global_cost_list,
             gs_cost_lists, sat_cost_lists, global_cost_to_go, gs_cost_to_gos, sat_cost_to_gos)

        # 记录训练日志
        if ep % 1 == 0:
            summary_msg = (
                f"\nEpisode {ep}: reward={ep_reward:.2f} global_cost={ep_global_cost:.2f} "
                f"λ_global={lagrange_global_loss().item():.2f} steps={T}"
            )
            print(summary_msg)

            summary_log_file = os.path.join(current_log_dir, "training_summary.log")
            with open(summary_log_file, 'a', encoding='utf-8') as f:
                f.write(summary_msg + "\n")

    # 最终总结
    final_summary_file = os.path.join(current_log_dir, "final_summary.log")
    with open(final_summary_file, 'w', encoding='utf-8') as f:
        f.write(f"训练完成!\n")
        f.write(f"总Episodes: {num_episodes}\n")
        f.write(f"最终拉格朗日乘子 - Global Loss: {lagrange_global_loss().item():.4f}\n")
        f.write(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\n训练完成! 所有日志已保存到: {current_log_dir}")