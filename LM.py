# LM.py - 内存优化版本：保持原逻辑，优化内存使用

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from datetime import datetime
import gc  # 垃圾回收


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
    内存优化版训练函数 - 保持原逻辑，减少内存占用
    """
    n_agents = mac.n_agents
    cost_limits = cost_limits or {'energy': 0.5, 'loss': 5}

    # 创建日志目录
    log_dir = "training_logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_log_dir = os.path.join(log_dir, f"training_{timestamp}")
    os.makedirs(current_log_dir, exist_ok=True)

    # === 初始化拉格朗日乘子（原逻辑不变）===
    lagrange_global_loss = LagrangeMultiplier(init_value=1.0, lr=0.01, device=device)
    lagrange_gs_energy = [LagrangeMultiplier(init_value=1.0, lr=0.01, device=device)
                          for _ in range(env.num_ground_stations)]
    lagrange_sat_energy = [LagrangeMultiplier(init_value=1.0, lr=0.01, device=device)
                           for _ in range(env.num_satellites)]

    # 内存优化：减小batch_size避免大张量
    effective_batch_size = min(batch_size, 32)  # 限制最大batch_size

    for ep in range(num_episodes):
        episode_log_file = os.path.join(current_log_dir, f"episode_{ep:04d}.log")

        obs = env.reset()
        done = False
        ep_reward = 0
        ep_global_cost = 0
        ep_gs_costs = [0] * env.num_ground_stations
        ep_sat_costs = [0] * env.num_satellites

        # 内存优化：使用列表而不是预分配大数组
        obs_buffer = []
        actions_buffer = []
        rewards_buffer = []
        global_obs_buffer = []
        global_cost_buffer = []
        gs_cost_buffers = [[] for _ in range(env.num_ground_stations)]
        sat_cost_buffers = [[] for _ in range(env.num_satellites)]

        with open(episode_log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Episode {ep} 开始 ===\n")

        step_count = 0
        while not done:
            neighbors = env._build_neighbors()
            actions = mac.select_actions(obs, neighbors)
            next_obs, rewards, done, costs, info = env.step(actions, neighbors)

            # === 收集成本数据（原逻辑）===
            global_cost = costs['loss'] if np.isscalar(costs['loss']) else np.sum(costs['loss'])

            gs_costs = []
            for j in range(env.num_ground_stations):
                if j < len(costs['energy']) - env.num_satellites:
                    gs_cost = costs['energy'][env.num_satellites + j] if hasattr(costs['energy'], '__len__') else 0
                else:
                    gs_cost = 0
                gs_costs.append(gs_cost)

            sat_costs = []
            for i in range(env.num_satellites):
                if i < len(costs['energy']) if hasattr(costs['energy'], '__len__') else 1:
                    sat_cost = costs['energy'][i] if hasattr(costs['energy'], '__len__') else costs['energy']
                else:
                    sat_cost = 0
                sat_costs.append(sat_cost)

            # 内存优化：存储数据到缓冲区，使用numpy而不是tensor
            global_obs = np.concatenate(obs, axis=0).astype(np.float32)  # 指定dtype
            obs_buffer.append(np.array(obs, dtype=np.float32))
            actions_buffer.append(np.array(actions, dtype=np.int32))  # 使用int32而不是int64
            rewards_buffer.append(np.array(rewards, dtype=np.float32))
            global_obs_buffer.append(global_obs)
            global_cost_buffer.append(np.float32(global_cost))  # 单个值

            for j in range(env.num_ground_stations):
                gs_cost_buffers[j].append(np.float32(gs_costs[j]))
            for i in range(env.num_satellites):
                sat_cost_buffers[i].append(np.float32(sat_costs[i]))

            obs = next_obs
            step_count += 1
            ep_reward += np.sum(rewards)
            ep_global_cost += global_cost

            for j in range(env.num_ground_stations):
                ep_gs_costs[j] += gs_costs[j]
            for i in range(env.num_satellites):
                ep_sat_costs[i] += sat_costs[i]

            # 内存优化：定期清理和检查内存
            if step_count % 100 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()

        T = len(obs_buffer)
        if T == 0:
            continue

        # === 计算cost-to-go（原逻辑，内存优化）===
        def compute_cost_to_go(cost_list, gamma):
            # 内存优化：直接计算，不存储中间结果
            cost_to_go = np.zeros(len(cost_list), dtype=np.float32)
            running = 0.0
            for t in reversed(range(len(cost_list))):
                is_done = (t == len(cost_list) - 1)
                running = cost_list[t] + gamma * running * (0.0 if is_done else 1.0)
                cost_to_go[t] = running
            return cost_to_go

        global_cost_to_go = compute_cost_to_go(global_cost_buffer, gamma)
        gs_cost_to_gos = [compute_cost_to_go(gs_cost_buffers[j], gamma) for j in range(env.num_ground_stations)]
        sat_cost_to_gos = [compute_cost_to_go(sat_cost_buffers[i], gamma) for i in range(env.num_satellites)]

        # === 内存优化的分批训练 ===
        # 使用更小的batch进行训练，减少峰值内存
        for start in range(0, T, effective_batch_size):
            end = min(start + effective_batch_size, T)
            B = end - start

            # 内存优化：只在需要时转换为tensor，立即释放numpy数组
            with torch.no_grad():
                obs_batch = torch.from_numpy(np.stack(obs_buffer[start:end])).to(device)
                act_batch = torch.from_numpy(np.stack(actions_buffer[start:end])).to(device)
                rew_batch = torch.from_numpy(np.stack(rewards_buffer[start:end])).to(device)
                global_obs_batch = torch.from_numpy(np.stack(global_obs_buffer[start:end])).to(device)

                done_batch = torch.zeros(B, dtype=torch.float32, device=device)
                if end == T:
                    done_batch[-1] = 1.0

                batch_global_cost_to_go = torch.from_numpy(global_cost_to_go[start:end]).to(device)

            # === 训练每个agent（原逻辑，内存优化）===
            for agent_idx in range(n_agents):
                agent = mac.actors[agent_idx]
                critic = mac.critics[agent_idx]
                optimizer_a = mac.optim_actors[agent_idx]
                optimizer_c = mac.optim_critics[agent_idx]

                # 内存优化：使用with torch.no_grad()减少梯度计算
                obs_agent = obs_batch[:, agent_idx, :]
                act_agent = act_batch[:, agent_idx]
                rew_agent = rew_batch[:, agent_idx]

                values = critic(obs_agent).squeeze(-1)

                with torch.no_grad():
                    if B > 1:
                        next_val = torch.cat([values[1:], values[-1:]])
                    else:
                        next_val = values

                td_target = rew_agent + gamma * next_val * (1 - done_batch)

                if B > 1:
                    advantage = td_target[:-1] - values[:-1]

                    logits = agent(obs_agent[:-1])

                    if torch.all(logits >= 0) and torch.allclose(logits.sum(dim=-1),
                                                                 torch.ones(logits.shape[0], device=device), atol=1e-6):
                        probs = logits
                    else:
                        probs = F.softmax(logits, dim=-1)

                    selected_probs = probs.gather(1, act_agent[:-1].unsqueeze(-1)).squeeze(-1)
                    logp = torch.log(selected_probs + 1e-8)

                    lagrange_loss = torch.tensor(0.0, device=device)
                    lagrange_loss += lagrange_global_loss() * batch_global_cost_to_go[:-1].mean()

                    if agent_idx < env.num_satellites:
                        sat_cost_to_go = torch.from_numpy(sat_cost_to_gos[agent_idx][start:end - 1]).to(device)
                        lagrange_loss += lagrange_sat_energy[agent_idx]() * sat_cost_to_go.mean()
                        del sat_cost_to_go  # 内存优化：立即释放
                    else:
                        gs_idx = agent_idx - env.num_satellites
                        if gs_idx < env.num_ground_stations:
                            gs_cost_to_go = torch.from_numpy(gs_cost_to_gos[gs_idx][start:end - 1]).to(device)
                            lagrange_loss += lagrange_gs_energy[gs_idx]() * gs_cost_to_go.mean()
                            del gs_cost_to_go  # 内存优化：立即释放

                    actor_loss = -torch.mean(logp * advantage.detach()) + lagrange_loss
                    critic_loss = F.mse_loss(values[:-1], td_target[:-1].detach())
                else:
                    actor_loss = torch.tensor(0.0, device=device)
                    critic_loss = F.mse_loss(values, td_target.detach())

                # 内存优化：清理梯度，限制梯度范数
                optimizer_a.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer_a.step()

                optimizer_c.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                optimizer_c.step()

                # 内存优化：删除中间变量
                del actor_loss, critic_loss, values, td_target
                if B > 1:
                    del advantage, logits, probs, selected_probs, logp, lagrange_loss

            # === 训练global critics（原逻辑，内存优化）===
            with torch.no_grad():
                global_reward_values = mac.global_reward_critic(global_obs_batch).squeeze(-1)
                reward_target = torch.from_numpy(
                    np.array([np.sum(rewards_buffer[t]) for t in range(start, end)], dtype=np.float32)).to(device)

            global_reward_loss = F.mse_loss(global_reward_values, reward_target.detach())

            global_cost_values = mac.global_cost_critic(global_obs_batch).squeeze(-1)
            global_cost_loss = F.mse_loss(global_cost_values, batch_global_cost_to_go.detach())

            mac.optim_global_reward_critic.zero_grad()
            global_reward_loss.backward()
            torch.nn.utils.clip_grad_norm_(mac.global_reward_critic.parameters(), max_norm=1.0)
            mac.optim_global_reward_critic.step()

            mac.optim_global_cost_critic.zero_grad()
            global_cost_loss.backward()
            torch.nn.utils.clip_grad_norm_(mac.global_cost_critic.parameters(), max_norm=1.0)
            mac.optim_global_cost_critic.step()

            # 内存优化：释放batch tensor
            del obs_batch, act_batch, rew_batch, global_obs_batch, done_batch, batch_global_cost_to_go
            del global_reward_values, reward_target, global_cost_values
            del global_reward_loss, global_cost_loss

            # 强制垃圾回收
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # === 更新拉格朗日乘子（原逻辑）===
        avg_global_cost = sum(global_cost_buffer) / T
        global_violation = avg_global_cost - cost_limits['loss']
        lagrange_global_loss.update(torch.tensor(global_violation, dtype=torch.float32, device=device))

        for j in range(env.num_ground_stations):
            avg_gs_cost = sum(gs_cost_buffers[j]) / T
            gs_violation = avg_gs_cost - cost_limits['energy']
            lagrange_gs_energy[j].update(torch.tensor(gs_violation, dtype=torch.float32, device=device))

        for i in range(env.num_satellites):
            avg_sat_cost = sum(sat_cost_buffers[i]) / T
            sat_violation = avg_sat_cost - cost_limits['energy']
            lagrange_sat_energy[i].update(torch.tensor(sat_violation, dtype=torch.float32, device=device))

        # 内存优化：清理episode数据
        del obs_buffer, actions_buffer, rewards_buffer, global_obs_buffer, global_cost_buffer
        del gs_cost_buffers, sat_cost_buffers, global_cost_to_go, gs_cost_to_gos, sat_cost_to_gos

        # 强制垃圾回收
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 记录训练日志（减少频率）
        if ep % 5 == 0:  # 改为每5个episode记录一次
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