# todo: 加入loss lagrange乘子对loss的贡献再训练

# todo: 加入loss lagrange乘子对loss的贡献再训练

import torch
import torch.nn.functional as F
import numpy as np


class LagrangeMultiplier:
    """ 单约束拉格朗日乘子 """

    def __init__(self, init_value=1.0, lr=0.01, min_val=1e-3, max_val=100.0, device='cpu'):
        self.value = torch.tensor([init_value], dtype=torch.float32, requires_grad=True, device=device)
        self.lr = lr
        self.min_val = min_val
        self.max_val = max_val

    def update(self, cost_violation):
        # cost_violation: torch scalar，>0 表示违反，<0表示未违反
        if not isinstance(cost_violation, torch.Tensor):
            cost_violation = torch.tensor(cost_violation, dtype=torch.float32, device=self.value.device)
        grad = cost_violation.detach()
        with torch.no_grad():
            self.value += self.lr * grad
            self.value.clamp_(self.min_val, self.max_val)
        self.value.requires_grad = True

    def __call__(self):
        return self.value


# === 训练主循环 ===
def train_cmadr(env, mac, num_episodes=500, gamma=0.98, cost_limits=None, device='cpu', batch_size=50):
    """
    内存优化版本的分批次处理episode数据的训练主循环

    修复了以下问题：
    1. Cost数据处理 - 确保存储标量值而非数组
    2. 张量创建优化 - 使用torch.from_numpy提高性能
    3. 尺寸匹配 - 确保所有张量尺寸正确
    4. 变量作用域 - 确保所有变量都正确定义
    5. 拉格朗日乘子更新 - 确保传入tensor类型
    6. Actor网络输出处理 - 修复logits vs 概率的问题
    """
    n_agents = mac.n_agents
    cost_limits = cost_limits or {'energy': 0.5, 'loss': 5}

    # 初始化拉格朗日乘子
    lagrange_energy = LagrangeMultiplier(init_value=1.0, lr=0.01, device=device)
    lagrange_loss = LagrangeMultiplier(init_value=1.0, lr=0.01, device=device)

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_energy_cost = 0
        ep_loss_cost = 0
        step_count = 0

        # 使用列表存储必要的数据，减少内存占用
        obs_list = []
        actions_list = []
        rewards_list = []
        cost_energy_list = []
        cost_loss_list = []
        global_obs_list = []

        # 调试信息：在第一个episode打印数据格式
        debug_first_step = (ep == 0)

        # 1. 运行当前episode，只收集必要数据
        while not done:
            neighbors = env._build_neighbors()
            actions = mac.select_actions(obs, neighbors)
            next_obs, rewards, done, costs, info = env.step(actions, neighbors)

            # 处理cost数据，确保是标量值
            # 调试信息：在第一步打印cost数据格式
            if debug_first_step and step_count == 0:
                print(
                    f"Debug - costs['energy'] type: {type(costs['energy'])}, shape: {getattr(costs['energy'], 'shape', 'scalar')}")
                print(
                    f"Debug - costs['loss'] type: {type(costs['loss'])}, shape: {getattr(costs['loss'], 'shape', 'scalar')}")

            # 确保cost是标量值（对所有智能体求和）
            current_energy_cost = np.sum(costs['energy']) if hasattr(costs['energy'], '__len__') else costs['energy']
            current_loss_cost = costs['loss'] if np.isscalar(costs['loss']) else np.sum(costs['loss'])

            # 只保存必要的数据
            global_obs = np.concatenate(obs, axis=0)
            obs_list.append(obs.copy())
            actions_list.append(actions.copy())
            rewards_list.append(rewards.copy())
            cost_energy_list.append(current_energy_cost)
            cost_loss_list.append(current_loss_cost)
            global_obs_list.append(global_obs.copy())

            obs = next_obs
            step_count += 1
            ep_reward += np.sum(rewards)
            ep_energy_cost += current_energy_cost
            ep_loss_cost += current_loss_cost

        T = len(obs_list)
        if T == 0:
            continue

        # 2. 流式计算cost_to_go以节省内存
        def compute_cost_to_go(cost_list, gamma):
            """流式计算cost_to_go，避免存储大张量"""
            cost_to_go = []
            running = 0.0
            for t in reversed(range(T)):
                # 只有当不是最后一步时才应用折扣
                is_done = (t == T - 1)
                running = cost_list[t] + gamma * running * (0.0 if is_done else 1.0)
                cost_to_go.insert(0, running)
            return cost_to_go

        # 计算cost_to_go（使用Python数值，避免张量开销）
        global_cost_energy_to_go = compute_cost_to_go(cost_energy_list, gamma)
        global_cost_loss_to_go = compute_cost_to_go(cost_loss_list, gamma)

        # 3. 分批次处理，及时释放内存
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            B = end - start

            # 创建当前批次的张量（先转换为numpy数组，再转换为tensor以提高性能）
            obs_batch = torch.from_numpy(
                np.array(obs_list[start:end], dtype=np.float32)
            ).to(device)
            act_batch = torch.from_numpy(
                np.array(actions_list[start:end], dtype=np.int64)
            ).to(device)
            rew_batch = torch.from_numpy(
                np.array(rewards_list[start:end], dtype=np.float32)
            ).to(device)
            global_obs_batch = torch.from_numpy(
                np.array(global_obs_list[start:end], dtype=np.float32)
            ).to(device)

            # 创建done标记（最后一步为done）
            done_batch = torch.zeros(B, dtype=torch.float32, device=device)
            if end == T:
                done_batch[-1] = 1.0

            # cost_to_go张量（现在应该是标量值的列表）
            batch_energy_to_go = torch.from_numpy(
                np.array(global_cost_energy_to_go[start:end], dtype=np.float32)
            ).to(device)
            batch_loss_to_go = torch.from_numpy(
                np.array(global_cost_loss_to_go[start:end], dtype=np.float32)
            ).to(device)

            # 计算全局值函数
            with torch.no_grad():
                global_values = mac.global_critic(global_obs_batch).squeeze(-1)

            # 4. 智能体训练循环
            for agent_idx in range(n_agents):
                agent = mac.actors[agent_idx]
                critic = mac.critics[agent_idx]
                optimizer_a = mac.optim_actors[agent_idx]
                optimizer_c = mac.optim_critics[agent_idx]

                # 提取当前智能体的数据
                obs_agent = obs_batch[:, agent_idx, :]
                act_agent = act_batch[:, agent_idx]
                rew_agent = rew_batch[:, agent_idx]

                # 计算值函数
                values = critic(obs_agent).squeeze(-1)

                # 计算下一步值函数
                with torch.no_grad():
                    if B > 1:
                        next_val = torch.cat([values[1:], values[-1:]])
                    else:
                        next_val = values

                # TD目标和优势
                td_target = rew_agent + gamma * next_val * (1 - done_batch)

                if B > 1:
                    advantage = td_target[:-1] - values[:-1]
                    # Actor损失 - 修复logits处理
                    logits = agent(obs_agent[:-1])

                    # 检查logits是否已经是概率分布
                    if torch.all(logits >= 0) and torch.allclose(logits.sum(dim=-1),
                                                                 torch.ones(logits.shape[0], device=device), atol=1e-6):
                        # 已经是概率分布
                        probs = logits
                    else:
                        # 是未归一化的logits，需要softmax
                        probs = F.softmax(logits, dim=-1)

                    # 计算log概率
                    selected_probs = probs.gather(1, act_agent[:-1].unsqueeze(-1)).squeeze(-1)
                    logp = torch.log(selected_probs + 1e-8)
                    actor_loss = -torch.mean(logp * advantage.detach())

                    # Critic损失
                    critic_loss = F.mse_loss(values[:-1], td_target[:-1].detach())
                else:
                    actor_loss = torch.tensor(0.0, device=device)
                    critic_loss = F.mse_loss(values, td_target.detach())

                # 立即优化，避免累积梯度
                optimizer_a.zero_grad()
                actor_loss.backward()
                optimizer_a.step()

                optimizer_c.zero_grad()
                critic_loss.backward()
                optimizer_c.step()

                # 清理中间变量
                del obs_agent, act_agent, rew_agent, values, td_target
                if B > 1:
                    del advantage, logits, probs, selected_probs, logp

            # 5. 全局网络训练
            global_values = mac.global_critic(global_obs_batch).squeeze(-1)

            if B > 1:
                global_critic_loss = F.mse_loss(
                    global_values[:-1], batch_energy_to_go[:-1].detach()
                )
                cost_violation_energy = batch_energy_to_go[:-1].mean() - cost_limits['energy']
                cost_violation_loss = batch_loss_to_go[:-1].mean() - cost_limits['loss']
            else:
                global_critic_loss = F.mse_loss(
                    global_values, batch_energy_to_go.detach()
                )
                cost_violation_energy = batch_energy_to_go.mean() - cost_limits['energy']
                cost_violation_loss = batch_loss_to_go.mean() - cost_limits['loss']

            # 拉格朗日约束项
            lagrange_energy_term = lagrange_energy() * cost_violation_energy
            lagrange_loss_term = lagrange_loss() * cost_violation_loss

            # 全局网络优化
            mac.optim_global_critic.zero_grad()
            total_global_loss = global_critic_loss + lagrange_energy_term + lagrange_loss_term
            total_global_loss.backward()
            mac.optim_global_critic.step()

            # 及时清理批次数据
            del (obs_batch, act_batch, rew_batch, global_obs_batch, done_batch,
                 batch_energy_to_go, batch_loss_to_go, global_values,
                 global_critic_loss, total_global_loss)

            # 清理GPU缓存（如果使用GPU）
            if device != 'cpu':
                torch.cuda.empty_cache()

        # 6. 更新拉格朗日乘子（使用episode平均值，确保传入tensor）
        avg_energy_cost = sum(global_cost_energy_to_go) / T
        avg_loss_cost = sum(global_cost_loss_to_go) / T

        total_energy_violation = avg_energy_cost - cost_limits['energy']
        total_loss_violation = avg_loss_cost - cost_limits['loss']

        # 转换为tensor再传入lagrange multiplier
        total_energy_violation_tensor = torch.tensor(total_energy_violation, dtype=torch.float32, device=device)
        total_loss_violation_tensor = torch.tensor(total_loss_violation, dtype=torch.float32, device=device)

        lagrange_energy.update(total_energy_violation_tensor)
        lagrange_loss.update(total_loss_violation_tensor)

        # 清理episode数据
        del (obs_list, actions_list, rewards_list, cost_energy_list,
             cost_loss_list, global_obs_list, global_cost_energy_to_go,
             global_cost_loss_to_go)

        # 打印训练日志
        if ep % 10 == 0:
            print(
                f"\nEpisode {ep}: reward={ep_reward:.2f} energy={ep_energy_cost:.2f} "
                f"loss={ep_loss_cost:.2f} λ_e={lagrange_energy().item():.2f} "
                f"λ_l={lagrange_loss().item():.2f} steps={T}"
            )
# def train_cmadr(env, mac, num_episodes=500, gamma=0.98, cost_limits=None, device='cpu'):
#     """
#     env: ISTNEnv
#     mac: MultiAgentSystem
#     cost_limits: dict, e.g. {'energy': 0.5, 'loss': 5}
#     """
#     n_agents = mac.n_agents
#     cost_limits = cost_limits or {'energy': 0.5, 'loss': 5}
#
#     # 1. 初始化拉格朗日乘子
#     lagrange_energy = LagrangeMultiplier(init_value=1.0, lr=0.01, device=device)
#     lagrange_loss = LagrangeMultiplier(init_value=1.0, lr=0.01, device=device)
#
#     for ep in range(num_episodes):
#         obs = env.reset()
#         episode_transitions = []
#         done = False
#         ep_reward = 0
#         ep_energy_cost = 0
#         ep_loss_cost = 0
#         step_count = 0
#         while not done:
#             # 动态生成当前时隙的拓扑
#             neighbors = env._build_neighbors()
#             actions = mac.select_actions(obs,neighbors)
#             next_obs, rewards, done, costs, info = env.step(actions, neighbors)
#
#             # 存储一条transition
#             global_obs = np.concatenate(obs, axis=0) # shape = [n_agents * obs_dim]
#             transition = {
#                 'obs': obs,
#                 'global_obs': global_obs,  # 新增
#                 'actions': actions,
#                 'rewards': rewards,
#                 'cost_energy': costs['energy'],
#                 'cost_loss': costs['loss'],
#                 'next_obs': next_obs,
#                 'done': done
#             }
#             episode_transitions.append(transition)
#             obs = next_obs
#             step_count += 1
#             ep_reward += np.sum(rewards)
#             ep_energy_cost += np.sum(costs['energy'])
#             ep_loss_cost += costs['loss']
#
#         # 2. 转换采样数据为批量
#         obs_batch = torch.tensor(np.array([tr['obs'] for tr in episode_transitions]), dtype=torch.float32, device=device)  # [T, n_agents, obs_dim]
#         act_batch = torch.tensor(np.array([tr['actions'] for tr in episode_transitions]), dtype=torch.long, device=device) # [T, n_agents]
#         rew_batch = torch.tensor(np.array([tr['rewards'] for tr in episode_transitions]), dtype=torch.float32, device=device) # [T, n_agents]
#         cost_energy_batch = torch.tensor(np.array([tr['cost_energy'] for tr in episode_transitions]), dtype=torch.float32, device=device)
#         cost_loss_batch = torch.tensor(np.array([tr['cost_loss'] for tr in episode_transitions]), dtype=torch.float32, device=device)
#         next_obs_batch = torch.tensor(np.array([tr['next_obs'] for tr in episode_transitions]), dtype=torch.float32, device=device)
#         done_batch = torch.tensor(np.array([tr['done'] for tr in episode_transitions]), dtype=torch.float32, device=device)
#         global_obs_batch = torch.tensor(np.array([tr['global_obs'] for tr in episode_transitions]), dtype=torch.float32, device=device) # [T, n_agents * obs_dim]
#
#         # 3. 计算advantage/target（简单时序差分或GAE均可，这里用TD）
#         # 对每个agent分别更新
#         for agent_idx in range(n_agents):
#             agent = mac.actors[agent_idx]
#             critic = mac.critics[agent_idx]
#             optimizer_a = mac.optim_actors[agent_idx]
#             optimizer_c = mac.optim_critics[agent_idx]
#             obs_agent = obs_batch[:, agent_idx, :]       # [T, obs_dim]
#             act_agent = act_batch[:, agent_idx]          # [T]
#             rew_agent = rew_batch[:, agent_idx]          # [T]
#             cost_e_agent = cost_energy_batch[:, agent_idx] # [T]
#
#             # 计算值函数和目标 - 基于reward而不是cost
#             values = critic(obs_agent).squeeze(-1)       # [T]
#             # bootstrapped TD target - 使用reward
#             td_target = rew_agent + gamma * torch.cat([values[1:], values[-1:]]) * (1-done_batch)
#             advantage = td_target[:-1] - values[:-1]
#
#             # Actor loss: 最大化reward (策略梯度)
#             logits = agent(obs_agent[:-1])
#             logp = torch.log(logits.gather(1, act_agent[:-1].unsqueeze(-1)).squeeze(-1) + 1e-8)
#             actor_loss = -torch.mean(logp * advantage.detach())  # 现在使用基于reward的advantage
#
#             # Critic loss: 预测reward的价值
#             critic_loss = F.mse_loss(values[:-1], td_target[:-1].detach())
#
#         # 全局cost处理（约束项）
#         global_values = mac.global_critic(global_obs_batch).squeeze(-1) # [T]
#
#         global_cost_enegy = cost_energy_batch.mean(dim=1)  # [T]
#         global_cost_enegy_to_go = []
#         running = 0
#         for t in reversed(range(len(global_cost_enegy))):
#             running = global_cost_enegy[t] + gamma * running * (1 - done_batch[t])
#             global_cost_enegy_to_go.insert(0, running)
#         global_cost_enegy_to_go = torch.tensor(global_cost_enegy_to_go, dtype=torch.float32, device=device)
#
#         global_cost_loss = cost_loss_batch  # [T]
#         global_cost_loss_to_go = []
#         running = 0
#         for t in reversed(range(len(global_cost_loss))):
#             running = global_cost_loss[t] + gamma * running * (1 - done_batch[t])
#             global_cost_loss_to_go.insert(0, running)
#         global_cost_loss_to_go = torch.tensor(global_cost_loss_to_go, dtype=torch.float32, device=device)
#
#
#
#         # 全局critic损失：预测cost
#         global_critic_loss = F.mse_loss(global_values[:-1], global_cost_enegy_to_go[:-1].detach())
#
#         # Lagrange约束项：惩罚cost超出限制
#         cost_violation = global_cost_enegy_to_go[:-1].mean() - cost_limits['energy']
#         lagrange_enegy_term = lagrange_energy() * cost_violation
#
#         # loss cost violation
#         cost_violation_loss = global_cost_loss_to_go[:-1].mean() - cost_limits['loss']
#         lagrange_loss_term = lagrange_loss() * cost_violation_loss
#
#         # 总损失：最大化reward + 约束cost
#         total_loss = actor_loss + critic_loss + global_critic_loss + lagrange_loss_term+ lagrange_enegy_term
#
#         optimizer_a.zero_grad()
#         optimizer_c.zero_grad()
#         mac.optim_global_critic.zero_grad()
#         total_loss.backward()
#         optimizer_a.step()
#         optimizer_c.step()
#         mac.optim_global_critic.step()
#
#         lagrange_energy.update(cost_violation)
#         lagrange_loss.update(cost_violation_loss)
#
#
#         # 4. 全局网络（可选：辅助优化/target value）
#         # joint_obs = obs_batch.reshape(obs_batch.shape[0], -1)  # [T, n_agents*obs_dim]
#         # mac.global_critic(joint_obs)  # ...
#
#         if ep % 10 == 0:
#             print(f"\nEpisode {ep}: reward={ep_reward:.2f} energy={ep_energy_cost:.2f} loss={ep_loss_cost:.2f} λ_e={lagrange_energy().item():.2f}\n")

