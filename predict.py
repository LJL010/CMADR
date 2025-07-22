import json
import argparse
import os
import numpy as np
import torch
from ISTN_ENV import ISTNEnv
from MASys import MultiAgentSystem


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_data(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_train_config(model_dir: str) -> dict:
    """加载训练时保存的配置"""
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            return json.load(f)

    # 如果没有train_config.json，尝试从config.json读取
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        return saved_config

    return None


def evaluate(env: ISTNEnv, mac: MultiAgentSystem):
    """
    修正版本：适应新的网络架构，在推理时只使用actor网络
    """
    # 设置为评估模式（按照论文，执行时只需要actor网络）
    mac.set_eval_mode()

    obs = env.reset()
    done = False
    total_loss = 0
    total_energy = 0.0
    delivered = 0
    total_delay = 0
    packets_in_transit = 0

    step_count = 0
    while not done:
        neighbors = env._build_neighbors()

        # === 按照论文，执行时只使用actor网络 ===
        with torch.no_grad():  # 推理时不需要梯度
            actions = mac.select_actions(obs, neighbors)

        obs, reward, done, costs, info = env.step(actions, neighbors)

        # 统计指标
        total_loss += costs['loss']
        total_energy += float(np.sum(costs['energy']))
        delivered += info['delivered_packets']
        packets_in_transit = info['packets_in_transit']
        total_delay += sum(info['delays']) if info['delays'] else 0

        step_count += 1

        # 打印步骤信息（可选）
        if step_count % 50 == 0:
            print(f"Step {step_count}: delivered={info['delivered_packets']}, "
                  f"transit={info['packets_in_transit']}, loss={costs['loss']}")

    # 计算最终指标
    loss_rate = total_loss / (delivered + total_loss) if delivered + total_loss > 0 else 0
    avg_delay = total_delay / delivered if delivered > 0 else 0

    print(f"\n=== 评估结果 ===")
    print(f"总丢包数: {total_loss}")
    print(f"成功交付数: {delivered}")
    print(f"总延迟: {total_delay}")
    print(f"在途包数: {packets_in_transit}")
    print(f"丢包率: {loss_rate:.3f}")
    print(f"总能耗: {total_energy:.3f}")
    print(f"平均延迟: {avg_delay:.3f}")

    return loss_rate, total_energy, avg_delay


def predict(config_path: str):
    """
    修正版本：从训练配置中读取网络参数
    """
    cfg = load_config(config_path)
    data_dir = cfg.get('data_dir', 'data')
    data_name = cfg.get('data_name', 'dataset')
    data_path = os.path.join(data_dir, f"{data_name}.json")
    data = load_data(data_path)

    # 创建环境
    env = ISTNEnv(
        num_satellites=len(data['sat_positions_per_slot'][0]),
        num_ground_stations=len(data['gs_positions']),
        max_time=cfg.get('predict', {}).get('max_time', len(data['sat_positions_per_slot'])),
        sat_positions_per_slot=data['sat_positions_per_slot'],
        gs_positions=[tuple(p) for p in data['gs_positions']],
        queries=data['predict_queries'],
    )

    # 模型目录
    model_dir = os.path.join(cfg.get('model_root', 'model'), f"{cfg['predict']['model_path']}")

    # === 关键修正：从训练配置中读取网络参数 ===
    train_config = load_train_config(model_dir)

    if train_config:
        print(f"从训练配置中读取网络参数...")
        # 使用训练时的网络参数
        hidden_dim = train_config.get('network', {}).get('hidden_dim', 64)
        computed_action_dim = train_config.get('computed_action_dim')

        print(f"训练时使用的hidden_dim: {hidden_dim}")
        if computed_action_dim:
            print(f"训练时计算的action_dim: {computed_action_dim}")
    else:
        print(f"警告：无法找到训练配置，使用默认参数")
        hidden_dim = 64
        computed_action_dim = None

    # 计算action_dim（与train.py保持一致）
    if computed_action_dim:
        action_dim = computed_action_dim
    else:
        max_satellite_neighbors = 4 + env.num_ground_stations
        max_gs_neighbors = env.num_satellites
        action_dim = max(max_satellite_neighbors, max_gs_neighbors)

    print(f"\n网络配置:")
    print(f"- 观测维度: {env.obs_dim}")
    print(f"- 动作维度: {action_dim}")
    print(f"- 隐藏层维度: {hidden_dim}")

    # 创建多智能体系统（使用训练时的参数）
    mac = MultiAgentSystem(
        n_agents=env.num_satellites + env.num_ground_stations,
        n_nodes=env.num_satellites + env.num_ground_stations,
        obs_dim=env.obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,  # 使用训练时的hidden_dim
        device=cfg.get('device', 'cpu'),
    )

    # 加载训练好的模型
    print(f"从 {model_dir} 加载模型...")
    try:
        mac.load(model_dir)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查模型路径是否正确")
        print("2. 确认模型文件完整性")
        print("3. 重新训练模型")
        return

    # 进行评估
    print("开始评估...")
    loss_rate, energy, avg_delay = evaluate(env, mac)

    print(f"\n=== 最终结果 ===")
    print(f"丢包率: {loss_rate:.3f}")
    print(f"总能耗: {energy:.3f}")
    print(f"平均延迟: {avg_delay:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to config file')
    args = parser.parse_args()
    predict(args.config)