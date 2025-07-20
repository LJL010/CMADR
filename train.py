import json
import argparse
import os
from ISTN_ENV import ISTNEnv
from MASys import MultiAgentSystem
from LM import train_cmadr
from typing import Union


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def build_env_from_data(data: dict, cfg) -> ISTNEnv:
    return ISTNEnv(
        num_satellites=len(data['sat_positions_per_slot'][0]),
        num_ground_stations=len(data['gs_positions']),
        max_time=cfg['train']['max_time'] or len(data['sat_positions_per_slot']),
        sat_positions_per_slot=data['sat_positions_per_slot'],
        gs_positions=[tuple(p) for p in data['gs_positions']],
        queries=data['train_queries'],
    )


def main(config_path: str):
    cfg = load_config(config_path)
    data_dir = cfg.get('data_dir', 'data')
    data_name = cfg.get('data_name', 'dataset')
    data_path = os.path.join(data_dir, f"{data_name}.json")

    train_cfg = cfg.get('train', {})

    with open(data_path, 'r') as f:
        data = json.load(f)

    # 创建环境
    env = build_env_from_data(data, cfg)

    # === 修正：动态计算action_dim ===
    # 在真实的ISTN网络中，action_dim应该是最大可能的邻居数
    # 对于卫星：通常4个ISL邻居 + 可能的地面站连接
    # 对于地面站：可能连接的卫星数
    max_satellite_neighbors = 4 + env.num_ground_stations  # ISL + 地面站
    max_gs_neighbors = env.num_satellites  # 可连接的卫星数
    action_dim = max(max_satellite_neighbors, max_gs_neighbors)

    print(f"环境信息:")
    print(f"- 卫星数量: {env.num_satellites}")
    print(f"- 地面站数量: {env.num_ground_stations}")
    print(f"- 观测维度: {env.obs_dim}")
    print(f"- 动作维度: {action_dim}")

    # === 修正：使用网络配置中的参数 ===
    network_cfg = cfg.get('network', {})
    hidden_dim = network_cfg.get('hidden_dim', 64)

    mac = MultiAgentSystem(
        n_agents=env.num_satellites + env.num_ground_stations,
        n_nodes=env.num_satellites + env.num_ground_stations,
        obs_dim=env.obs_dim,
        action_dim=action_dim,  # 修正：使用计算得到的action_dim
        hidden_dim=hidden_dim,  # 修正：从配置中读取
        device=cfg.get('device', 'cpu'),
    )

    # 打印网络信息
    model_info = mac.get_model_info()
    print(f"\n网络信息:")
    print(f"- 总参数数量: {model_info['total_parameters']:,}")
    print(f"- 隐藏层维度: {model_info['hidden_dim']}")
    print(f"- 设备: {model_info['device']}")

    # === 修正：传递所有必要的参数给train_cmadr ===
    model_dir = os.path.join(cfg.get('model_root', 'model'), f"round_{train_cfg.get('num_episodes', 300)}_{data_name}")

    print(f"\n开始训练...")
    print(f"- 训练轮数: {train_cfg.get('num_episodes', 10)}")
    print(f"- 批处理大小: {train_cfg.get('batch_size', 50)}")
    print(f"- 约束限制: {train_cfg.get('cost_limits', {'energy': 0.5, 'loss': 5})}")

    train_cmadr(
        env,
        mac,
        num_episodes=train_cfg.get('num_episodes', 10),
        gamma=train_cfg.get('gamma', 0.98),
        cost_limits=train_cfg.get('cost_limits', {'energy': 0.5, 'loss': 5}),
        device=cfg.get('device', 'cpu'),
        batch_size=train_cfg.get('batch_size', 50),  # 修正：传递batch_size参数
    )

    # 保存模型
    print(f"\n保存模型到: {model_dir}")
    mac.save(model_dir)

    # 保存训练配置
    config_save_path = os.path.join(model_dir, 'train_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"训练完成！模型和配置已保存到: {model_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CMADR算法训练脚本')
    parser.add_argument('--config', default='config.json', help='配置文件路径')
    args = parser.parse_args()
    main(args.config)