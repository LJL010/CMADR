import json
import argparse
import os
import gc
import torch
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


def optimize_memory_settings():
    """优化内存设置"""
    # PyTorch内存优化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False  # 减少内存碎片
        torch.backends.cudnn.deterministic = True

    # 设置更积极的垃圾回收
    gc.set_threshold(700, 10, 10)  # 更频繁的垃圾回收

    print("内存优化设置已应用")


def main(config_path: str):
    # 内存优化设置
    optimize_memory_settings()

    cfg = load_config(config_path)
    data_dir = cfg.get('data_dir', 'data')
    data_name = cfg.get('data_name', 'dataset')
    data_path = os.path.join(data_dir, f"{data_name}.json")

    train_cfg = cfg.get('train', {})

    print("正在加载数据...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    # 创建环境
    env = build_env_from_data(data, cfg)

    # === 内存优化：减小网络规模 ===
    # 动态计算action_dim，但设置合理上限
    max_satellite_neighbors = 4 + env.num_ground_stations  # ISL + 地面站
    max_gs_neighbors = env.num_satellites  # 可连接的卫星数
    action_dim = max(max_satellite_neighbors, max_gs_neighbors)

    print(f"环境信息:")
    print(f"- 卫星数量: {env.num_satellites}")
    print(f"- 地面站数量: {env.num_ground_stations}")
    print(f"- 观测维度: {env.obs_dim}")
    print(f"- 动作维度: {action_dim} (优化后)")

    # === 内存优化：减小网络规模 ===
    network_cfg = cfg.get('network', {})
    # 根据数据规模动态调整hidden_dim
    base_hidden_dim = network_cfg.get('hidden_dim', 64)
    if env.num_satellites > 500:
        hidden_dim = min(base_hidden_dim, 32)  # 大规模网络使用更小的隐藏层
    elif env.num_satellites > 200:
        hidden_dim = min(base_hidden_dim, 48)
    else:
        hidden_dim = base_hidden_dim

    print(f"- 隐藏层维度: {hidden_dim} (原始: {base_hidden_dim})")

    mac = MultiAgentSystem(
        n_agents=env.num_satellites + env.num_ground_stations,
        n_nodes=env.num_satellites + env.num_ground_stations,
        obs_dim=env.obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        device=cfg.get('device', 'cpu'),
    )

    # 内存优化：强制垃圾回收
    del data  # 释放数据
    gc.collect()

    # 打印网络信息
    try:
        model_info = mac.get_model_info()
        print(f"\n网络信息:")
        print(f"- 总参数数量: {model_info['total_parameters']:,}")
        print(f"- 隐藏层维度: {model_info['hidden_dim']}")
        print(f"- 设备: {model_info['device']}")
    except:
        print("\n网络已创建")

    # === 内存优化：调整训练参数 ===
    model_dir = os.path.join(cfg.get('model_root', 'model'), f"round_{train_cfg.get('num_episodes', 300)}_{data_name}")

    # 动态调整batch_size和episode数量
    original_episodes = train_cfg.get('num_episodes', 10)
    original_batch_size = train_cfg.get('batch_size', 50)

    # 根据网络规模调整
    if env.num_satellites > 500:
        episodes = min(original_episodes, 200)  # 减少训练轮数
        batch_size = min(original_batch_size, 20)  # 减小batch size
    elif env.num_satellites > 200:
        episodes = min(original_episodes, 250)
        batch_size = min(original_batch_size, 30)
    else:
        episodes = original_episodes
        batch_size = original_batch_size

    print(f"\n开始训练...")
    print(f"- 训练轮数: {episodes} (原始: {original_episodes})")
    print(f"- 批处理大小: {batch_size} (原始: {original_batch_size})")
    print(f"- 约束限制: {train_cfg.get('cost_limits', {'energy': 0.5, 'loss': 5})}")

    # 内存优化：在训练前再次清理
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    try:
        train_cmadr(
            env,
            mac,
            num_episodes=episodes,
            gamma=train_cfg.get('gamma', 0.98),
            cost_limits=train_cfg.get('cost_limits', {'energy': 0.5, 'loss': 5}),
            device=cfg.get('device', 'cpu'),
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        # 保存当前状态
        emergency_dir = os.path.join(model_dir, "emergency_save")
        os.makedirs(emergency_dir, exist_ok=True)
        try:
            mac.save(emergency_dir)
            print(f"紧急保存模型到: {emergency_dir}")
        except:
            print("紧急保存失败")
        raise e

    # 保存模型
    print(f"\n保存模型到: {model_dir}")
    try:
        mac.save(model_dir)

        # 保存训练配置
        config_save_path = os.path.join(model_dir, 'train_config.json')
        # 更新配置以反映实际使用的参数
        actual_cfg = cfg.copy()
        actual_cfg['train']['num_episodes'] = episodes
        actual_cfg['train']['batch_size'] = batch_size
        actual_cfg['network']['hidden_dim'] = hidden_dim
        actual_cfg['computed_action_dim'] = action_dim

        with open(config_save_path, 'w') as f:
            json.dump(actual_cfg, f, indent=2)

        print(f"训练完成！模型和配置已保存到: {model_dir}")

    except Exception as e:
        print(f"保存模型时出错: {e}")

    finally:
        # 最终清理
        del mac, env
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        print("内存清理完成")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CMADR算法训练脚本 - 内存优化版')
    parser.add_argument('--config', default='config.json', help='配置文件路径')
    args = parser.parse_args()
    main(args.config)