from ISTN_ENV import ISTNEnv
import json
import argparse
import os
import numpy as np

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def load_data(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

# 协作搭建neighbors
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to config file')
    args = parser.parse_args()
    config_path = args.config

    cfg = load_config(config_path)
    data_dir = cfg.get('data_dir', 'data')
    data_name = cfg.get('data_name', 'dataset')
    data_path = os.path.join(data_dir, f"{data_name}.json")
    data = load_data(data_path)
    env = ISTNEnv(
        num_satellites=len(data['sat_positions_per_slot'][0]),
        num_ground_stations=len(data['gs_positions']),
        max_time=cfg.get('predict', {}).get('max_time', len(data['sat_positions_per_slot'])),
        sat_positions_per_slot=data['sat_positions_per_slot'],
        gs_positions=[tuple(p) for p in data['gs_positions']],
        queries=data['predict_queries'],
    )
    # 创建保存目录（如果不存在）
    output_dir = "neighbors_data"
    os.makedirs(output_dir, exist_ok=True)

    env.time_slot += 1
    neighbors = env._build_neighbors()
    print(neighbors)
    print("------------------")
    print({k: sorted(v) for k, v in neighbors.items()})
