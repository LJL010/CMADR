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


    output_dir = "neighbors_data"
    os.makedirs(output_dir, exist_ok=True)
    while env.time_slot < env.max_time:
        env.sat_positions = [np.array(p) for p in env.sat_positions_per_slot[env.time_slot]]
        neighbors = env._build_neighbors()
        # 使用 env.time_slot 生成文件名
        filename = os.path.join(output_dir, f"neighbors_slot_{env.time_slot}.json")
        # 保存结果到文件
        try:
            with open(filename, 'w') as f:
                json.dump(neighbors, f, indent=2)
            print(f"Neighbors data saved to {filename}")
        except Exception as e:
            print(f"Error saving file {filename}: {e}")

        # 继续下一个时间步
        env.time_slot += 1