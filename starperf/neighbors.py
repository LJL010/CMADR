from ISTN_ENV import ISTNEnv
import json
import argparse
import os
import numpy as np
from pathlib import Path

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

def load_data(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == '__main__':

    data_dir = Path("/Users/shaoyang/Desktop/CMADR/CMADR/starperf")
    data_name = "satellite_positions"
    data_path = os.path.join(data_dir, f"{data_name}.json")
    data = load_data(data_path)
    env = ISTNEnv(
        num_satellites=len(data['sat_positions_per_slot'][0]),
        num_ground_stations=261,
        max_time=len(data['sat_positions_per_slot']),
        sat_positions_per_slot=data['sat_positions_per_slot'],
        gs_positions=[tuple(p) for p in data['gs_positions']],
    )

    output_dir = "neighbors_data_slot_test"
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