import json
import argparse
import os
import numpy as np
from ISTN_ENV import ISTNEnv
from dijkstra_router import DijkstraRouter


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_data(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def evaluate_dijkstra(env: ISTNEnv, router: DijkstraRouter):
    """
    使用Dijkstra算法评估路由性能
    """
    obs = env.reset()
    done = False
    total_loss = 0
    total_energy = 0.0
    delivered = 0
    total_delay = 0
    total_hops = 0
    packets_generated = 0

    print("开始Dijkstra算法评估...")

    while not done:
        # 获取当前时隙的网络拓扑
        neighbors = env._build_neighbors()

        # 统计当前有多少包在系统中
        current_packets = 0
        for i in range(env.num_satellites):
            current_packets += len(env.satellites[i]['buffer'])
        for i in range(env.num_ground_stations):
            current_packets += len(env.ground_stations[i]['buffer'])

        print(f"时隙 {env.time_slot}: 系统中有 {current_packets} 个数据包")

        # 使用Dijkstra算法选择动作
        actions = router.select_actions(env, neighbors)

        # 执行动作
        obs, reward, done, costs, info = env.step(actions, neighbors)

        # 累计统计信息
        total_loss += costs['loss']
        total_energy += float(np.sum(costs['energy']))
        delivered += info['delivered_packets']
        packets_in_transit = info['packets_in_transit']
        total_delay += sum(info['delays'])

        # 统计跳数
        for i in range(env.num_satellites):
            for packet in env.satellites[i]['buffer']:
                total_hops += packet['hop']
        for i in range(env.num_ground_stations):
            for packet in env.ground_stations[i]['buffer']:
                total_hops += packet['hop']

        print(f"  - 本时隙交付: {info['delivered_packets']} 包")
        print(f"  - 本时隙丢失: {costs['loss']} 包")
        print(f"  - 本时隙能耗: {np.sum(costs['energy']):.3f}")

    # 计算最终指标
    loss_rate = total_loss / (delivered + total_loss) if delivered + total_loss > 0 else 0
    avg_delay = total_delay / delivered if delivered > 0 else 0
    avg_hops = total_hops / delivered if delivered > 0 else 0

    print(f"\n=== Dijkstra算法评估结果 ===")
    print(f"总丢包数: {total_loss}")
    print(f"总交付数: {delivered}")
    print(f"总延迟: {total_delay:.2f} 秒")
    print(f"在途包数: {packets_in_transit}")
    print(f"丢包率: {loss_rate:.3f}")
    print(f"平均延迟: {avg_delay:.3f} 秒")
    print(f"平均跳数: {avg_hops:.2f}")
    print(f"总能耗: {total_energy:.3f}")

    return {
        'loss_rate': loss_rate,
        'total_energy': total_energy,
        'avg_delay': avg_delay,
        'avg_hops': avg_hops,
        'delivered_packets': delivered,
        'lost_packets': total_loss,
        'packets_in_transit': packets_in_transit
    }


def compare_with_random_baseline(env: ISTNEnv):
    """
    随机选择baseline，用于对比
    """
    obs = env.reset()
    done = False
    total_loss = 0
    total_energy = 0.0
    delivered = 0
    total_delay = 0

    print("\n开始随机算法评估...")

    while not done:
        neighbors = env._build_neighbors()

        # 随机选择动作
        actions = []
        for i in range(env.n_agents):
            current_neighbors = neighbors.get(i, [])
            if current_neighbors:
                action = np.random.randint(0, len(current_neighbors))
            else:
                action = 0
            actions.append(action)

        obs, reward, done, costs, info = env.step(actions, neighbors)

        total_loss += costs['loss']
        total_energy += float(np.sum(costs['energy']))
        delivered += info['delivered_packets']
        total_delay += sum(info['delays'])

    loss_rate = total_loss / (delivered + total_loss) if delivered + total_loss > 0 else 0
    avg_delay = total_delay / delivered if delivered > 0 else 0

    print(f"\n=== 随机算法评估结果 ===")
    print(f"丢包率: {loss_rate:.3f}")
    print(f"平均延迟: {avg_delay:.3f} 秒")
    print(f"总能耗: {total_energy:.3f}")
    print(f"交付包数: {delivered}")

    return {
        'loss_rate': loss_rate,
        'total_energy': total_energy,
        'avg_delay': avg_delay,
        'delivered_packets': delivered,
        'lost_packets': total_loss
    }


def main(config_path: str, compare_random: bool = False):
    cfg = load_config(config_path)
    data_dir = cfg.get('data_dir', 'data')
    data_name = cfg.get('data_name', 'dataset')
    data_path = os.path.join(data_dir, f"{data_name}.json")

    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return

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

    # 创建Dijkstra路由器
    router = DijkstraRouter(
        num_satellites=env.num_satellites,
        num_ground_stations=env.num_ground_stations
    )

    # 评估Dijkstra算法
    dijkstra_results = evaluate_dijkstra(env, router)

    # 可选：与随机算法对比
    if compare_random:
        random_results = compare_with_random_baseline(env)

        print(f"\n=== 算法对比 ===")
        print(f"{'指标':<15} {'Dijkstra':<12} {'Random':<12} {'改进':<12}")
        print("-" * 55)

        # 丢包率对比
        improvement = (random_results['loss_rate'] - dijkstra_results['loss_rate']) / random_results['loss_rate'] * 100
        print(
            f"{'丢包率':<15} {dijkstra_results['loss_rate']:<12.3f} {random_results['loss_rate']:<12.3f} {improvement:<12.1f}%")

        # 平均延迟对比
        if random_results['avg_delay'] > 0:
            improvement = (random_results['avg_delay'] - dijkstra_results['avg_delay']) / random_results[
                'avg_delay'] * 100
            print(
                f"{'平均延迟(s)':<15} {dijkstra_results['avg_delay']:<12.3f} {random_results['avg_delay']:<12.3f} {improvement:<12.1f}%")

        # 能耗对比
        improvement = (random_results['total_energy'] - dijkstra_results['total_energy']) / random_results[
            'total_energy'] * 100
        print(
            f"{'总能耗':<15} {dijkstra_results['total_energy']:<12.3f} {random_results['total_energy']:<12.3f} {improvement:<12.1f}%")

        # 交付率对比
        improvement = (dijkstra_results['delivered_packets'] - random_results['delivered_packets']) / random_results[
            'delivered_packets'] * 100
        print(
            f"{'交付包数':<15} {dijkstra_results['delivered_packets']:<12d} {random_results['delivered_packets']:<12d} {improvement:<12.1f}%")

    # 保存结果
    results_dir = "baseline_results"
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"dijkstra_results_{data_name}.json")
    with open(results_file, 'w') as f:
        json.dump(dijkstra_results, f, indent=2)

    print(f"\n结果已保存到: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--compare-random', action='store_true', help='Compare with random baseline')
    args = parser.parse_args()

    main(args.config, args.compare_random)