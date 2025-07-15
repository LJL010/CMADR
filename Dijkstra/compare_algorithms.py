import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from ISTN_ENV import ISTNEnv
from MASys import MultiAgentSystem
from dijkstra_router import DijkstraRouter


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_data(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def evaluate_rl_model(env: ISTNEnv, mac: MultiAgentSystem):
    """评估强化学习模型"""
    obs = env.reset()
    done = False
    total_loss = 0
    total_energy = 0.0
    delivered = 0
    total_delay = 0

    print("开始RL算法评估...")

    while not done:
        neighbors = env._build_neighbors()
        actions = mac.select_actions(obs, neighbors)
        obs, reward, done, costs, info = env.step(actions, neighbors)

        total_loss += costs['loss']
        total_energy += float(np.sum(costs['energy']))
        delivered += info['delivered_packets']
        total_delay += sum(info['delays'])

    loss_rate = total_loss / (delivered + total_loss) if delivered + total_loss > 0 else 0
    avg_delay = total_delay / delivered if delivered > 0 else 0

    return {
        'loss_rate': loss_rate,
        'total_energy': total_energy,
        'avg_delay': avg_delay,
        'delivered_packets': delivered,
        'lost_packets': total_loss
    }


def evaluate_dijkstra_model(env: ISTNEnv, router: DijkstraRouter):
    """评估Dijkstra算法"""
    obs = env.reset()
    done = False
    total_loss = 0
    total_energy = 0.0
    delivered = 0
    total_delay = 0

    print("开始Dijkstra算法评估...")

    while not done:
        neighbors = env._build_neighbors()
        actions = router.select_actions(env, neighbors)
        obs, reward, done, costs, info = env.step(actions, neighbors)

        total_loss += costs['loss']
        total_energy += float(np.sum(costs['energy']))
        delivered += info['delivered_packets']
        total_delay += sum(info['delays'])

    loss_rate = total_loss / (delivered + total_loss) if delivered + total_loss > 0 else 0
    avg_delay = total_delay / delivered if delivered > 0 else 0

    return {
        'loss_rate': loss_rate,
        'total_energy': total_energy,
        'avg_delay': avg_delay,
        'delivered_packets': delivered,
        'lost_packets': total_loss
    }


def evaluate_random_model(env: ISTNEnv):
    """评估随机算法"""
    obs = env.reset()
    done = False
    total_loss = 0
    total_energy = 0.0
    delivered = 0
    total_delay = 0

    print("开始随机算法评估...")

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

    return {
        'loss_rate': loss_rate,
        'total_energy': total_energy,
        'avg_delay': avg_delay,
        'delivered_packets': delivered,
        'lost_packets': total_loss
    }


def create_comparison_plots(results: dict, data_name: str):
    """创建比较图表"""
    algorithms = list(results.keys())
    metrics = ['loss_rate', 'avg_delay', 'total_energy', 'delivered_packets']
    metric_names = ['丢包率', '平均延迟(s)', '总能耗', '交付包数']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'路由算法性能比较 - {data_name}', fontsize=16)

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i // 2, i % 2]

        values = [results[alg][metric] for alg in algorithms]
        colors = ['skyblue', 'lightgreen', 'lightcoral']

        bars = ax.bar(algorithms, values, color=colors[:len(algorithms)])
        ax.set_title(name, fontsize=14)
        ax.set_ylabel('值')

        # 在柱子上显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.3f}' if metric != 'delivered_packets' else f'{int(value)}',
                    ha='center', va='bottom')

    plt.tight_layout()

    # 保存图表
    plots_dir = "comparison_plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_file = os.path.join(plots_dir, f"algorithm_comparison_{data_name}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"比较图表已保存到: {plot_file}")


def main(config_path: str, include_rl: bool = True, include_random: bool = True):
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

    results = {}

    # 评估Dijkstra算法
    router = DijkstraRouter(
        num_satellites=env.num_satellites,
        num_ground_stations=env.num_ground_stations
    )
    results['Dijkstra'] = evaluate_dijkstra_model(env, router)

    # 评估强化学习模型（如果存在）
    if include_rl:
        try:
            mac = MultiAgentSystem(
                n_agents=env.num_satellites + env.num_ground_stations,
                n_nodes=env.num_satellites + env.num_ground_stations,
                obs_dim=env.obs_dim,
                action_dim=env.action_dim,
                hidden_dim=64,
                device=cfg.get('device', 'cpu'),
            )

            model_dir = os.path.join(cfg.get('model_root', 'model'),
                                     f"round_{cfg['train']['num_episodes']}_{data_name}")

            if os.path.exists(model_dir):
                mac.load(model_dir)
                results['RL'] = evaluate_rl_model(env, mac)
                print("RL模型评估完成")
            else:
                print(f"RL模型不存在: {model_dir}")
                include_rl = False
        except Exception as e:
            print(f"RL模型评估失败: {e}")
            include_rl = False

    # 评估随机算法
    if include_random:
        results['Random'] = evaluate_random_model(env)

    # 打印比较结果
    print(f"\n{'=' * 60}")
    print(f"{'算法性能比较 - ' + data_name:^60}")
    print(f"{'=' * 60}")

    print(f"{'算法':<10} {'丢包率':<10} {'平均延迟':<12} {'总能耗':<10} {'交付包数':<10}")
    print("-" * 60)

    for alg, result in results.items():
        print(f"{alg:<10} {result['loss_rate']:<10.3f} {result['avg_delay']:<12.3f} "
              f"{result['total_energy']:<10.3f} {result['delivered_packets']:<10d}")

    # 计算改进率（相对于随机算法）
    if include_random and len(results) > 1:
        print(f"\n{'=' * 60}")
        print("相对于随机算法的改进率:")
        print(f"{'=' * 60}")

        random_result = results['Random']
        for alg, result in results.items():
            if alg != 'Random':
                print(f"\n{alg} vs Random:")

                # 丢包率改进
                loss_improvement = (random_result['loss_rate'] - result['loss_rate']) / random_result['loss_rate'] * 100
                print(f"  丢包率改进: {loss_improvement:.1f}%")

                # 延迟改进
                if random_result['avg_delay'] > 0:
                    delay_improvement = (random_result['avg_delay'] - result['avg_delay']) / random_result[
                        'avg_delay'] * 100
                    print(f"  延迟改进: {delay_improvement:.1f}%")

                # 能耗改进
                energy_improvement = (random_result['total_energy'] - result['total_energy']) / random_result[
                    'total_energy'] * 100
                print(f"  能耗改进: {energy_improvement:.1f}%")

                # 交付率改进
                delivery_improvement = (result['delivered_packets'] - random_result['delivered_packets']) / \
                                       random_result['delivered_packets'] * 100
                print(f"  交付率改进: {delivery_improvement:.1f}%")

    # 创建比较图表
    create_comparison_plots(results, data_name)

    # 保存详细结果
    results_dir = "comparison_results"
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"algorithm_comparison_{data_name}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n详细结果已保存到: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--no-rl', action='store_true', help='Skip RL model evaluation')
    parser.add_argument('--no-random', action='store_true', help='Skip random baseline evaluation')
    args = parser.parse_args()

    main(args.config, include_rl=not args.no_rl, include_random=not args.no_random)