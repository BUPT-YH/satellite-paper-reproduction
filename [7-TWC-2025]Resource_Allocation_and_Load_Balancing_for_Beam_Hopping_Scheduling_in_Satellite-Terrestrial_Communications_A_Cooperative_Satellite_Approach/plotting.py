"""
绘图模块 — 统一风格的论文复现图表
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import config as cfg

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = cfg.DPI
plt.rcParams['savefig.dpi'] = cfg.DPI
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 12


def plot_fig3(training_rewards_list, save_path='output/fig3_training_reward.png'):
    """
    Fig.3: DQN训练期间的平均奖励曲线 (各卫星分别显示, 匹配论文)
    training_rewards_list: 各卫星的训练奖励列表
    """
    fig, ax = plt.subplots(figsize=cfg.FIG_SIZE_SINGLE)

    all_rewards = np.array(training_rewards_list)
    n_sat = all_rewards.shape[0]
    window = 50
    colors_per_sat = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']

    # 各卫星分别绘制原始曲线(淡) + 滑动平均(实线)
    for s in range(min(n_sat, 4)):
        raw = all_rewards[s]
        smooth = np.convolve(raw, np.ones(window)/window, mode='valid')
        ax.plot(raw, alpha=0.15, color=colors_per_sat[s], linewidth=0.3)
        ax.plot(range(window-1, len(raw)), smooth,
                color=colors_per_sat[s], linewidth=1.5,
                label=f'Satellite {s+1}')

    ax.set_xlabel('Training Time Slot')
    ax.set_ylabel('Average Reward')
    ax.set_title('Fig.3: Average Reward During Training (β=0.7, 35 Gbps)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_fig4(convergence_data, save_path='output/fig4_mm_convergence.png'):
    """
    Fig.4: MM算法收敛性能
    convergence_data: dict with keys 'whole', 'neighbor', 'no_coop'
    """
    fig, ax = plt.subplots(figsize=cfg.FIG_SIZE_SINGLE)

    labels = {'whole': 'Whole Constellation',
              'neighbor': 'Neighbor Cooperation',
              'no_coop': 'No Cooperation'}
    colors = {'whole': cfg.COLORS['proposed'],
              'neighbor': cfg.COLORS['without_ra'],
              'no_coop': cfg.COLORS['original']}
    markers = {'whole': 'o', 'neighbor': 's', 'no_coop': '^'}

    for key, data in convergence_data.items():
        ax.plot(data, color=colors[key], marker=markers[key],
                markevery=max(1, len(data)//10), label=labels[key], linewidth=2)

    ax.set_xlabel('MM Iteration')
    ax.set_ylabel('Total Throughput (Gbps)')
    ax.set_title('Fig.4: Convergence of Resource Allocation Algorithm')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_fig5(throughput_data, latency_data, save_path='output/fig5_method_comparison.png'):
    """
    Fig.5: 不同方法对比 — 吞吐量和延迟度量 vs 输入流量
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg.FIG_SIZE_DOUBLE)

    traffic_rates = sorted(throughput_data.keys())

    # (a) 吞吐量
    for method in ['proposed', 'drl_avoid', 'max_uswg', 'pre_scheduling']:
        if method in throughput_data.get(traffic_rates[0], {}):
            tp_values = [throughput_data[r][method] for r in traffic_rates]
            ax1.plot(traffic_rates, tp_values,
                     color=cfg.COLORS[method], marker=cfg.MARKERS[method],
                     label=_method_label(method), linewidth=2, markersize=8,
                     markevery=max(1, len(traffic_rates)//8))

    ax1.set_xlabel('Total Data Rate of Input Traffic (Gbps)')
    ax1.set_ylabel('Throughput per Cell (Mbps)')
    ax1.set_title('(a) Throughput')
    ax1.legend(fontsize=10)

    # (b) 延迟度量
    for method in ['proposed', 'drl_avoid', 'max_uswg', 'pre_scheduling']:
        if method in latency_data.get(traffic_rates[0], {}):
            lt_values = [latency_data[r][method] for r in traffic_rates]
            ax2.plot(traffic_rates, lt_values,
                     color=cfg.COLORS[method], marker=cfg.MARKERS[method],
                     label=_method_label(method), linewidth=2, markersize=8,
                     markevery=max(1, len(traffic_rates)//8))

    ax2.set_xlabel('Total Data Rate of Input Traffic (Gbps)')
    ax2.set_ylabel('Latency Metric')
    ax2.set_title('(b) Latency Metric')
    ax2.legend(fontsize=10)

    fig.suptitle('Fig.5: Performance Comparison with Different Methods', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_fig6(throughput_data, latency_data, beta_range, save_path='output/fig6_beta_tradeoff.png'):
    """
    Fig.6: 不同折衷系数 β 下的吞吐量和延迟度量
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg.FIG_SIZE_DOUBLE)

    for method in ['proposed', 'without_drl', 'without_ra', 'without_lb', 'original']:
        if method in throughput_data:
            tp_values = throughput_data[method]
            ax1.plot(beta_range[:len(tp_values)], tp_values,
                     color=cfg.COLORS[method], marker=cfg.MARKERS[method],
                     label=_method_label(method), linewidth=2, markersize=7,
                     markevery=max(1, len(beta_range)//6))

    ax1.set_xlabel('Trade-off Coefficient β')
    ax1.set_ylabel('Throughput (Gbps)')
    ax1.set_title('(a) Throughput')
    ax1.legend(fontsize=9)

    for method in ['proposed', 'without_drl', 'without_ra', 'without_lb', 'original']:
        if method in latency_data:
            lt_values = latency_data[method]
            ax2.plot(beta_range[:len(lt_values)], lt_values,
                     color=cfg.COLORS[method], marker=cfg.MARKERS[method],
                     label=_method_label(method), linewidth=2, markersize=7,
                     markevery=max(1, len(beta_range)//6))

    ax2.set_xlabel('Trade-off Coefficient β')
    ax2.set_ylabel('Latency Metric')
    ax2.set_title('(b) Latency Metric')
    ax2.legend(fontsize=9)

    fig.suptitle('Fig.6: Performance vs Trade-off Coefficient β (35 Gbps)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_fig7(throughput_data, latency_data, save_path='output/fig7_traffic_load.png'):
    """
    Fig.7: 不同输入流量下的吞吐量和延迟度量 (消融实验)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg.FIG_SIZE_DOUBLE)

    traffic_rates = sorted(throughput_data.keys())

    for method in ['proposed', 'without_drl', 'without_ra', 'without_lb', 'original']:
        if method in throughput_data.get(traffic_rates[0] if traffic_rates else 0, {}):
            tp_values = [throughput_data[r][method] for r in traffic_rates]
            ax1.plot(traffic_rates, tp_values,
                     color=cfg.COLORS[method], marker=cfg.MARKERS[method],
                     label=_method_label(method), linewidth=2, markersize=7,
                     markevery=max(1, len(traffic_rates)//8))

    ax1.set_xlabel('Total Data Rate of Input Traffic (Gbps)')
    ax1.set_ylabel('Throughput per Cell (Mbps)')
    ax1.set_title('(a) Throughput')
    ax1.legend(fontsize=9)

    for method in ['proposed', 'without_drl', 'without_ra', 'without_lb', 'original']:
        if method in latency_data.get(traffic_rates[0] if traffic_rates else 0, {}):
            lt_values = [latency_data[r][method] for r in traffic_rates]
            ax2.plot(traffic_rates, lt_values,
                     color=cfg.COLORS[method], marker=cfg.MARKERS[method],
                     label=_method_label(method), linewidth=2, markersize=7,
                     markevery=max(1, len(traffic_rates)//8))

    ax2.set_xlabel('Total Data Rate of Input Traffic (Gbps)')
    ax2.set_ylabel('Latency Metric')
    ax2.set_title('(b) Latency Metric')
    ax2.legend(fontsize=9)

    fig.suptitle('Fig.7: Performance vs Input Traffic (β=0.7)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_fig8(throughput_data, latency_data, ns_range, save_path='output/fig8_satellite_number.png'):
    """
    Fig.8: 不同卫星数量下的吞吐量和延迟度量
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cfg.FIG_SIZE_DOUBLE)

    for method in ['proposed', 'without_drl', 'without_ra', 'without_lb', 'original']:
        if method in throughput_data:
            tp_values = throughput_data[method]
            ax1.plot(ns_range[:len(tp_values)], tp_values,
                     color=cfg.COLORS[method], marker=cfg.MARKERS[method],
                     label=_method_label(method), linewidth=2, markersize=7)

    ax1.set_xlabel('Number of Satellites Ns')
    ax1.set_ylabel('Throughput per Cell (Mbps)')
    ax1.set_title('(a) Throughput')
    ax1.legend(fontsize=9)

    for method in ['proposed', 'without_drl', 'without_ra', 'without_lb', 'original']:
        if method in latency_data:
            lt_values = latency_data[method]
            ax2.plot(ns_range[:len(lt_values)], lt_values,
                     color=cfg.COLORS[method], marker=cfg.MARKERS[method],
                     label=_method_label(method), linewidth=2, markersize=7)

    ax2.set_xlabel('Number of Satellites Ns')
    ax2.set_ylabel('Latency Metric')
    ax2.set_title('(b) Latency Metric')
    ax2.legend(fontsize=9)

    fig.suptitle('Fig.8: Performance vs Number of Satellites (β=0.7, 35 Gbps)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def _method_label(method):
    """方法标签映射"""
    labels = {
        'proposed': 'Proposed Method',
        'without_drl': 'Without DRL',
        'without_ra': 'Without Resource Allocation',
        'without_lb': 'Without Load Balancing',
        'original': 'Original Benchmark',
        'drl_avoid': 'DRL + Adjacent Avoidance [26]',
        'max_uswg': 'Maximum USWG [21]',
        'pre_scheduling': 'Pre-scheduling [14]',
    }
    return labels.get(method, method)
