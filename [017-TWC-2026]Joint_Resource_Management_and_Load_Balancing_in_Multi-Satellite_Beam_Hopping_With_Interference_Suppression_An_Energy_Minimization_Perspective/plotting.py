"""
IEEE 期刊风格绘图模块
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os

# ===== IEEE 期刊风格全局配置 =====
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05
rcParams['axes.linewidth'] = 0.8
rcParams['lines.linewidth'] = 1.5
rcParams['lines.markersize'] = 6
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True
rcParams['xtick.minor.width'] = 0.5
rcParams['ytick.minor.width'] = 0.5
rcParams['grid.linewidth'] = 0.3
rcParams['grid.alpha'] = 0.3

# IEEE 单栏/双栏尺寸
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)

# 颜色 + 标记 + 线型
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_bcd_convergence(results, filename='fig2_bcd_convergence.png'):
    """
    Fig. 2: BCD 算法迭代收敛曲线
    显示原始 Λ 值 (不归一化), 与论文 Fig.2 一致
    """
    fig, ax = plt.subplots(1, 1, figsize=FIG_SINGLE)

    for idx, (num_sats, obj_history) in enumerate(sorted(results.items())):
        iters = list(range(1, len(obj_history) + 1))
        ax.plot(iters, obj_history,
                color=COLORS[idx], marker=MARKERS[idx], linestyle=LINESTYLES[0],
                label=f'$|\\mathcal{{S}}|$ = {num_sats}',
                markevery=max(1, len(iters) // 8))

    ax.set_xlabel('BCD Iteration')
    ax.set_ylabel('Objective Value $\\Lambda$')
    ax.legend(framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  保存: {filepath}")
    return filepath


def plot_v_tradeoff(results_dict, filename='fig3_v_tradeoff.png'):
    """
    Fig. 3: 不同 V 值下的 (a) 平均队列长度 (b) 平均功率
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    V_values = sorted(results_dict.keys())

    # (a) 平均队列长度 vs 时隙
    for idx, V in enumerate(V_values):
        res = results_dict[V]
        slots = np.arange(1, len(res['queue_per_slot']) + 1)
        ax1.plot(slots, res['queue_per_slot'],
                 color=COLORS[idx], linestyle=LINESTYLES[idx % len(LINESTYLES)],
                 label=f'$V$ = {V}')

    ax1.set_xlabel('Time Slot')
    ax1.set_ylabel('Average Queue Length (packets)')
    ax1.legend(framealpha=0.9, edgecolor='none', loc='best')
    ax1.grid(True, alpha=0.3)

    # (b) 平均功率 vs 时隙
    for idx, V in enumerate(V_values):
        res = results_dict[V]
        slots = np.arange(1, len(res['power_per_slot']) + 1)
        ax2.plot(slots, res['power_per_slot'],
                 color=COLORS[idx], linestyle=LINESTYLES[idx % len(LINESTYLES)],
                 label=f'$V$ = {V}')

    ax2.set_xlabel('Time Slot')
    ax2.set_ylabel('Average Power (W)')
    ax2.legend(framealpha=0.9, edgecolor='none', loc='best')
    ax2.grid(True, alpha=0.3)

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  保存: {filepath}")
    return filepath


def plot_interference_threshold(results_dict, c_max_values, z_max_dbw_range,
                                filename='fig4_interference_threshold.png'):
    """
    Fig. 4: 不同干扰阈值和 ISL 传输限制下的平均功率
    """
    fig, ax = plt.subplots(1, 1, figsize=FIG_SINGLE)

    for idx, c_max in enumerate(c_max_values):
        powers = []
        for z_dbw in z_max_dbw_range:
            key = (c_max, z_dbw)
            if key in results_dict:
                powers.append(results_dict[key]['avg_power'])
            else:
                powers.append(np.nan)

        ax.plot(z_max_dbw_range, powers,
                color=COLORS[idx], marker=MARKERS[idx],
                linestyle=LINESTYLES[idx % len(LINESTYLES)],
                label=f'$c_{{\\max}}$ = {c_max}')

    ax.set_xlabel('Interference Threshold $Z_{\\max}$ (dBW)')
    ax.set_ylabel('Average Power (W)')
    ax.legend(framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  保存: {filepath}")
    return filepath


def plot_method_comparison(results_dict, demand_range_gbps,
                           filename='fig6_method_comparison.png'):
    """
    Fig. 6: 不同方法在不同通信需求下的平均功率 (无限存储)
    """
    fig, ax = plt.subplots(1, 1, figsize=FIG_SINGLE)

    method_names = {
        'proposed': 'Proposed',
        'drl': 'DRL for BH Pattern',
        'pre_scheduling': 'Pre-scheduling',
        'no_freq_div': 'No Freq. Division',
        'no_lb': 'Without Load Balancing',
        'max_uswg': 'Maximal USWG',
    }

    for idx, (method_key, method_label) in enumerate(method_names.items()):
        powers = []
        for demand in demand_range_gbps:
            key = (method_key, demand)
            if key in results_dict:
                powers.append(results_dict[key]['avg_power'])
            else:
                powers.append(np.nan)

        ax.plot(demand_range_gbps, powers,
                color=COLORS[idx], marker=MARKERS[idx],
                linestyle=LINESTYLES[idx % len(LINESTYLES)],
                label=method_label)

    ax.set_xlabel('Communication Demand (Gbps)')
    ax.set_ylabel('Average Power (W)')
    ax.legend(framealpha=0.9, edgecolor='none', loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  保存: {filepath}")
    return filepath
