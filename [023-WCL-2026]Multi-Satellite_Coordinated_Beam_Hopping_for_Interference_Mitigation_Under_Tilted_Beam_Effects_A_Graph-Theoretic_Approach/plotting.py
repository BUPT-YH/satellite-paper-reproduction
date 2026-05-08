"""
IEEE期刊风格绘图模块
生成Fig.4(a)(b)(c)和Table I
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# ===== IEEE期刊风格全局配置 =====
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05
rcParams['axes.linewidth'] = 0.8
rcParams['lines.linewidth'] = 1.5
rcParams['lines.markersize'] = 5
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True
rcParams['grid.linewidth'] = 0.3
rcParams['grid.alpha'] = 0.3

# 颜色和标记
COLORS = {
    'MCMF-TS-GC': '#0072BD',
    'Gurobi': '#A2142F',
    'NITB': '#EDB120',
    'WMIS': '#7E2F8E',
    'Greedy': '#77AC30',
}
MARKERS = {
    'MCMF-TS-GC': 'o',
    'Gurobi': 'D',
    'NITB': '^',
    'WMIS': 's',
    'Greedy': 'v',
}
LINESTYLES = {
    'MCMF-TS-GC': '-',
    'Gurobi': '--',
    'NITB': '-.',
    'WMIS': ':',
    'Greedy': (0, (3, 1, 1, 1)),
}

FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)


def plot_satisfaction_rate(results1, results2, T_range, save_path='output/fig4a_satisfaction_rate.png'):
    """
    Fig. 4(a): 服务满足率 vs BH周期T
    两个子图：Case 1 和 Case 2
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    methods = ['gurobi', 'mcmf_ts_gc', 'nitb', 'wmis', 'greedy']
    labels = ['Gurobi', 'MCMF-TS-GC', 'NITB', 'WMIS', 'Greedy']

    import matplotlib.lines as mlines
    handles = []

    for method, label in zip(methods, labels):
        # Case 1
        y1 = [results1[method]['sat_rate'].get(T, 0) for T in T_range]
        ax1.plot(T_range, y1, color=COLORS[label], marker=MARKERS[label],
                 linestyle=LINESTYLES[label], label=label, markersize=5)
        # Case 2
        y2 = [results2[method]['sat_rate'].get(T, 0) for T in T_range]
        ax2.plot(T_range, y2, color=COLORS[label], marker=MARKERS[label],
                 linestyle=LINESTYLES[label], label=label, markersize=5)
        # 显式创建图例句柄
        handles.append(mlines.Line2D([], [], color=COLORS[label], marker=MARKERS[label],
                       linestyle=LINESTYLES[label], label=label, markersize=5))

    for ax, title in [(ax1, 'Case 1: C=148'), (ax2, 'Case 2: C=928')]:
        ax.set_xlabel('BH Period $T$')
        ax.set_ylabel('Service Satisfaction Rate (%)')
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.set_xticks(T_range)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=handles, loc='lower right', framealpha=0.9, edgecolor='none')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_min_sinr(results1, results2, T_range, save_path='output/fig4b_min_sinr.png'):
    """
    Fig. 4(b): 最低SINR vs BH周期T
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    import matplotlib.lines as mlines
    methods = ['gurobi', 'mcmf_ts_gc', 'nitb', 'wmis', 'greedy']
    labels = ['Gurobi', 'MCMF-TS-GC', 'NITB', 'WMIS', 'Greedy']
    handles = []

    for method, label in zip(methods, labels):
        y1 = [results1[method]['min_sinr'].get(T, -5) for T in T_range]
        ax1.plot(T_range, y1, color=COLORS[label], marker=MARKERS[label],
                 linestyle=LINESTYLES[label], label=label, markersize=5)
        y2 = [results2[method]['min_sinr'].get(T, -5) for T in T_range]
        ax2.plot(T_range, y2, color=COLORS[label], marker=MARKERS[label],
                 linestyle=LINESTYLES[label], label=label, markersize=5)
        handles.append(mlines.Line2D([], [], color=COLORS[label], marker=MARKERS[label],
                       linestyle=LINESTYLES[label], label=label, markersize=5))

    for ax, title in [(ax1, 'Case 1: C=148'), (ax2, 'Case 2: C=928')]:
        ax.set_xlabel('BH Period $T$')
        ax.set_ylabel('Minimum SINR (dB)')
        ax.set_title(title)
        ax.set_xticks(T_range)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=handles, loc='lower right', framealpha=0.9, edgecolor='none')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_beam_visualization(power_map, xx, yy, mask, cell_centers, cell_radius,
                            s_assign, t_assign, slot_idx,
                            save_path='output/fig4c_beam_pattern.png'):
    """
    Fig. 4(c): 地面接收功率可视化
    """
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))

    # 绘制功率热力图
    im = ax.pcolormesh(xx, yy, power_map, cmap='jet', shading='auto',
                       vmin=-140, vmax=-100)

    # 标注小区边界
    for c in range(len(cell_centers)):
        circle = plt.Circle(cell_centers[c], cell_radius, fill=False,
                           edgecolor='gray', linewidth=0.3, alpha=0.5)
        ax.add_patch(circle)

    # 标注被服务的小区
    active_cells = np.where(t_assign == slot_idx)[0]
    for c in active_cells:
        circle = plt.Circle(cell_centers[c], cell_radius, fill=False,
                           edgecolor='white', linewidth=0.8)
        ax.add_patch(circle)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title(f'BH Slot {slot_idx+1} - Receive Power (dBW)')
    ax.set_aspect('equal')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Receive Power (dBW)')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def print_runtime_table(results1, results2, T_range):
    """
    打印 Table I: 不同BH方法的平均运行时间
    """
    print("\n" + "="*60)
    print("TABLE I: Average Runtime (seconds)")
    print("="*60)
    print(f"{'Method':<15} {'Case1 (C=148)':<18} {'Case2 (C=928)':<18}")
    print("-"*51)

    methods = [('MCMF-TS-GC', 'mcmf_ts_gc'), ('WMIS', 'wmis'),
               ('Greedy', 'greedy'), ('NITB', 'nitb')]

    for label, key in methods:
        t1 = np.mean([results1[key]['time'].get(T, 0) for T in T_range])
        t2 = np.mean([results2[key]['time'].get(T, 0) for T in T_range])
        print(f"{label:<15} {t1:<18.3f} {t2:<18.3f}")

    print("="*60)
