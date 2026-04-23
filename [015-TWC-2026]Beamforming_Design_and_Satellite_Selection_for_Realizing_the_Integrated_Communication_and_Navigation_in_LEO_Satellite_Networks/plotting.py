"""
IEEE 期刊风格绘图模块
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

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

# 图尺寸
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)

# 颜色方案（高对比度，灰度可区分）
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]


def plot_fig2(C_range, results, output_dir='output'):
    """
    绘制 Fig. 2: 不同波束赋形方案的速率性能
    X: Number of UEs, Y: Sum Rate (Mbps)
    """
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    schemes = ['Proposed DC', 'MRT', 'ZF', 'MMSE', 'ST-ZF']

    for i, scheme in enumerate(schemes):
        rates = [results[c][scheme] / 1e6 for c in C_range]  # 转换为 Mbps
        ax.plot(C_range, rates, color=COLORS[i], marker=MARKERS[i],
                linestyle=LINESTYLES[i], label=scheme, linewidth=1.5, markersize=6)

    ax.set_xlabel('Number of UEs')
    ax.set_ylabel('Sum Rate (Mbps)')
    ax.set_xticks(list(C_range))
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='0.8')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(C_range[0] - 0.3, C_range[-1] + 0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_beamforming_comparison.png', dpi=300)
    plt.close()
    print(f'  Saved fig2_beamforming_comparison.png')


def plot_fig5(S_range, results, output_dir='output'):
    """
    绘制 Fig. 5: 不同卫星选择方案的速率和 GDOP 性能
    (a) Sum Rate vs Number of Available Satellites
    (b) Average GDOP vs Number of Available Satellites
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    schemes = ['Proposed (ρ=1)', 'Proposed (ρ=0.5)', 'Proposed (ρ=0)',
               'Comm-oriented', 'Nav-oriented', 'Heuristic ICAN', 'Coalitional ICAN']

    for i, scheme in enumerate(schemes):
        rates = [results[s][scheme]['rate'] / 1e6 for s in S_range]
        gdops = [results[s][scheme]['gdop'] for s in S_range]

        ax1.plot(S_range, rates, color=COLORS[i], marker=MARKERS[i],
                 linestyle=LINESTYLES[min(i, len(LINESTYLES)-1)], label=scheme,
                 linewidth=1.5, markersize=5)
        ax2.plot(S_range, gdops, color=COLORS[i], marker=MARKERS[i],
                 linestyle=LINESTYLES[min(i, len(LINESTYLES)-1)], label=scheme,
                 linewidth=1.5, markersize=5)

    ax1.set_xlabel('Number of Available Satellites')
    ax1.set_ylabel('Sum Rate (Mbps)')
    ax1.set_xticks(list(S_range))
    ax1.legend(loc='upper left', framealpha=0.9, edgecolor='0.8', fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('(a) Sum Rate', fontsize=10)

    ax2.set_xlabel('Number of Available Satellites')
    ax2.set_ylabel('Average GDOP')
    ax2.set_xticks(list(S_range))
    ax2.legend(loc='upper left', framealpha=0.9, edgecolor='0.8', fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('(b) Average GDOP', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_satellite_selection.png', dpi=300)
    plt.close()
    print(f'  Saved fig5_satellite_selection.png')


def plot_fig8(rho_range, results, output_dir='output'):
    """
    绘制 Fig. 8: 通信速率与导航 GDOP 的权衡关系
    X: Weight factor ρ
    双 Y 轴: Sum Rate 和 Average GDOP
    """
    fig, ax1 = plt.subplots(figsize=FIG_DOUBLE)
    ax2 = ax1.twinx()

    I_values = [5, 6, 7]

    for idx, I_val in enumerate(I_values):
        rates = [results[I_val][rho]['rate'] / 1e6 for rho in rho_range]
        gdops = [results[I_val][rho]['gdop'] for rho in rho_range]

        l1 = ax1.plot(rho_range, rates, color=COLORS[idx], marker=MARKERS[idx],
                       linestyle='-', label=f'Sum Rate (I={I_val})',
                       linewidth=1.5, markersize=6)
        l2 = ax2.plot(rho_range, gdops, color=COLORS[idx], marker=MARKERS[idx],
                       linestyle='--', label=f'Avg GDOP (I={I_val})',
                       linewidth=1.5, markersize=6)

    ax1.set_xlabel('Weight Factor ρ')
    ax1.set_ylabel('Sum Rate (Mbps)', color='black')
    ax2.set_ylabel('Average GDOP', color='black')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left',
               framealpha=0.9, edgecolor='0.8', fontsize=8)

    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig8_tradeoff.png', dpi=300)
    plt.close()
    print(f'  Saved fig8_tradeoff.png')
