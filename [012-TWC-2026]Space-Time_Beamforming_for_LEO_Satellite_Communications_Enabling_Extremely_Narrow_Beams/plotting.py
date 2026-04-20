"""
绘图模块 — IEEE 期刊仿真图标准风格
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

# IEEE 尺寸
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)

# 颜色和标记
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]


def plot_fig5(results, output_dir='output'):
    """
    Fig. 5: 部分连接网络 — SE vs P (K=3)
    实线：完美 CSIT，虚线：有延迟误差
    """
    P = results['P_dBm']
    fig, ax = plt.subplots(1, 1, figsize=FIG_DOUBLE)

    methods = ['MRT', 'ZF', 'SLNR', 'TDMA', 'ST-ZF']
    labels = ['MRT', 'ZF', 'SLNR (MMSE)', 'TDMA', 'ST-ZF']

    for i, (method, label) in enumerate(zip(methods, labels)):
        # 完美 CSIT（实线）
        ax.plot(P, results['perfect'][method],
                color=COLORS[i], marker=MARKERS[i], linestyle='-',
                label=label, linewidth=1.5, markersize=5)
        # 延迟误差（虚线）
        ax.plot(P, results['imperfect'][method],
                color=COLORS[i], marker=MARKERS[i], linestyle='--',
                linewidth=1.2, markersize=4, alpha=0.7)

    ax.set_xlabel('Transmit Power $P$ (dBm)')
    ax.set_ylabel('Ergodic Sum Spectral Efficiency (bps/Hz)')
    ax.set_title('Partially Connected Network ($K=3$)')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([20, 50])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_partial_se_vs_P.png', dpi=300)
    plt.close()
    print(f"Saved fig5_partial_se_vs_P.png")


def plot_fig7(results, output_dir='output'):
    """
    Fig. 7: 全连接网络 — SE vs P (M=3, K=4)
    实线：完美 CSIT，虚线：有延迟误差
    """
    P = results['P_dBm']
    fig, ax = plt.subplots(1, 1, figsize=FIG_DOUBLE)

    methods = ['MRT', 'SLNR', 'TDMA', 'ST-SLNR']
    labels = ['MRT', 'SLNR (MMSE)', 'TDMA', 'ST-SLNR']

    for i, (method, label) in enumerate(zip(methods, labels)):
        ax.plot(P, results['perfect'][method],
                color=COLORS[i], marker=MARKERS[i], linestyle='-',
                label=label, linewidth=1.5, markersize=5)
        ax.plot(P, results['imperfect'][method],
                color=COLORS[i], marker=MARKERS[i], linestyle='--',
                linewidth=1.2, markersize=4, alpha=0.7)

    ax.set_xlabel('Transmit Power $P$ (dBm)')
    ax.set_ylabel('Ergodic Sum Spectral Efficiency (bps/Hz)')
    ax.set_title('Fully Connected Network ($M=3$, $K=4$)')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([20, 50])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig7_full_se_vs_P.png', dpi=300)
    plt.close()
    print(f"Saved fig7_full_se_vs_P.png")


def plot_fig8(results, K_range, M_range, output_dir='output'):
    """
    Fig. 8: 全连接网络 — ST-SLNR SE vs K 和 M (P=40 dBm)
    """
    fig, ax = plt.subplots(1, 1, figsize=FIG_DOUBLE)

    for i, M in enumerate(M_range):
        se_values = [results[K][M] for K in K_range]
        ax.plot(K_range, se_values,
                color=COLORS[i], marker=MARKERS[i], linestyle=LINESTYLES[i],
                label=f'$M={M}$', linewidth=1.5, markersize=6)

    ax.set_xlabel('Number of Users $K$')
    ax.set_ylabel('Ergodic Sum Spectral Efficiency (bps/Hz)')
    ax.set_title('ST-SLNR: Sum SE vs $K$ and $M$ ($P=40$ dBm)')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_range)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig8_full_se_vs_K_M.png', dpi=300)
    plt.close()
    print(f"Saved fig8_full_se_vs_K_M.png")
