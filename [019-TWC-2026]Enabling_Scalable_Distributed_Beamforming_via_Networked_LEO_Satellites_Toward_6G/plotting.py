"""
绘图模块 — IEEE期刊风格
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
rcParams['xtick.minor.width'] = 0.5
rcParams['ytick.minor.width'] = 0.5
rcParams['grid.linewidth'] = 0.3
rcParams['grid.alpha'] = 0.3

FIG_DOUBLE = (7.16, 3.5)

# 颜色与标记方案
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '-', '-', '--', '--', '-.', ':']


def plot_fig9(Ps_range_dbm, rates_dict, output_dir):
    """
    Fig. 9: Sum rate vs power budget Ps
    rates_dict: {scheme_name: [rates]}
    """
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    for i, (name, rates) in enumerate(rates_dict.items()):
        Ps_dbm = np.array(Ps_range_dbm)
        ax.plot(Ps_dbm, rates,
                color=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                label=name, markersize=5)

    ax.set_xlabel('Power Budget $P_s$ (dBm)')
    ax.set_ylabel('Sum Rate (bps/Hz)')
    ax.set_title('Sum Rate vs. Power Budget')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='0.8')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(Ps_range_dbm[0], Ps_range_dbm[-1])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig9_sum_rate_vs_power.png', dpi=300)
    plt.close()
    print(f"  Saved fig9_sum_rate_vs_power.png")


def plot_fig10(N_range, rates_dict, output_dir):
    """
    Fig. 10: Sum rate vs antenna number N
    """
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    for i, (name, rates) in enumerate(rates_dict.items()):
        ax.plot(N_range, rates,
                color=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                label=name, markersize=5)

    ax.set_xlabel('Number of Antennas $N$')
    ax.set_ylabel('Sum Rate (bps/Hz)')
    ax.set_title('Sum Rate vs. Antenna Number')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='0.8')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(N_range)
    ax.set_xticklabels([str(n) for n in N_range])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig10_sum_rate_vs_antenna.png', dpi=300)
    plt.close()
    print(f"  Saved fig10_sum_rate_vs_antenna.png")


def plot_fig12(S_range, rates_dict, output_dir):
    """
    Fig. 12: Sum rate vs satellite number S
    """
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    for i, (name, rates) in enumerate(rates_dict.items()):
        ax.plot(S_range, rates,
                color=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                label=name, markersize=5)

    ax.set_xlabel('Number of LEO Satellites $S$')
    ax.set_ylabel('Sum Rate (bps/Hz)')
    ax.set_title('Sum Rate vs. Satellite Number')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='0.8')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(S_range[0], S_range[-1])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig12_sum_rate_vs_satellite.png', dpi=300)
    plt.close()
    print(f"  Saved fig12_sum_rate_vs_satellite.png")
