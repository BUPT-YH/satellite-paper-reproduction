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

# IEEE 尺寸
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)

# 颜色和标记
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]


def plot_reduced_scenario(uc_data, ec_data, tts_data, user_counts, output_dir):
    """
    Fig. 6: 缩减场景对比
    uc_data, ec_data, tts_data: dict {method: [values per user count]}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    methods = list(uc_data.keys())
    x = np.arange(len(user_counts))

    # (a) UC 和 EC
    for idx, method in enumerate(methods):
        ax1.plot(x, uc_data[method], color=COLORS[idx], marker=MARKERS[idx],
                linestyle='-', label=f'UC - {method}', markersize=5)
    for idx, method in enumerate(methods):
        ax1.plot(x, ec_data[method], color=COLORS[idx], marker=MARKERS[idx],
                linestyle='--', label=f'EC - {method}', markersize=5, alpha=0.7)

    ax1.set_xlabel('Number of Users')
    ax1.set_ylabel('Percentage of Requested Capacity (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(user_counts)
    ax1.legend(fontsize=7, loc='upper left', framealpha=0.9, edgecolor='none')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('(a) Traffic-based PoIs: UC and EC')
    ax1.set_ylim(bottom=0)

    # (b) TTS
    for idx, method in enumerate(methods):
        ax2.plot(x, tts_data[method], color=COLORS[idx], marker=MARKERS[idx],
                linestyle=LINESTYLES[idx], label=method, markersize=5)

    ax2.set_xlabel('Number of Users')
    ax2.set_ylabel('Average TTS (time slots)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(user_counts)
    ax2.legend(fontsize=8, loc='upper left', framealpha=0.9, edgecolor='none')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('(b) Time-based PoI: TTS')
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_reduced_scenario.png'))
    plt.close()
    print(f"  Saved fig6_reduced_scenario.png")


def plot_enlarged_scenario(uc_data, ec_data, tts_data, user_counts, output_dir):
    """
    Fig. 8: 扩展场景对比
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    methods = list(uc_data.keys())
    x = np.arange(len(user_counts))

    # (a) UC 和 EC
    for idx, method in enumerate(methods):
        ax1.plot(x, uc_data[method], color=COLORS[idx], marker=MARKERS[idx],
                linestyle='-', label=f'UC - {method}', markersize=5)
    for idx, method in enumerate(methods):
        ax1.plot(x, ec_data[method], color=COLORS[idx], marker=MARKERS[idx],
                linestyle='--', label=f'EC - {method}', markersize=5, alpha=0.7)

    ax1.set_xlabel('Number of Users')
    ax1.set_ylabel('Percentage of Requested Capacity (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(user_counts)
    ax1.legend(fontsize=7, loc='upper left', framealpha=0.9, edgecolor='none')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('(a) Traffic-based PoIs: UC and EC')
    ax1.set_ylim(bottom=0)

    # (b) TTS
    for idx, method in enumerate(methods):
        ax2.plot(x, tts_data[method], color=COLORS[idx], marker=MARKERS[idx],
                linestyle=LINESTYLES[idx], label=method, markersize=5)

    ax2.set_xlabel('Number of Users')
    ax2.set_ylabel('Average TTS (time slots)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(user_counts)
    ax2.legend(fontsize=8, loc='upper left', framealpha=0.9, edgecolor='none')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('(b) Time-based PoI: TTS')
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig8_enlarged_scenario.png'))
    plt.close()
    print(f"  Saved fig8_enlarged_scenario.png")


def plot_weighting_study(beta_values, uc_vals, ec_vals, tts_vals, output_dir):
    """
    Fig. 5: 权重参数评估
    """
    fig, ax = plt.subplots(1, 1, figsize=FIG_SINGLE)

    ax.plot(beta_values, uc_vals, color=COLORS[0], marker=MARKERS[0],
            linestyle='-', label='UC', markersize=4)
    ax.plot(beta_values, ec_vals, color=COLORS[1], marker=MARKERS[1],
            linestyle='-', label='EC', markersize=4)
    ax.plot(beta_values, tts_vals, color=COLORS[2], marker=MARKERS[2],
            linestyle='-', label='TTS', markersize=4)

    # 标注 β = 0.7 的位置
    idx_07 = np.argmin(np.abs(np.array(beta_values) - 0.7))
    ax.axvline(x=0.7, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_xlabel(r'Weighting parameter $\beta$')
    ax.set_ylabel('Normalized KPI value')
    ax.legend(fontsize=8, loc='center right', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_weighting_study.png'))
    plt.close()
    print(f"  Saved fig5_weighting_study.png")
