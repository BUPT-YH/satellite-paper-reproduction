"""
绘图模块 — IEEE 期刊标准风格
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

FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)

# 颜色与标记
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]


def plot_fig4(results, output_dir='output'):
    """
    绘制 Fig. 4: 基线 INR CDF
    不同波束数和切换策略下主用户 INR 的 CDF
    """
    fig, ax = plt.subplots(1, 1, figsize=FIG_DOUBLE)

    plot_idx = 0
    # 选择关键配置进行绘制
    configs = [
        ('NB8_PHE_SHE', '$N_B=8$, HE/HE', '-'),
        ('NB16_PHE_SHE', '$N_B=16$, HE/HE', '-'),
        ('NB24_PHE_SHE', '$N_B=24$, HE/HE', '-'),
        ('NB32_PHE_SHE', '$N_B=32$, HE/HE', '-'),
        ('NB16_PHE_SMCT', '$N_B=16$, HE/MCT', '--'),
        ('NB24_PHE_SMCT', '$N_B=24$, HE/MCT', '--'),
    ]

    for key, label, ls in configs:
        if key not in results:
            continue
        data = results[key]
        valid = data[data > -900]
        if len(valid) == 0:
            continue

        sorted_data = np.sort(valid)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        color = COLORS[plot_idx % len(COLORS)]
        marker = MARKERS[plot_idx % len(MARKERS)]

        ax.plot(sorted_data, cdf, linestyle=ls, color=color,
                marker=marker, markevery=max(1, len(sorted_data) // 10),
                label=label, linewidth=1.5, markersize=5)
        plot_idx += 1

    # ITU阈值参考线
    ax.axvline(x=-12.2, color='k', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(-11.5, 0.15, 'ITU\n$-12.2$ dB', fontsize=7, color='k')

    ax.set_xlabel('INR (dB)')
    ax.set_ylabel('CDF')
    ax.set_xlim([-40, 15])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='0.8')
    ax.set_title('Fig. 4: CDF of INR (Baseline, No Protection)')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig4_baseline_inr_cdf.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filepath}")
    return filepath


def plot_fig5(results, output_dir='output'):
    """
    绘制 Fig. 5: 提出方案下 INR CDF
    不同 INR_max_th 下的主用户 INR CDF
    """
    fig, ax = plt.subplots(1, 1, figsize=FIG_DOUBLE)

    configs = [
        ('INRmax_-6', '$\\mathrm{INR}^{\\max}_{th}=-6$ dB'),
        ('INRmax_0', '$\\mathrm{INR}^{\\max}_{th}=0$ dB'),
        ('INRmax_3', '$\\mathrm{INR}^{\\max}_{th}=3$ dB'),
        ('INRmax_inf', '$\\mathrm{INR}^{\\max}_{th}=\\infty$'),
    ]

    for plot_idx, (key, label) in enumerate(configs):
        if key not in results:
            continue
        data = results[key]['inr']
        valid = data[data > -900]
        if len(valid) == 0:
            continue

        sorted_data = np.sort(valid)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        color = COLORS[plot_idx % len(COLORS)]
        marker = MARKERS[plot_idx % len(MARKERS)]
        ls = LINESTYLES[plot_idx % len(LINESTYLES)]

        ax.plot(sorted_data, cdf, linestyle=ls, color=color,
                marker=marker, markevery=max(1, len(sorted_data) // 10),
                label=label, linewidth=1.5, markersize=5)

    # 参考线
    ax.axvline(x=-6, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(-5.5, 0.1, '$\\mathrm{INR}_{th}=-6$ dB', fontsize=7, color='gray')
    ax.axvline(x=-12.2, color='k', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.text(-11.5, 0.05, 'ITU\n$-12.2$ dB', fontsize=7, color='k')

    ax.set_xlabel('INR (dB)')
    ax.set_ylabel('CDF')
    ax.set_xlim([-40, 15])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='0.8')
    ax.set_title('Fig. 5: CDF of INR (Proposed, $\\mathrm{INR}_{th}=-6$ dB, $N_B=16$)')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'fig5_proposed_inr_cdf.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filepath}")
    return filepath
