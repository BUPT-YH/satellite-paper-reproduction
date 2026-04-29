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

# IEEE 单栏 3.5 inches, 双栏 7.16 inches
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)

# 颜色和标记方案
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]


def plot_fig5(data, output_dir='output'):
    """
    Fig. 5: GU通信时间对比
    6条柱状图 (3直接 + 3网关辅助), 每组20个GU
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE, sharey=True)

    gu_idx = data['gu_indices'] + 1  # 1-indexed
    width = 0.25
    x = np.arange(len(gu_idx))

    # 左图: 直接通信
    bars1 = ax1.bar(x - width, data['times_di_dwoa'], width,
                    label='DI-DWOA', color=COLORS[0], alpha=0.85)
    bars2 = ax1.bar(x, data['times_di_gre'], width,
                    label='DI-GRE', color=COLORS[1], alpha=0.85)
    bars3 = ax1.bar(x + width, data['times_di_pri'], width,
                    label='DI-PRI', color=COLORS[2], alpha=0.85)

    ax1.set_xlabel('Ground User Index')
    ax1.set_ylabel('Communication Time (s)')
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(gu_idx[::2])
    ax1.legend(loc='upper left', framealpha=0.9, edgecolor='none')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_title('Direct Communication')

    # 右图: 网关辅助
    bars4 = ax2.bar(x - width, data['times_ga_dwoa'], width,
                    label='GA-DWOA', color=COLORS[3], alpha=0.85)
    bars5 = ax2.bar(x, data['times_ga_gre'], width,
                    label='GA-GRE', color=COLORS[4], alpha=0.85)
    bars6 = ax2.bar(x + width, data['times_ga_pri'], width,
                    label='GA-PRI', color=COLORS[5], alpha=0.85)

    ax2.set_xlabel('Ground User Index')
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels(gu_idx[::2])
    ax2.legend(loc='upper left', framealpha=0.9, edgecolor='none')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_title('Gateway-Assisted')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_communication_time.png', dpi=300)
    plt.close()
    print(f"  Fig. 5 saved to {output_dir}/fig5_communication_time.png")

    # 打印平均延迟对比
    print(f"  GA-DWOA avg: {np.mean(data['times_ga_dwoa']):.4f}s")
    print(f"  GA-GRE avg: {np.mean(data['times_ga_gre']):.4f}s")
    print(f"  GA-PRI avg: {np.mean(data['times_ga_pri']):.4f}s")
    print(f"  DI-DWOA avg: {np.mean(data['times_di_dwoa']):.4f}s")


def plot_fig6(snr_direct, snr_gateway, assisted_indices, output_dir='output'):
    """
    Fig. 6: 各GU的信道质量(SNR)对比
    两组柱状图: 直接通信 vs 网关辅助
    """
    num_gus = len(snr_direct)
    gu_idx = np.arange(1, num_gus + 1)

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    width = 0.35
    x = np.arange(num_gus)

    bars1 = ax.bar(x - width/2, snr_direct, width,
                   label='Direct Communication', color=COLORS[0], alpha=0.85)
    bars2 = ax.bar(x + width/2, snr_gateway, width,
                   label='Gateway-Assisted', color=COLORS[1], alpha=0.85)

    ax.set_xlabel('Ground User Index')
    ax.set_ylabel('SNR (linear)')
    ax.set_xticks(x)
    ax.set_xticklabels(gu_idx)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig6_channel_quality.png', dpi=300)
    plt.close()
    print(f"  Fig. 6 saved to {output_dir}/fig6_channel_quality.png")


def plot_fig7(latency_results, output_dir='output'):
    """
    Fig. 7: 平均传输延迟 vs GU数量 (不同网关辅助比例)
    """
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    for i, (frac, data) in enumerate(latency_results.items()):
        label = f'{int(frac*100)}% Gateway-Assisted'
        ax.plot(data['gu_counts'], data['latencies'],
                color=COLORS[i], marker=MARKERS[i], linestyle=LINESTYLES[i],
                label=label, linewidth=1.5, markersize=6)

    ax.set_xlabel('Number of Ground Users')
    ax.set_ylabel('Average Transmission Latency (s)')
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig7_latency_vs_gu_count.png', dpi=300)
    plt.close()
    print(f"  Fig. 7 saved to {output_dir}/fig7_latency_vs_gu_count.png")


def plot_fig9(psnr_results, output_dir='output'):
    """
    Fig. 9: 不同语义压缩率下的PSNR vs SNR
    """
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    for i, (name, data) in enumerate(psnr_results.items()):
        ax.plot(data['snr'], data['psnr'],
                color=COLORS[i], marker=MARKERS[i], linestyle=LINESTYLES[i],
                label=f'SCR = {name}', linewidth=1.5, markersize=4,
                markevery=4)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('PSNR (dB)')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 19)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig9_psnr_vs_snr.png', dpi=300)
    plt.close()
    print(f"  Fig. 9 saved to {output_dir}/fig9_psnr_vs_snr.png")
