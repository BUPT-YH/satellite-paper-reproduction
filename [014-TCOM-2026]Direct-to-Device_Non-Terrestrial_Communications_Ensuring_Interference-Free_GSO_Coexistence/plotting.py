"""
论文复现 - IEEE 期刊风格绘图模块
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

# IEEE 单栏 3.5 inches，双栏 7.16 inches
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)

# 颜色方案（高对比度，灰度可区分）
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]


def plot_fig4a(time_array, num_fz_sats, save_path='output/fig4a_fz_satellites.png'):
    """
    Fig. 4(a): 用户位置处禁区卫星数量随时间变化
    """
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    ax.plot(time_array / 60, num_fz_sats, color=COLORS[0], linewidth=1.2)
    ax.fill_between(time_array / 60, 0, num_fz_sats, alpha=0.15, color=COLORS[0])
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Number of Satellites with FZ at User')
    ax.set_ylim(bottom=0)
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(True, alpha=0.3)
    ax.set_title('(a)', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'  Saved {save_path}')
    return save_path


def plot_fig4b(cone_angles, percentages_low_alt, percentages_high_alt,
               save_path='output/fig4b_fz_percentage.png'):
    """
    Fig. 4(b): 不同锥角和高度下用户落入禁区的百分比
    """
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    ax.plot(cone_angles, percentages_low_alt, color=COLORS[0], marker=MARKERS[0],
            linestyle=LINESTYLES[0], label='550/540 km', markersize=6)
    ax.plot(cone_angles, percentages_high_alt, color=COLORS[1], marker=MARKERS[1],
            linestyle=LINESTYLES[0], label='1150/1140 km', markersize=6)

    ax.set_xlabel('Cone Angle (°)')
    ax.set_ylabel('Percentage of Time in FZ (%)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')
    ax.set_title('(b)', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'  Saved {save_path}')
    return save_path


def plot_fig6(epfd_thresholds, ccdf_homn, ccdf_evmn, ccdf_lnmx,
              se_bins, cdf_se_homn, cdf_se_evmn, cdf_se_lnmx,
              save_path='output/fig6_epfd_se.png'):
    """
    Fig. 6: EPFD CCDF (左轴) + 频谱效率 CDF (右轴) 双子图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # (a) EPFD CCDF
    ax1.semilogy(epfd_thresholds, ccdf_homn, color=COLORS[0], marker=MARKERS[0],
                 linestyle=LINESTYLES[0], label='HOMN', markevery=5, markersize=5)
    ax1.semilogy(epfd_thresholds, ccdf_evmn, color=COLORS[1], marker=MARKERS[1],
                 linestyle=LINESTYLES[0], label='Evmn', markevery=5, markersize=5)
    ax1.semilogy(epfd_thresholds, ccdf_lnmx, color=COLORS[2], marker=MARKERS[2],
                 linestyle=LINESTYLES[0], label='Lnmx', markevery=5, markersize=5)

    ax1.set_xlabel('EPFD (dB(W/m²/100MHz))')
    ax1.set_ylabel('CCDF')
    ax1.set_ylim(1e-3, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='none')
    ax1.set_title('(a)', fontsize=10)

    # (b) Spectral Efficiency CDF
    ax2.plot(se_bins, cdf_se_homn, color=COLORS[0], marker=MARKERS[0],
             linestyle=LINESTYLES[0], label='HOMN', markevery=5, markersize=5)
    ax2.plot(se_bins, cdf_se_evmn, color=COLORS[1], marker=MARKERS[1],
             linestyle=LINESTYLES[0], label='Evmn', markevery=5, markersize=5)
    ax2.plot(se_bins, cdf_se_lnmx, color=COLORS[2], marker=MARKERS[2],
             linestyle=LINESTYLES[0], label='Lnmx', markevery=5, markersize=5)

    ax2.set_xlabel('Spectral Efficiency (bps/Hz)')
    ax2.set_ylabel('CDF')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', framealpha=0.9, edgecolor='none')
    ax2.set_title('(b)', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'  Saved {save_path}')
    return save_path
