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

# 图尺寸
FIG_DOUBLE = (7.16, 3.2)   # 双栏图

# 颜色与标记
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']
MARKERS = ['o', 's', '^', 'D', 'v']
LINESTYLES = ['-', '--', '-.', ':', '-']

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


def plot_fig5(terr_results, sat_results, user_range):
    """
    绘制 Fig. 5: 不同用户数下干扰小区的平均服务容量
    (a) 地面小区 (b) 卫星小区
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    schemes = ['Interfered', 'Fixed Freq Div', 'DSS', 'BHSS']
    labels = ['Interfered', 'Fixed Freq Div', 'DSS', 'BHSS (Proposed)']

    for i, (scheme, label) in enumerate(zip(schemes, labels)):
        ax1.plot(user_range, terr_results[scheme],
                 color=COLORS[i], marker=MARKERS[i], linestyle=LINESTYLES[i],
                 label=label, markersize=4, markerfacecolor='none')
        ax2.plot(user_range, sat_results[scheme],
                 color=COLORS[i], marker=MARKERS[i], linestyle=LINESTYLES[i],
                 label=label, markersize=4, markerfacecolor='none')

    # (a) 地面小区
    ax1.set_xlabel('Total Number of Users')
    ax1.set_ylabel('Average Service Capacity (Mbps)')
    ax1.set_title('(a) Terrestrial Cells', fontsize=10)
    ax1.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(user_range[0], user_range[-1])

    # (b) 卫星小区
    ax2.set_xlabel('Total Number of Users')
    ax2.set_ylabel('Average Service Capacity (Mbps)')
    ax2.set_title('(b) Satellite Cells', fontsize=10)
    ax2.legend(loc='upper left', framealpha=0.9, edgecolor='gray')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(user_range[0], user_range[-1])

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig5_service_capacity.png')
    plt.savefig(out_path)
    plt.close()
    print(f'Saved: {out_path}')
    return out_path


def plot_fig6(terr_eff, sat_eff, T_range_ms):
    """
    绘制 Fig. 6: 不同时隙长度下的时间同步效率
    (a) 地面小区 (b) 卫星小区
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    methods = ['Ideal', 'Proposed', 'Timeslot-based', 'General Sync', 'Terr-prior']
    labels = ['Ideal', 'Proposed', 'Timeslot-based', 'General Sync', 'Terr-prior']

    for i, (method, label) in enumerate(zip(methods, labels)):
        ax1.plot(T_range_ms, [e * 100 for e in terr_eff[method]],
                 color=COLORS[i], marker=MARKERS[i], linestyle=LINESTYLES[i],
                 label=label, markersize=4, markerfacecolor='none')
        ax2.plot(T_range_ms, [e * 100 for e in sat_eff[method]],
                 color=COLORS[i], marker=MARKERS[i], linestyle=LINESTYLES[i],
                 label=label, markersize=4, markerfacecolor='none')

    # (a) 地面小区
    ax1.set_xlabel('Timeslot Length (ms)')
    ax1.set_ylabel('Time Synchronization Efficiency (%)')
    ax1.set_title('(a) Terrestrial Cells', fontsize=10)
    ax1.legend(loc='best', framealpha=0.9, edgecolor='gray', fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(T_range_ms[0], T_range_ms[-1])
    ax1.set_ylim(50, 105)

    # (b) 卫星小区
    ax2.set_xlabel('Timeslot Length (ms)')
    ax2.set_ylabel('Time Synchronization Efficiency (%)')
    ax2.set_title('(b) Satellite Cells', fontsize=10)
    ax2.legend(loc='best', framealpha=0.9, edgecolor='gray', fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(T_range_ms[0], T_range_ms[-1])
    ax2.set_ylim(50, 105)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig6_time_sync_efficiency.png')
    plt.savefig(out_path)
    plt.close()
    print(f'Saved: {out_path}')
    return out_path
