"""
IEEE期刊风格绘图模块
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

# IEEE 单栏 3.5 inches, 双栏 7.16 inches
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)

# 颜色方案
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]


def plot_teis(t_array, elev_array, teis_db, title, filename, vmin=-15, vmax=15):
    """
    绘制TEIS热力图 (对应论文 Fig. 5)

    参数:
        t_array: 时间数组 (s)
        elev_array: 仰角数组 (度, 0=南, 90=天顶, 180=北)
        teis_db: (T, E) INR矩阵 (dB)
        title: 图标题
        filename: 保存文件名
    """
    fig, ax = plt.subplots(figsize=(7.16, 3.5))

    # 转换时间为分钟
    t_min = t_array / 60.0

    im = ax.pcolormesh(t_min, elev_array, teis_db.T,
                       cmap='jet', vmin=vmin, vmax=vmax, shading='auto')

    cbar = plt.colorbar(im, ax=ax, label='INR (dB)')
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Elevation Angle (°)')
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filename}')


def plot_inr_pdf(inr_samples_list, labels, colors, filename, title=''):
    """
    绘制INR概率密度函数 (对应论文 Fig. 9)

    参数:
        inr_samples_list: 多组INR样本列表 (每组为1D array, dB)
        labels: 对应标签列表
        colors: 对应颜色列表
        filename: 保存文件名
        title: 图标题
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for samples, label, color, marker, ls in zip(
            inr_samples_list, labels, colors, MARKERS, LINESTYLES):
        # 过滤无效值
        valid = samples[samples > -25]
        if len(valid) < 10:
            continue

        # 计算PDF
        bins = np.linspace(valid.min() - 2, valid.max() + 2, 60)
        hist, edges = np.histogram(valid, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2

        ax.plot(centers, hist, color=color, linewidth=1.5, linestyle=ls,
                label=label, marker=marker, markevery=8, markersize=5)

    ax.set_xlabel('INR (dB)')
    ax.set_ylabel('Probability Density')
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='both', direction='in')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filename}')


def plot_outage_heatmap(t_array, elev_array, outage_prob, title, filename):
    """
    绘制中断概率热力图 (对应论文 Fig. 11)
    使用imshow确保每个数据点精确映射到像素
    """
    fig, ax = plt.subplots(figsize=(7.16, 3.5))

    t_min = t_array / 60.0
    extent = [t_min[0], t_min[-1], elev_array[0], elev_array[-1]]

    im = ax.imshow(outage_prob.T, origin='lower', aspect='auto',
                   extent=extent, cmap='jet', vmin=0, vmax=1,
                   interpolation='none')

    cbar = plt.colorbar(im, ax=ax, label='Outage Probability')
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Elevation Angle (°)')
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filename}')


def plot_visible_satellites(t_array, n_vis_proposed, n_vis_stk, filename):
    """
    绘制可见卫星数量对比图 (对应论文 Fig. 4)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.16, 4), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    t_min = t_array / 60.0

    ax1.plot(t_min, n_vis_stk, 'b-', linewidth=1.0, label='STK', marker='', alpha=0.8)
    ax1.plot(t_min, n_vis_proposed, 'r--', linewidth=1.0, label='Proposed', marker='', alpha=0.8)
    ax1.set_ylabel('Number of Visible Satellites')
    ax1.legend(fontsize=9, framealpha=0.9, edgecolor='none')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='both', direction='in')

    diff = np.array(n_vis_proposed) - np.array(n_vis_stk)
    ax2.plot(t_min, diff, 'k-', linewidth=0.8)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Difference')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='both', direction='in')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filename}')
