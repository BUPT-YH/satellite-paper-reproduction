# -*- coding: utf-8 -*-
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

# IEEE 单栏 3.5 inches，双栏 7.16 inches
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 4.0)

# 颜色方案（高对比度，灰度可区分）
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30',
          '#4DBEEE', '#A2142F', '#008080']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', 'X']
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']


def plot_min_sinr(x_data, results_dict, xlabel, title, filename,
                  x_label_unit='', figsize=None, ylim=None, ncol=2):
    """
    绘制最小 SINR 对比图 (Fig 2-10 通用)

    Parameters:
        x_data: x 轴数据数组
        results_dict: {scheme_name: sinr_array} 各方案的 SINR 结果
        xlabel: x 轴标签
        title: 图标题
        filename: 输出文件名
        x_label_unit: x 轴单位
        figsize: 图尺寸
        ylim: y 轴范围
        ncol: 图例列数
    """
    if figsize is None:
        figsize = FIG_DOUBLE
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for idx, (name, sinr_vals) in enumerate(results_dict.items()):
        ax.plot(x_data, sinr_vals,
                color=COLORS[idx % len(COLORS)],
                marker=MARKERS[idx % len(MARKERS)],
                linestyle=LINESTYLES[idx % len(LINESTYLES)],
                label=name, markersize=5)
    ax.set_xlabel(xlabel + x_label_unit)
    ax.set_ylabel('Minimum SINR $\\gamma$ (dB)')
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9, edgecolor='gray', ncol=ncol)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(f'output/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: output/{filename}')


def plot_sum_rate(x_data, results_dict, xlabel, title, filename,
                  x_label_unit='', figsize=None, ylim=None):
    """绘制总可达速率对比图"""
    if figsize is None:
        figsize = FIG_DOUBLE
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for idx, (name, rate_vals) in enumerate(results_dict.items()):
        ax.plot(x_data, rate_vals,
                color=COLORS[idx % len(COLORS)],
                marker=MARKERS[idx % len(MARKERS)],
                linestyle=LINESTYLES[idx % len(LINESTYLES)],
                label=name, markersize=5)
    ax.set_xlabel(xlabel + x_label_unit)
    ax.set_ylabel('Sum Achievable Rate (bps/Hz)')
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(f'output/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: output/{filename}')


def plot_execution_time(categories, time_data, title, filename, figsize=None):
    """绘制执行时间对比柱状图 (Fig 13)"""
    if figsize is None:
        figsize = FIG_DOUBLE
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = np.arange(len(categories))
    width = 0.3
    n_bars = len(time_data)
    for i, (label, times) in enumerate(time_data.items()):
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x + offset, times, width, label=label,
               color=COLORS[i % len(COLORS)], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (s)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'output/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: output/{filename}')
