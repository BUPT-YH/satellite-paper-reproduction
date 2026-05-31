"""
plotting.py — IEEE期刊风格绘图模块
论文: Co-Existence Analysis of Terrestrial and Non-Terrestrial Networks
      in S-Band Using Stochastic Geometry (IEEE TCOM 2026)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============================================================
# IEEE 期刊风格全局配置
# ============================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
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
rcParams['grid.linewidth'] = 0.3
rcParams['grid.alpha'] = 0.3

# ============================================================
# 图表尺寸
# ============================================================
FIG_SINGLE = (3.5, 2.8)      # 单栏图
FIG_DOUBLE = (7.16, 3.5)     # 双栏图
FIG_DOUBLE_WIDE = (7.16, 4.5)

# ============================================================
# 颜色和标记
# ============================================================
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]


def setup_axis(ax, xlabel, ylabel, title=None, grid=True):
    """
    配置坐标轴样式
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, which='both', linewidth=0.3, alpha=0.3)
    ax.tick_params(which='both', direction='in', width=0.8)


def save_fig(fig, filepath, tight=True):
    """
    保存图表
    """
    fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"  图表已保存: {filepath}")
