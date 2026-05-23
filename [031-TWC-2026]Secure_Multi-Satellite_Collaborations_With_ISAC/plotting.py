"""
绘图模块 — IEEE 期刊标准风格
论文: Secure Multi-Satellite Collaborations With ISAC
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============================================================
# IEEE 期刊风格设置
# ============================================================
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

# ============================================================
# 样式常量
# ============================================================
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)

# 9 种算法的配色 (3 分配 x 3 BF)
# 分配方法用颜色区分, BF 方法用线型和标记区分
COLORS = {
    'SHP': '#7E2F8E',   # 紫色
    'DP':  '#0072BD',    # 蓝色 (最优)
    'CP':  '#EDB120',    # 橙色
}

MARKERS = {
    'PA':     'o',
    'IA':     's',
    'JSC-BF': '^',
}

LINESTYLES = {
    'PA':     '--',
    'IA':     '-.',
    'JSC-BF': '-',
}

# Fig.9 的 M0 配色
M0_COLORS = ['#A2142F', '#D95319', '#EDB120', '#77AC30', '#0072BD']
M0_MARKERS = ['o', 's', '^', 'D', 'v']
M0_LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]


def _algo_style(alloc, bf):
    """获取算法组合的绘图样式"""
    return COLORS[alloc], MARKERS[bf], LINESTYLES[bf]


def plot_fig3(results, output_dir='output'):
    """
    绘制 Fig. 3: 感知 SNR vs 功率预算 P_m
    9 条曲线, 双栏尺寸
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    # 绘图顺序: SHP -> CP -> DP, 每组 PA -> IA -> JSC-BF
    allocs = ['SHP', 'CP', 'DP']
    bfs = ['PA', 'IA', 'JSC-BF']

    for alloc in allocs:
        for bf in bfs:
            label = f'{alloc}-{bf}'
            if label not in results:
                continue
            Pm, snr_mean, snr_std = results[label]
            color, marker, ls = _algo_style(alloc, bf)
            ax.plot(Pm, snr_mean, color=color, marker=marker, linestyle=ls,
                    label=label, markersize=5, markevery=1)

    ax.set_xlabel(r'Power Budget $P_m$ (dBW)')
    ax.set_ylabel(r'Sensing SNR $\gamma_s$ (dB)')
    ax.set_title('Fig. 3: Sensing SNR vs. Power Budget')
    ax.legend(loc='upper left', framealpha=0.9, ncol=3, fontsize=7.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Fig3_sensing_SNR_vs_Pm.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f'  Fig. 3 saved to {filepath}')
    return filepath


def plot_fig6(results, output_dir='output'):
    """
    绘制 Fig. 6: 感知 SNR vs 协作卫星数 M_0
    9 条曲线, 双栏尺寸
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    allocs = ['SHP', 'CP', 'DP']
    bfs = ['PA', 'IA', 'JSC-BF']

    for alloc in allocs:
        for bf in bfs:
            label = f'{alloc}-{bf}'
            if label not in results:
                continue
            M0_arr, snr_mean, snr_std = results[label]
            color, marker, ls = _algo_style(alloc, bf)
            ax.plot(M0_arr, snr_mean, color=color, marker=marker, linestyle=ls,
                    label=label, markersize=5, markevery=1)

    ax.set_xlabel(r'Number of Cooperative Satellites $M_0$')
    ax.set_ylabel(r'Sensing SNR $\gamma_s$ (dB)')
    ax.set_title(r'Fig. 6: Sensing SNR vs. Cooperative Satellites ($P_m$=25 dBW)')
    ax.legend(loc='upper left', framealpha=0.9, ncol=3, fontsize=7.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Fig6_sensing_SNR_vs_M0.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f'  Fig. 6 saved to {filepath}')
    return filepath


def plot_fig9(results, output_dir='output'):
    """
    绘制 Fig. 9: CRB 定位误差 vs P_m, 不同 M_0
    使用 DP-JSC-BF 算法, 5 条曲线
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    idx = 0
    for key in sorted(results.keys()):
        Pm, crb_mean, crb_std, M0 = results[key]
        color = M0_COLORS[idx % len(M0_COLORS)]
        marker = M0_MARKERS[idx % len(M0_MARKERS)]
        ls = M0_LINESTYLES[idx % len(M0_LINESTYLES)]
        ax.plot(Pm, crb_mean, color=color, marker=marker, linestyle=ls,
                label=f'$M_0$={M0}', markersize=5, markevery=1)
        idx += 1

    ax.set_xlabel(r'Power Budget $P_m$ (dBW)')
    ax.set_ylabel(r'CRB$^{1/2}$ (m)')
    ax.set_title(r'Fig. 9: CRB vs. Power Budget (DP-JSC-BF)')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Fig9_CRB_vs_Pm.png')
    fig.savefig(filepath)
    plt.close(fig)
    print(f'  Fig. 9 saved to {filepath}')
    return filepath
