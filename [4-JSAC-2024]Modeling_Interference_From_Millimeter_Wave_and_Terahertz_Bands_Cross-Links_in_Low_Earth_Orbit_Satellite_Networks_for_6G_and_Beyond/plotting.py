"""
绘图模块 - 统一风格
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 12

# 颜色方案 (colorblind-friendly)
COLORS = {
    'mmWave_5': '#1f77b4',     # 蓝色
    'mmWave_10': '#ff7f0e',    # 橙色
    'mmWave_30': '#2ca02c',    # 绿色
    'mmWave_40': '#d62728',    # 红色
    'subTHz_1': '#9467bd',     # 紫色
    'subTHz_3': '#8c564b',     # 棕色
    'subTHz_5': '#e377c2',     # 粉色
    'limit': '#000000',        # 黑色
    'reference': '#7f7f7f',    # 灰色
}

MARKERS = {
    'mmWave_5': 'o',
    'mmWave_10': 's',
    'mmWave_30': '^',
    'mmWave_40': 'D',
    'subTHz_1': 'v',
    'subTHz_3': 'P',
    'subTHz_5': 'X',
}


def save_figure(fig, filename, output_dir='output'):
    """保存图片"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存: {filepath}")


def plot_figure5a(N_range, sir_data, SIR_limit_dB, output_dir='output'):
    """
    Figure 5(a): 单轨道 SIR vs N
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for key, sir_vals in sir_data.items():
        sir_dB = 10 * np.log10(np.clip(sir_vals, 1e-10, None))
        color = COLORS.get(key, '#333333')
        marker = MARKERS.get(key, 'o')
        # 论文中用不同线型区分
        label = key.replace('_', ' = ').replace('mmWave', 'mmWave').replace('subTHz', 'sub-THz')
        ax.plot(N_range, sir_dB, color=color, linewidth=1.5, marker=marker,
                markersize=3, markevery=10, label=label)

    # 理论极限线
    ax.axhline(y=SIR_limit_dB, color='black', linestyle='--', linewidth=1.5,
               label=f'Theoretical limit ({SIR_limit_dB} dB)')

    ax.set_xlabel('Number of Satellites per Orbit, $N$')
    ax.set_ylabel('SIR, $\\Gamma_1$ (dB)')
    ax.set_title('Single Orbit: SIR vs Number of Satellites')
    ax.set_xlim([5, 200])
    ax.set_ylim([-10, 60])
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'figure5a_SIR_vs_N.png', output_dir)


def plot_figure5b(N_range, sinr_data, SIR_limit_dB, output_dir='output'):
    """
    Figure 5(b): 单轨道 SINR vs N
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for key, sinr_vals in sinr_data.items():
        sinr_dB = 10 * np.log10(np.clip(sinr_vals, 1e-10, None))
        color = COLORS.get(key, '#333333')
        marker = MARKERS.get(key, 'o')
        label = key.replace('_', ' = ').replace('mmWave', 'mmWave').replace('subTHz', 'sub-THz')
        ax.plot(N_range, sinr_dB, color=color, linewidth=1.5, marker=marker,
                markersize=3, markevery=10, label=label)

    # 理论极限线
    ax.axhline(y=SIR_limit_dB, color='black', linestyle='--', linewidth=1.5,
               label=f'SIR limit ({SIR_limit_dB} dB)')

    ax.set_xlabel('Number of Satellites per Orbit, $N$')
    ax.set_ylabel('SINR, $S_1$ (dB)')
    ax.set_title('Single Orbit: SINR vs Number of Satellites')
    ax.set_xlim([5, 200])
    ax.set_ylim([-10, 80])
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'figure5b_SINR_vs_N.png', output_dir)


def plot_figure5c(N_range, capacity_data, cap_limit_mmWave, cap_limit_subTHz, output_dir='output'):
    """
    Figure 5(c): 单轨道信道容量 vs N
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for key, cap_vals in capacity_data.items():
        cap_Gbps = np.array(cap_vals) / 1e9
        color = COLORS.get(key, '#333333')
        marker = MARKERS.get(key, 'o')
        label = key.replace('_', ' = ').replace('mmWave', 'mmWave').replace('subTHz', 'sub-THz')
        ax.plot(N_range, cap_Gbps, color=color, linewidth=1.5, marker=marker,
                markersize=3, markevery=10, label=label)

    # 容量极限线
    ax.axhline(y=cap_limit_mmWave / 1e9, color='black', linestyle='--', linewidth=1,
               label=f'mmWave capacity limit')
    ax.axhline(y=cap_limit_subTHz / 1e9, color='gray', linestyle=':', linewidth=1,
               label=f'sub-THz capacity limit')

    ax.set_xlabel('Number of Satellites per Orbit, $N$')
    ax.set_ylabel('Channel Capacity (Gbps)')
    ax.set_title('Single Orbit: Channel Capacity vs Number of Satellites')
    ax.set_xlim([5, 200])
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'figure5c_capacity_vs_N.png', output_dir)


def plot_figure9b(alpha_range, sinr_data_mmWave, sinr_data_subTHz, snr_ref=None, output_dir='output'):
    """
    Figure 9(b): 偏移轨道 SINR vs 波束宽度
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for key, sinr_vals in sinr_data_mmWave.items():
        sinr_dB = 10 * np.log10(np.clip(sinr_vals, 1e-10, None))
        color = COLORS.get(key, '#333333')
        label = f'mmWave, {key}'
        ax.plot(alpha_range, sinr_dB, color=color, linewidth=1.5, label=label)

    for key, sinr_vals in sinr_data_subTHz.items():
        sinr_dB = 10 * np.log10(np.clip(sinr_vals, 1e-10, None))
        color = COLORS.get(key, '#666666')
        linestyle = '--'
        label = f'sub-THz, {key}'
        ax.plot(alpha_range, sinr_dB, color=color, linewidth=1.5, linestyle=linestyle, label=label)

    if snr_ref is not None:
        snr_dB = 10 * np.log10(np.clip(snr_ref, 1e-10, None))
        ax.plot(alpha_range, snr_dB, color='gray', linestyle=':', linewidth=1.5,
                label='SNR (N=50, no interference)')

    ax.set_xlabel('Beamwidth, $\\alpha$ (degrees)')
    ax.set_ylabel('SINR, $S_3$ (dB)')
    ax.set_title('Shifted Orbits: SINR vs Beamwidth')
    ax.set_xlim([1, 40])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'figure9b_SINR_vs_beamwidth.png', output_dir)


def plot_figure10(N_range, capacity_data, output_dir='output'):
    """
    Figure 10: 完整双星座部署信道容量 vs N
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, cap_vals in capacity_data.items():
        cap_Gbps = np.array(cap_vals) / 1e9
        ax.plot(N_range, cap_Gbps, linewidth=1.8, label=key)

    ax.set_xlabel('Number of Satellites per Orbit, $N$')
    ax.set_ylabel('Channel Capacity (Gbps)')
    ax.set_title('Full Two-Constellation Deployment: Channel Capacity')
    ax.set_xlim([10, 500])
    ax.set_ylim([0, None])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'figure10_full_constellation_capacity.png', output_dir)
