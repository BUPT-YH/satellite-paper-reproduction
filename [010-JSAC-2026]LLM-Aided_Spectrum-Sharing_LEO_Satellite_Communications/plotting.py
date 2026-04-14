"""
IEEE 期刊标准绘图模块
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from config import FIG_SINGLE, FIG_DOUBLE, COLORS, MARKERS, LINESTYLES

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


# ==================== Fig. 8: Pout,1 vs PS ====================

def plot_fig8(ps_range, pout_data, output_dir):
    """Fig. 8: Pout,1 vs PS for different dSD values

    参数:
        ps_range: PS 数组 (dBW)
        pout_data: dict {dSD_km: {'analytical': array, 'sim': array}}
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    for idx, (dSD_km, data) in enumerate(pout_data.items()):
        color = COLORS[idx]
        marker = MARKERS[idx]

        # 解析曲线 (实线)
        ax.semilogy(ps_range, data['analytical'], color=color,
                     linestyle='-', linewidth=1.5,
                     label=f'$d_{{SD}}={dSD_km}$ km (Analytical)')

        # 仿真曲线 (标记点)
        ax.semilogy(ps_range, data['sim'], color=color,
                     linestyle='none', marker=marker, markersize=6,
                     label=f'$d_{{SD}}={dSD_km}$ km (Simulation)')

    ax.set_xlabel('$P_S$ (dBW)')
    ax.set_ylabel('$P_{out,1}$')
    ax.set_title('Fig. 8: Outage Probability (Spectrum Sharing)')
    ax.legend(loc='best', framealpha=0.9, edgecolor='none', ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(ps_range[0], ps_range[-1])
    fig.tight_layout()
    fig.savefig(f'{output_dir}/fig8_pout1_vs_PS.png')
    plt.close(fig)
    print(f"[✓] Fig.8 已保存")


# ==================== Fig. 9: Pout,2 vs γth ====================

def plot_fig9(gamma_range, pout_data, output_dir):
    """Fig. 9: Pout,2 vs γth for different dSD values

    参数:
        gamma_range: γth 数组 (dB)
        pout_data: dict {dSD_km: {'analytical': array, 'sim': array}}
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    for idx, (dSD_km, data) in enumerate(pout_data.items()):
        color = COLORS[idx]
        marker = MARKERS[idx]

        ax.semilogy(gamma_range, data['analytical'], color=color,
                     linestyle='-', linewidth=1.5,
                     label=f'$d_{{SD}}={dSD_km}$ km (Analytical)')

        ax.semilogy(gamma_range, data['sim'], color=color,
                     linestyle='none', marker=marker, markersize=6,
                     label=f'$d_{{SD}}={dSD_km}$ km (Simulation)')

    ax.set_xlabel('$\\gamma_{th}$ (dB)')
    ax.set_ylabel('$P_{out,2}$')
    ax.set_title('Fig. 9: Outage Probability (Fixed-Band)')
    ax.legend(loc='best', framealpha=0.9, edgecolor='none', ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(gamma_range[0], gamma_range[-1])
    fig.tight_layout()
    fig.savefig(f'{output_dir}/fig9_pout2_vs_gamma_dSD.png')
    plt.close(fig)
    print(f"[✓] Fig.9 已保存")


# ==================== Fig. 10: Pout,2 vs γth, varying λe·Δfs ====================

def plot_fig10(gamma_range, pout_data, output_dir):
    """Fig. 10: Pout,2 vs γth for different λe·Δfs values

    参数:
        gamma_range: γth 数组 (dB)
        pout_data: dict {lambda_val: {'analytical': array, 'sim': array}}
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    for idx, (lambda_val, data) in enumerate(sorted(pout_data.items())):
        color = COLORS[idx]
        marker = MARKERS[idx]

        ax.semilogy(gamma_range, data['analytical'], color=color,
                     linestyle='-', linewidth=1.5,
                     label=f'$\\lambda_e \\Delta f_s={lambda_val}$ (Analytical)')

        ax.semilogy(gamma_range, data['sim'], color=color,
                     linestyle='none', marker=marker, markersize=6,
                     label=f'$\\lambda_e \\Delta f_s={lambda_val}$ (Simulation)')

    ax.set_xlabel('$\\gamma_{th}$ (dB)')
    ax.set_ylabel('$P_{out,2}$')
    ax.set_title('Fig. 10: Outage Probability vs Interference Density')
    ax.legend(loc='best', framealpha=0.9, edgecolor='none', ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(gamma_range[0], gamma_range[-1])
    fig.tight_layout()
    fig.savefig(f'{output_dir}/fig10_pout2_vs_gamma_lambda.png')
    plt.close(fig)
    print(f"[✓] Fig.10 已保存")


# ==================== Fig. 11: LLM vs Fixed OP comparison ====================

def plot_fig11(gamma_range, pout_data, output_dir):
    """Fig. 11: OP with LLM decision (Pout,1) vs Fixed-band (Pout,2)

    参数:
        gamma_range: γth 数组 (dB)
        pout_data: dict {PS_dB: {'llm': array, 'fixed': array}}
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    for idx, (ps_dB, data) in enumerate(sorted(pout_data.items())):
        color = COLORS[idx]

        # Fixed-band: 实线
        ax.semilogy(gamma_range, data['fixed'], color=color,
                     linestyle='-', linewidth=1.5,
                     label=f'$P_S={ps_dB}$ dB (Fixed-Band)')

        # LLM decision: 虚线
        ax.semilogy(gamma_range, data['llm'], color=color,
                     linestyle='--', linewidth=1.5,
                     label=f'$P_S={ps_dB}$ dB (LLM)')

    ax.set_xlabel('$\\gamma_{th}$ (dB)')
    ax.set_ylabel('Outage Probability')
    ax.set_title('Fig. 11: LLM Decision vs Fixed-Band')
    ax.legend(loc='best', framealpha=0.9, edgecolor='none', ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(gamma_range[0], gamma_range[-1])
    fig.tight_layout()
    fig.savefig(f'{output_dir}/fig11_llm_vs_fixed.png')
    plt.close(fig)
    print(f"[✓] Fig.11 已保存")


# ==================== Fig. 12: Transmitted data volume ====================

def plot_fig12(t_seq, data_schemes, output_dir):
    """Fig. 12: Transmitted data volume over time

    参数:
        t_seq: 时间数组
        data_schemes: dict {scheme_name: cumulative_data_array (GB)}
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    for idx, (name, data) in enumerate(data_schemes.items()):
        color = COLORS[idx]
        marker = MARKERS[idx]
        # 每隔一些点画标记
        markevery = max(1, len(t_seq) // 15)
        ax.plot(t_seq, data, color=color, linestyle='-', linewidth=1.5,
                marker=marker, markersize=5, markevery=markevery,
                label=name)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Transmitted Data Volume (GB)')
    ax.set_title('Fig. 12: Transmitted Data Volume Comparison')
    ax.legend(loc='best', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_seq[-1])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/fig12_data_volume.png')
    plt.close(fig)
    print(f"[✓] Fig.12 已保存")


# ==================== Fig. 13: Transmission and waiting delay ====================

def plot_fig13(delay_data, output_dir):
    """Fig. 13: Stacked bar chart of transmission and waiting delay

    参数:
        delay_data: dict {scheme_name: {'wait': dict(service->time),
                                         'trans': dict(service->time)}}
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    services = ['A', 'B', 'C', 'D']
    scheme_names = list(delay_data.keys())
    n_schemes = len(scheme_names)

    x = np.arange(n_schemes)
    width = 0.6

    service_colors = {
        'A_wait': '#AEC7E8', 'A_trans': '#1F77B4',
        'B_wait': '#FFBB78', 'B_trans': '#FF7F0E',
        'C_wait': '#98DF8A', 'C_trans': '#2CA02C',
        'D_wait': '#FF9896', 'D_trans': '#D62728',
    }

    for s_idx, s in enumerate(services):
        bottoms_wait = np.zeros(n_schemes)
        bottoms_trans = np.zeros(n_schemes)
        waits = []
        trans = []

        for scheme_name in scheme_names:
            w = delay_data[scheme_name]['wait'][s]
            t = delay_data[scheme_name]['trans'][s]
            waits.append(w)
            trans.append(t)

        # 计算累积底部
        for j in range(n_schemes):
            bottoms_wait[j] = sum(delay_data[scheme_names[j]]['wait'][s2] +
                                  delay_data[scheme_names[j]]['trans'][s2]
                                  for s2 in services[:s_idx])
            bottoms_trans[j] = bottoms_wait[j] + waits[j]

        # 等待时间 (浅色)
        ax.bar(x, waits, width, bottom=bottoms_wait,
               color=service_colors[f'{s}_wait'],
               edgecolor='white', linewidth=0.3,
               label=f'{s} Wait' if True else '')

        # 传输时间 (深色)
        ax.bar(x, trans, width, bottom=bottoms_trans,
               color=service_colors[f'{s}_trans'],
               edgecolor='white', linewidth=0.3,
               label=f'{s} Trans')

    ax.set_xticks(x)
    ax.set_xticklabels(scheme_names, fontsize=9)
    ax.set_ylabel('Transmission and Waiting Delay (s)')
    ax.set_title('Fig. 13: Delay Comparison')
    ax.legend(loc='best', framealpha=0.9, edgecolor='none',
              ncol=4, fontsize=7, columnspacing=1.0)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{output_dir}/fig13_delay_comparison.png')
    plt.close(fig)
    print(f"[✓] Fig.13 已保存")
