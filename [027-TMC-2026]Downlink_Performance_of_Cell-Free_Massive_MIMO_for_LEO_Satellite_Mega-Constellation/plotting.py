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

# IEEE 尺寸
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)

# 颜色方案
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]


def save_fig(fig, name, output_dir='output'):
    """保存图片"""
    fig.savefig(f'{output_dir}/{name}', dpi=300, bbox_inches='tight',
                pad_inches=0.05)
    plt.close(fig)
    print(f'  Saved {name}')


def plot_fig4(gamma_th, ppp_cov, starlink_random, starlink_fixed, labels, output_dir='output'):
    """Fig.4: PPP vs Starlink覆盖概率对比"""
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    for i, (cov, label) in enumerate(zip(
            [ppp_cov] + starlink_random + starlink_fixed, labels)):
        if cov is not None:
            ax.plot(gamma_th, cov, label=label, fillstyle='none',
                    color=COLORS[i % len(COLORS)],
                    marker=MARKERS[i % len(MARKERS)],
                    linestyle=LINESTYLES[i % len(LINESTYLES)],
                    markersize=4, markevery=3)

    ax.set_xlabel('SINR Threshold (dB)')
    ax.set_ylabel('Coverage Probability')
    ax.set_xlim([-5, 15])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7)
    save_fig(fig, 'fig4_coverage_ppp_vs_starlink.png', output_dir)


def plot_fig5(x_range, ccdf_perfect, ccdf_imperfect_dict, output_dir='output'):
    """Fig.5: DSS的CCDF"""
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    ax.plot(x_range, ccdf_perfect, color=COLORS[0], marker=MARKERS[0],
            ls=LINESTYLES[0], label='Analytical (Perfect CSI)',
            fillstyle='none', markevery=max(1, len(x_range)//10), markersize=4)

    colors_imp = [COLORS[1], COLORS[2], COLORS[3]]
    markers_imp = [MARKERS[1], MARKERS[2], MARKERS[3]]
    ls_imp = [LINESTYLES[1], LINESTYLES[2], LINESTYLES[3]]
    for i, (tau_p_val, ccdf) in enumerate(ccdf_imperfect_dict.items()):
        ax.plot(x_range, ccdf, color=colors_imp[i], marker=markers_imp[i],
                ls=ls_imp[i], label=f'Simulation ($\\tau_p$={tau_p_val})',
                fillstyle='none', markevery=max(1, len(x_range)//8), markersize=4)

    ax.set_xlabel('Desired Signal Strength (DSS) ($\\times 10^{-5}$ W)')
    # 用×1e-5显示x轴刻度
    ticks = ax.get_xticks()
    ax.set_xticklabels([f'{t/1e-5:.1f}' for t in ticks])
    ax.set_ylabel('CCDF')
    ax.set_yscale('log')
    ax.set_ylim([1e-3, 1.1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7)
    save_fig(fig, 'fig5_dss_ccdf.png', output_dir)


def plot_fig6(gamma_th, curves, output_dir='output'):
    """Fig.6: 不同Nakagami m参数的覆盖概率"""
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    idx = 0
    for label, (cov_ana, cov_mc) in curves.items():
        c = COLORS[idx % len(COLORS)]
        mk = MARKERS[idx % len(MARKERS)]
        if cov_ana is not None:
            ax.plot(gamma_th, cov_ana, color=c, ls='-', label=f'{label} (Ana.)')
        if cov_mc is not None:
            ax.plot(gamma_th, cov_mc, color=c, ls='--', marker=mk,
                    fillstyle='none', markersize=4, markevery=3,
                    label=f'{label} (Sim.)')
        idx += 1

    ax.set_xlabel('SINR Threshold (dB)')
    ax.set_ylabel('Coverage Probability')
    ax.set_xlim([-2, 15])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7,
              ncol=2)
    save_fig(fig, 'fig6_coverage_nakagami_m.png', output_dir)


def plot_fig7(gamma_th, cf_curves, cell_curves, output_dir='output'):
    """Fig.7: CF-based vs Cell-based"""
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    idx = 0
    for label, cov in cf_curves.items():
        c = COLORS[idx % len(COLORS)]
        ax.plot(gamma_th, cov, color=c, ls='-', marker=MARKERS[idx % len(MARKERS)],
                fillstyle='none', markersize=4, markevery=3,
                label=f'CF: {label}')
        idx += 1

    idx = 0
    for label, cov in cell_curves.items():
        c = COLORS[idx % len(COLORS)]
        ax.plot(gamma_th, cov, color=c, ls='--', marker=MARKERS[idx % len(MARKERS)],
                fillstyle='none', markersize=4, markevery=3,
                label=f'Cell: {label}')
        idx += 1

    ax.set_xlabel('SINR Threshold (dB)')
    ax.set_ylabel('Coverage Probability')
    ax.set_xlim([-2, 15])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7,
              ncol=2)
    save_fig(fig, 'fig7_cf_vs_cell.png', output_dir)


def plot_fig8(gamma_th, curves, output_dir='output'):
    """Fig.8: 有/无波束赋形"""
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    idx = 0
    for label, (cov_ana, cov_mc) in curves.items():
        c = COLORS[idx % len(COLORS)]
        mk = MARKERS[idx % len(MARKERS)]
        ls = LINESTYLES[idx % len(LINESTYLES)]
        if cov_ana is not None:
            ax.plot(gamma_th, cov_ana, color=c, ls=ls, label=f'{label} (Ana.)')
        if cov_mc is not None:
            ax.plot(gamma_th, cov_mc, color=c, ls='--', marker=mk,
                    fillstyle='none', markersize=4, markevery=3,
                    label=f'{label} (Sim.)')
        idx += 1

    ax.set_xlabel('SINR Threshold (dB)')
    ax.set_ylabel('Coverage Probability')
    ax.set_xlim([-2, 15])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=7,
              ncol=2)
    save_fig(fig, 'fig8_with_without_beamforming.png', output_dir)


def plot_fig9(gamma_th, eta_curves, output_dir='output'):
    """Fig.9: 不同dome angle η的覆盖概率"""
    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    idx = 0
    for label, cov in eta_curves.items():
        c = COLORS[idx % len(COLORS)]
        mk = MARKERS[idx % len(MARKERS)]
        ax.plot(gamma_th, cov, color=c, ls=LINESTYLES[idx % len(LINESTYLES)],
                marker=mk, fillstyle='none', markersize=4, markevery=3,
                label=label)
        idx += 1

    ax.set_xlabel('SINR Threshold (dB)')
    ax.set_ylabel('Coverage Probability')
    ax.set_xlim([-2, 15])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.02, 1), borderaxespad=0,
              framealpha=0.9, edgecolor='none', fontsize=7)
    fig.savefig(f'{output_dir}/fig9_coverage_dome_angle.png', dpi=300,
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'  Saved fig9_coverage_dome_angle.png')


def plot_fig10(altitudes, n_saps, coverage_3d, output_dir='output'):
    """Fig.10: 3D覆盖概率 (高度 vs SAP数)"""
    fig = plt.figure(figsize=(5.5, 4))
    ax = fig.add_subplot(111, projection='3d')

    A, N = np.meshgrid(altitudes, n_saps)
    surf = ax.plot_surface(A, N, coverage_3d.T, cmap='viridis', alpha=0.9,
                           edgecolor='none')
    ax.set_xlabel('Orbital Altitude (km)')
    ax.set_ylabel('Number of SAPs')
    ax.set_zlabel('Coverage Probability')
    ax.set_zlim([0, 1])
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.12)
    cbar.set_label('$P_{cov}$')
    fig.subplots_adjust(right=0.85)
    fig.savefig(f'{output_dir}/fig10_coverage_3d_altitude_saps.png', dpi=300,
                bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f'  Saved fig10_coverage_3d_altitude_saps.png')


def plot_fig11(n_saps_arr, n_uts_arr, coverage_3d, output_dir='output'):
    """Fig.11: 3D覆盖概率 (SAP数 vs UT数)"""
    fig = plt.figure(figsize=(5.5, 4))
    ax = fig.add_subplot(111, projection='3d')

    S, U = np.meshgrid(n_saps_arr, n_uts_arr)
    surf = ax.plot_surface(S, U, coverage_3d, cmap='viridis', alpha=0.9,
                           edgecolor='none')
    ax.set_xlabel('Number of SAPs')
    ax.set_ylabel('Number of UTs')
    ax.set_zlabel('Coverage Probability')
    ax.set_zlim([0, 1])
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.12)
    cbar.set_label('$P_{cov}$')
    fig.subplots_adjust(right=0.85)
    fig.savefig(f'{output_dir}/fig11_coverage_3d_saps_uts.png', dpi=300,
                bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f'  Saved fig11_coverage_3d_saps_uts.png')


def plot_fig12(n_ut_range, capacity_curves, output_dir='output'):
    """Fig.12: 系统容量 vs UT数"""
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    idx = 0
    for label, cap in capacity_curves.items():
        c = COLORS[idx % len(COLORS)]
        mk = MARKERS[idx % len(MARKERS)]
        ls = LINESTYLES[idx % len(LINESTYLES)]
        ax.plot(n_ut_range / 1000, np.array(cap) / 1e9, color=c, ls=ls,
                marker=mk, fillstyle='none', markersize=4, markevery=3,
                label=label)
        idx += 1

    ax.set_xlabel('Number of UTs ($\\times 10^3$)')
    ax.set_ylabel('System Capacity (Gbps)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, edgecolor='none', fontsize=7)
    save_fig(fig, 'fig12_system_capacity.png', output_dir)


def plot_fig13(n_ut_range, per_user_curves, output_dir='output'):
    """Fig.13: 每用户容量 vs UT数"""
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    idx = 0
    for label, cap in per_user_curves.items():
        c = COLORS[idx % len(COLORS)]
        mk = MARKERS[idx % len(MARKERS)]
        ls = LINESTYLES[idx % len(LINESTYLES)]
        ax.plot(n_ut_range / 1000, np.array(cap) / 1e6, color=c, ls=ls,
                marker=mk, fillstyle='none', markersize=4, markevery=3,
                label=label)
        idx += 1

    ax.set_xlabel('Number of UTs ($\\times 10^3$)')
    ax.set_ylabel('Per-user Capacity (Mbps)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, edgecolor='none', fontsize=7)
    save_fig(fig, 'fig13_per_user_capacity.png', output_dir)
