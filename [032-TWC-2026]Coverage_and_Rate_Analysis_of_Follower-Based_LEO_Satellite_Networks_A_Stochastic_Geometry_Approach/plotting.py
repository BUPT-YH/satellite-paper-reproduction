"""
IEEE期刊风格绘图模块
Coverage and Rate Analysis of Follower-Based LEO Satellite Networks

绘图规范:
- 字体: Times New Roman 10pt
- 单栏图: 3.5 × 2.8 inches
- 双栏图: 7.16 × 3.5 inches
- 线宽: 数据线 1.5pt, 坐标轴 0.8pt
- 刻度朝内
- 三重区分: color + marker + linestyle
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

import config as cfg


# ============================================================
# IEEE绘图风格设置
# ============================================================

def setup_ieee_style():
    """配置IEEE期刊绘图风格"""
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 10
    rcParams['axes.linewidth'] = 0.8
    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    rcParams['xtick.minor.width'] = 0.6
    rcParams['ytick.minor.width'] = 0.6
    rcParams['legend.fontsize'] = 8
    rcParams['legend.framealpha'] = 0.9
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['mathtext.fontset'] = 'stix'


def ensure_output_dir():
    """确保输出目录存在"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ============================================================
# Fig. 2: 中断概率验证
# ============================================================

def plot_fig2(results):
    """
    绘制Fig. 2: 中断概率 vs γ_th

    参数:
        results: run_fig2_outage_probability()的返回结果
    """
    setup_ieee_style()
    output_dir = ensure_output_dir()

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))

    gamma_th = results['gamma_th_dB']

    # Theorem 1: Leader中断概率 (实线)
    ax.semilogy(gamma_th, results['theorem1'], '-',
                color=cfg.IEEE_colors[0], linewidth=1.5,
                label='Theorem 1 ($N_F$=0)')

    # Theorem 2: Cluster中断概率 (实线)
    ax.semilogy(gamma_th, results['theorem2'], '-',
                color=cfg.IEEE_colors[1], linewidth=1.5,
                label='Theorem 2 ($N_F$=10)')

    # Corollary 1: 上界 (虚线)
    ax.semilogy(gamma_th, results['upper_bound'], '--',
                color=cfg.IEEE_colors[2], linewidth=1.2,
                label='Corollary 1 (Upper)')

    # Corollary 1: 下界 (虚线)
    ax.semilogy(gamma_th, results['lower_bound'], '-.',
                color=cfg.IEEE_colors[3], linewidth=1.2,
                label='Corollary 1 (Lower)')

    # Monte Carlo仿真点 - Leader
    if results['mc_leader']:
        mc_g, mc_p = zip(*results['mc_leader'])
        mc_g = np.array(mc_g)
        mc_p = np.array(mc_p)
        # 过滤掉0值
        valid = mc_p > 0
        ax.semilogy(mc_g[valid], mc_p[valid], 'o',
                     color=cfg.IEEE_colors[0], markersize=5,
                     markerfacecolor='none', markeredgewidth=1.2,
                     label='MC (Leader)')

    # Monte Carlo仿真点 - Cluster
    if results['mc_cluster']:
        mc_g, mc_p = zip(*results['mc_cluster'])
        mc_g = np.array(mc_g)
        mc_p = np.array(mc_p)
        valid = mc_p > 0
        ax.semilogy(mc_g[valid], mc_p[valid], 's',
                     color=cfg.IEEE_colors[1], markersize=5,
                     markerfacecolor='none', markeredgewidth=1.2,
                     label='MC (Cluster)')

    ax.set_xlabel('$\\gamma_{\\mathrm{th}}$ (dB)')
    ax.set_ylabel('Outage Probability')
    ax.set_xlim([-10, 5])
    ax.set_ylim([1e-4, 1])
    ax.grid(True, which='both', linestyle=':', linewidth=0.3, alpha=0.5)
    ax.legend(loc='lower left', fontsize=6, ncol=2)

    fig.tight_layout()
    filepath = os.path.join(output_dir, 'fig2_outage_probability.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig. 2 已保存: {filepath}")
    return filepath


# ============================================================
# Fig. 4: 平均速率 vs N_F
# ============================================================

def plot_fig4(results):
    """
    绘制Fig. 4: 平均速率 vs N_F 和 ρ_FU

    参数:
        results: run_fig4_avg_rate_vs_nf()的返回结果
    """
    setup_ieee_style()
    output_dir = ensure_output_dir()

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))

    N_F = results['N_F']
    rho_FU_range = results['rho_FU_dBW']

    for idx, rho_dBW in enumerate(rho_FU_range):
        color = cfg.IEEE_colors[idx]
        marker = cfg.IEEE_markers[idx]

        # 解析曲线 (中值近似)
        rates = results['middle'][rho_dBW] / 1e9  # 转换为Gbps
        ax.plot(N_F, rates, '-', color=color, linewidth=1.5,
                marker=marker, markersize=4, markerfacecolor='none',
                markeredgewidth=1.0,
                label=f'$\\rho_{{FU}}$={rho_dBW} dBW')

        # 上界 (虚线)
        upper = results['upper'][rho_dBW] / 1e9
        ax.plot(N_F, upper, '--', color=color, linewidth=0.8, alpha=0.6)

        # 下界 (点线)
        lower = results['lower'][rho_dBW] / 1e9
        ax.plot(N_F, lower, ':', color=color, linewidth=0.8, alpha=0.6)

    ax.set_xlabel('Number of Followers ($N_F$)')
    ax.set_ylabel('Average Rate (Gbps)')
    ax.set_xlim([0, 20])
    ax.grid(True, which='major', linestyle=':', linewidth=0.3, alpha=0.5)
    ax.legend(loc='upper left', fontsize=7)

    fig.tight_layout()
    filepath = os.path.join(output_dir, 'fig4_avg_rate_vs_nf.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig. 4 已保存: {filepath}")
    return filepath


# ============================================================
# Fig. 6: L.F vs N.F 速率对比
# ============================================================

def plot_fig6(results):
    """
    绘制Fig. 6: L.F vs N.F 速率对比

    参数:
        results: run_fig6_lf_vs_nf()的返回结果
    """
    setup_ieee_style()
    output_dir = ensure_output_dir()

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))

    N_F = results['N_F']
    rho_total_range = results['rho_total_dBW']

    for idx, rho_total in enumerate(rho_total_range):
        color = cfg.IEEE_colors[idx]
        marker_lf = cfg.IEEE_markers[idx]
        marker_nf = cfg.IEEE_markers[min(idx + 1, len(cfg.IEEE_markers) - 1)]

        # L.F 方案曲线
        lf_rates = results['lf_rates'][rho_total] / 1e9  # Gbps
        ax.plot(N_F, lf_rates, '-', color=color, linewidth=1.5,
                marker=marker_lf, markersize=4, markerfacecolor='none',
                markeredgewidth=1.0,
                label=f'L.F, $\\rho_{{total}}$={rho_total} dBW')

        # N.F 方案: 水平虚线 (不随N_F变化)
        nf_rate = results['nf_rates'][rho_total] / 1e9  # Gbps
        ax.axhline(y=nf_rate, color=color, linestyle='--', linewidth=0.8,
                   alpha=0.6)

    ax.set_xlabel('Number of Followers ($N_F$)')
    ax.set_ylabel('Average Rate (Gbps)')
    ax.set_xlim([0, 20])
    ax.grid(True, which='major', linestyle=':', linewidth=0.3, alpha=0.5)
    ax.legend(loc='upper left', fontsize=6, ncol=2)

    fig.tight_layout()
    filepath = os.path.join(output_dir, 'fig6_lf_vs_nf.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Fig. 6 已保存: {filepath}")
    return filepath


# ============================================================
# 诊断图 (用于验证中间结果)
# ============================================================

def plot_diagnostic_pdf():
    """绘制接触角PDF的诊断图"""
    setup_ieee_style()
    output_dir = ensure_output_dir()

    import stochastic_geometry as sg

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    # 左图: 接触角PDF
    ax = axes[0]
    theta = np.linspace(0.001, np.deg2rad(30), 500)

    pdf_lu = np.array([sg.pdf_theta_LU(t) for t in theta])
    pdf_min = np.array([sg.pdf_theta_min(t) for t in theta if t <= cfg.theta_max - cfg.theta_cap])
    pdf_max = np.array([sg.pdf_theta_max_contact(t) for t in theta if t >= cfg.theta_cap])

    theta_min_valid = theta[:len(pdf_min)]
    theta_max_valid = theta[len(theta) - len(pdf_max):]

    ax.plot(np.rad2deg(theta), pdf_lu, '-', color=cfg.IEEE_colors[0], linewidth=1.5, label='$f_{\\theta_{LU}}$')
    ax.plot(np.rad2deg(theta_min_valid), pdf_min, '--', color=cfg.IEEE_colors[1], linewidth=1.2, label='$f_{\\theta_{min}}$')
    ax.plot(np.rad2deg(theta_max_valid), pdf_max, '-.', color=cfg.IEEE_colors[2], linewidth=1.2, label='$f_{\\theta_{max}}$')

    ax.set_xlabel('$\\theta$ (deg)')
    ax.set_ylabel('PDF')
    ax.legend(fontsize=7)
    ax.grid(True, linestyle=':', linewidth=0.3, alpha=0.5)

    # 右图: Shadowed-Rician CDF
    ax = axes[1]
    w = np.linspace(0.001, 20, 500)
    cdf = np.array([sg.cdf_W(wi) for wi in w])

    ax.plot(w, cdf, '-', color=cfg.IEEE_colors[0], linewidth=1.5, label='Gamma approx.')
    ax.set_xlabel('$w$')
    ax.set_ylabel('CDF $F_W(w)$')
    ax.legend(fontsize=7)
    ax.grid(True, linestyle=':', linewidth=0.3, alpha=0.5)

    fig.tight_layout()
    filepath = os.path.join(output_dir, 'diagnostic_pdf_cdf.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  诊断图已保存: {filepath}")
