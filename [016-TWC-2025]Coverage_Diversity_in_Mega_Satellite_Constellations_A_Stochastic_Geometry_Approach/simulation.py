"""
主仿真脚本 - 运行所有复现实验
Fig.4: 覆盖概率对比（理论 vs Monte Carlo）
Fig.5: Starlink Phase 2 对比
Fig.6: 用户密度对覆盖概率的影响
"""

import numpy as np
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from stochastic_geometry import (
    coverage_selection_single_shell,
    coverage_selection_multi_shell,
    coverage_combining_single_shell,
    coverage_combining_multi_shell,
    monte_carlo_selection,
    monte_carlo_combining,
    avg_interference,
    avg_n_satellites,
    phi_max_m,
)
from plotting import (
    FIG_DOUBLE, FIG_SINGLE, COLORS, MARKERS, LINESTYLES,
    setup_axes, save_fig
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


def run_fig4():
    """
    Fig.4: 覆盖概率对比 - 理论 vs Monte Carlo
    (a) 单壳层 + 多壳层，卫星选择分集
    (b) 单壳层 + 多壳层，合并分集
    """
    print('\n' + '='*60)
    print('Fig.4: Coverage Probability - Theory vs Monte Carlo')
    print('='*60)

    gamma_db = np.arange(-25, 2, 1.0)
    gamma_lin = 10 ** (gamma_db / 10)

    # ---- 理论: 选择分集 ----
    print('  Computing selection diversity (theory)...')
    P_sel_ss_th = []
    P_sel_ms_th = []
    for g in gamma_lin:
        P_sel_ss_th.append(coverage_selection_single_shell(
            g, cfg.SINGLE_SHELL['Nm'], cfg.SINGLE_SHELL['Rm']))
        P_sel_ms_th.append(coverage_selection_multi_shell(
            g, cfg.MULTI_SHELL['Nm'], cfg.MULTI_SHELL['Rm']))

    # ---- MC: 选择分集 ----
    print('  Running MC for selection diversity...')
    mc_n = 10000
    P_sel_ss_mc = []
    P_sel_ms_mc = []
    for g in gamma_lin:
        p1 = monte_carlo_selection(g, cfg.SINGLE_SHELL['Nm'], cfg.SINGLE_SHELL['Rm'],
                                    n_samples=mc_n)
        P_sel_ss_mc.append(p1)
        p2 = monte_carlo_selection(g, cfg.MULTI_SHELL['Nm'], cfg.MULTI_SHELL['Rm'],
                                    n_samples=mc_n)
        P_sel_ms_mc.append(p2)

    # ---- MC: 合并分集 (Euler 反演) ----
    print('  Computing combining diversity (Euler inversion)...')
    P_com_ss_th = []
    P_com_ms_th = []
    for g in gamma_lin:
        try:
            p = coverage_combining_single_shell(g, cfg.SINGLE_SHELL['Nm'], cfg.SINGLE_SHELL['Rm'])
            P_com_ss_th.append(min(max(p, 0), 1))
        except:
            P_com_ss_th.append(0)
        try:
            p = coverage_combining_multi_shell(g, cfg.MULTI_SHELL['Nm'], cfg.MULTI_SHELL['Rm'])
            P_com_ms_th.append(min(max(p, 0), 1))
        except:
            P_com_ms_th.append(0)

    # ---- MC: 合并分集 ----
    print('  Running MC for combining diversity...')
    P_com_ss_mc = []
    P_com_ms_mc = []
    for g in gamma_lin:
        p1 = monte_carlo_combining(g, cfg.SINGLE_SHELL['Nm'], cfg.SINGLE_SHELL['Rm'],
                                    n_samples=mc_n)
        P_com_ss_mc.append(p1)
        p2 = monte_carlo_combining(g, cfg.MULTI_SHELL['Nm'], cfg.MULTI_SHELL['Rm'],
                                    n_samples=mc_n)
        P_com_ms_mc.append(p2)

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # 左图: 选择分集
    ax = axes[0]
    setup_axes(ax, 'SINR Threshold $\\gamma_0$ (dB)',
               'Coverage Probability $P_c(\\gamma_0)$',
               'Satellite Selection')
    ax.semilogy(gamma_db, P_sel_ss_th, color=COLORS[0], ls='-',
                marker=MARKERS[0], markevery=4, label='SS Theory', zorder=3)
    ax.semilogy(gamma_db, P_sel_ss_mc, color=COLORS[0], ls='none',
                marker='x', markevery=4, alpha=0.6, label='SS MC', zorder=2)
    ax.semilogy(gamma_db, P_sel_ms_th, color=COLORS[2], ls='-',
                marker=MARKERS[2], markevery=4, label='MS Theory', zorder=3)
    ax.semilogy(gamma_db, P_sel_ms_mc, color=COLORS[2], ls='none',
                marker='x', markevery=4, alpha=0.6, label='MS MC', zorder=2)
    ax.legend(loc='lower left', framealpha=0.9, edgecolor='none')
    ax.set_ylim(1e-3, 1.05)

    # 右图: 合并分集
    ax = axes[1]
    setup_axes(ax, 'SINR Threshold $\\gamma_0$ (dB)',
               'Coverage Probability $P_c(\\gamma_0)$',
               'Combining Diversity (MRC)')
    ax.semilogy(gamma_db, P_com_ss_th, color=COLORS[1], ls='-',
                marker=MARKERS[1], markevery=4, label='SS Theory', zorder=3)
    ax.semilogy(gamma_db, P_com_ss_mc, color=COLORS[1], ls='none',
                marker='x', markevery=4, alpha=0.6, label='SS MC', zorder=2)
    ax.semilogy(gamma_db, P_com_ms_th, color=COLORS[3], ls='-',
                marker=MARKERS[3], markevery=4, label='MS Theory', zorder=3)
    ax.semilogy(gamma_db, P_com_ms_mc, color=COLORS[3], ls='none',
                marker='x', markevery=4, alpha=0.6, label='MS MC', zorder=2)
    ax.legend(loc='lower left', framealpha=0.9, edgecolor='none')
    ax.set_ylim(1e-3, 1.05)

    fig.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig4_coverage_comparison.png'))
    return True


def run_fig5():
    """
    Fig.5: Starlink Phase 2 对比
    """
    print('\n' + '='*60)
    print('Fig.5: Starlink Phase 2 Comparison')
    print('='*60)

    gamma_db = np.arange(-25, 2, 1.0)
    gamma_lin = 10 ** (gamma_db / 10)

    Nm_starlink = cfg.STARLINK_PHASE2['Nm']
    Rm_starlink = cfg.STARLINK_PHASE2['Rm']

    # 理论: 选择分集
    print('  Computing Starlink selection...')
    P_sel = []
    for g in gamma_lin:
        p = coverage_selection_multi_shell(g, Nm_starlink, Rm_starlink)
        P_sel.append(p)

    # 理论: 合并分集 (用 MC 验证)
    print('  Computing Starlink combining...')
    P_com = []
    mc_n = 8000
    for g in gamma_lin:
        p = monte_carlo_combining(g, Nm_starlink, Rm_starlink, n_samples=mc_n)
        P_com.append(p)

    # ---- 绘图 ----
    fig, ax = plt.subplots(1, 1, figsize=FIG_SINGLE)
    setup_axes(ax, 'SINR Threshold $\\gamma_0$ (dB)',
               'Coverage Probability $P_c(\\gamma_0)$')

    ax.semilogy(gamma_db, P_sel, color=COLORS[0], ls='-',
                marker=MARKERS[0], markevery=4,
                label='Selection Diversity')
    ax.semilogy(gamma_db, P_com, color=COLORS[1], ls='-',
                marker=MARKERS[1], markevery=4,
                label='Combining Diversity')

    ax.legend(loc='lower left', framealpha=0.9, edgecolor='none')
    ax.set_ylim(1e-3, 1.05)
    ax.set_title("Starlink Phase 2 Theoretical Model")

    fig.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig5_starlink_comparison.png'))
    return True


def run_fig6():
    """
    Fig.6: 用户密度对覆盖概率的影响 (γo = -20 dB)
    """
    print('\n' + '='*60)
    print('Fig.6: Effect of Ground User Intensity')
    print('='*60)

    gamma_o = cfg.GAMMA_FIG6  # -20 dB 线性值

    # 用户密度扫描
    lambda_o_sweep = np.logspace(-12, -6, 30)
    duty_cycle = cfg.D_O

    # 三种星座配置
    configs = [
        {'label': '1 Shell (600 km)', 'Nm_list': [900], 'Rm_list': [600]},
        {'label': '2 Shells (600+900 km)', 'Nm_list': [900, 900], 'Rm_list': [600, 900]},
        {'label': '3 Shells (600+900+1200 km)', 'Nm_list': [900, 900, 900], 'Rm_list': [600, 900, 1200]},
    ]

    # 选择分集
    print('  Computing selection diversity...')
    P_sel_results = {}
    for config_item in configs:
        label = config_item['label']
        Nm = config_item['Nm_list']
        Rm = config_item['Rm_list']
        Ps = []
        for lam_o in lambda_o_sweep:
            lam_u = duty_cycle * lam_o
            if len(Nm) == 1:
                p = coverage_selection_single_shell(gamma_o, Nm[0], Rm[0], lam_u)
            else:
                p = coverage_selection_multi_shell(gamma_o, Nm, Rm, lam_u)
            Ps.append(p)
        P_sel_results[label] = Ps
        print(f'    {label}: done')

    # 合并分集 (MC)
    print('  Computing combining diversity...')
    P_com_results = {}
    mc_n = 5000
    for config_item in configs:
        label = config_item['label']
        Nm = config_item['Nm_list']
        Rm = config_item['Rm_list']
        Ps = []
        for lam_o in lambda_o_sweep:
            lam_u = duty_cycle * lam_o
            if len(Nm) == 1:
                p = monte_carlo_combining(gamma_o, Nm[0], Rm[0], lam_u, n_samples=mc_n)
            else:
                p = monte_carlo_combining(gamma_o, Nm, Rm, lam_u, n_samples=mc_n)
            Ps.append(p)
        P_com_results[label] = Ps
        print(f'    {label}: done')

    # x 轴: λ_o users/m² → per 100 km²
    x_vals = lambda_o_sweep * 1e8  # users per 100 km²

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # (a) 选择分集
    ax = axes[0]
    setup_axes(ax, 'User Density (per $100\\,\\mathrm{km}^2$)',
               'Coverage Probability', '(a) Satellite Selection')
    for i, config_item in enumerate(configs):
        label = config_item['label']
        ax.semilogx(x_vals, P_sel_results[label], color=COLORS[i], ls='-',
                     marker=MARKERS[i], markevery=4, label=label)
    ax.legend(loc='lower left', framealpha=0.9, fontsize=7)
    ax.set_ylim(0, 1.05)

    # (b) 合并分集
    ax = axes[1]
    setup_axes(ax, 'User Density (per $100\\,\\mathrm{km}^2$)',
               'Coverage Probability', '(b) Combining Diversity')
    for i, config_item in enumerate(configs):
        label = config_item['label']
        ax.semilogx(x_vals, P_com_results[label], color=COLORS[i], ls='-',
                     marker=MARKERS[i], markevery=4, label=label)
    ax.legend(loc='lower left', framealpha=0.9, fontsize=7)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig6_user_intensity.png'))
    return True
