"""
一键复现脚本 - 优化版
跳过慢速 Euler 反演，使用理论(选择) + MC(合并) 策略
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as cfg
from stochastic_geometry import (
    coverage_selection_single_shell,
    coverage_selection_multi_shell,
    monte_carlo_selection,
    monte_carlo_combining,
)
from plotting import FIG_DOUBLE, FIG_SINGLE, COLORS, MARKERS, setup_axes, save_fig

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


def run_fig4():
    """Fig.4: 覆盖概率对比"""
    print('\n' + '='*60)
    print('Fig.4: Coverage Probability')
    print('='*60)

    gamma_db = np.arange(-25, 2, 1.0)
    gamma_lin = 10 ** (gamma_db / 10)
    mc_n = 10000

    # 选择分集 - 理论 + MC
    print('  Selection diversity...')
    P_sel_ss_th = [coverage_selection_single_shell(g, 900, 600) for g in gamma_lin]
    P_sel_ms_th = [coverage_selection_multi_shell(g, [900,400,100], [600,900,1200]) for g in gamma_lin]
    P_sel_ss_mc = [monte_carlo_selection(g, 900, 600, n_samples=mc_n) for g in gamma_lin]
    P_sel_ms_mc = [monte_carlo_selection(g, [900,400,100], [600,900,1200], n_samples=mc_n) for g in gamma_lin]

    # 合并分集 - MC
    print('  Combining diversity (MC)...')
    P_com_ss_mc = [monte_carlo_combining(g, 900, 600, n_samples=mc_n) for g in gamma_lin]
    P_com_ms_mc = [monte_carlo_combining(g, [900,400,100], [600,900,1200], n_samples=mc_n) for g in gamma_lin]

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    ax = axes[0]
    setup_axes(ax, 'SINR Threshold $\\gamma_0$ (dB)', 'Coverage Probability', 'Satellite Selection')
    ax.semilogy(gamma_db, P_sel_ss_th, color=COLORS[0], ls='-', marker=MARKERS[0], markevery=4, label='1-Shell Theory')
    ax.semilogy(gamma_db, P_sel_ss_mc, color=COLORS[0], ls='none', marker='x', markevery=4, alpha=0.5, label='1-Shell MC')
    ax.semilogy(gamma_db, P_sel_ms_th, color=COLORS[2], ls='-', marker=MARKERS[2], markevery=4, label='3-Shell Theory')
    ax.semilogy(gamma_db, P_sel_ms_mc, color=COLORS[2], ls='none', marker='x', markevery=4, alpha=0.5, label='3-Shell MC')
    ax.legend(loc='lower left', framealpha=0.9, fontsize=7, edgecolor='none')
    ax.set_ylim(1e-3, 1.05)

    ax = axes[1]
    setup_axes(ax, 'SINR Threshold $\\gamma_0$ (dB)', 'Coverage Probability', 'Combining Diversity (MRC)')
    ax.semilogy(gamma_db, P_com_ss_mc, color=COLORS[1], ls='-', marker=MARKERS[1], markevery=4, label='1-Shell MC')
    ax.semilogy(gamma_db, P_com_ms_mc, color=COLORS[3], ls='-', marker=MARKERS[3], markevery=4, label='3-Shell MC')
    ax.legend(loc='lower left', framealpha=0.9, edgecolor='none')
    ax.set_ylim(1e-3, 1.05)

    fig.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig4_coverage_comparison.png'))

    # 打印结果表
    print(f'\n  {"γ(dB)":>6s} | {"Sel-SS-Th":>10s} | {"Sel-SS-MC":>10s} | {"Sel-MS-Th":>10s} | {"Com-SS-MC":>10s} | {"Com-MS-MC":>10s}')
    for i in range(0, len(gamma_db), 5):
        print(f'  {gamma_db[i]:>6.0f} | {P_sel_ss_th[i]:>10.4f} | {P_sel_ss_mc[i]:>10.4f} | {P_sel_ms_th[i]:>10.4f} | {P_com_ss_mc[i]:>10.4f} | {P_com_ms_mc[i]:>10.4f}')


def run_fig5():
    """Fig.5: Starlink Phase 2"""
    print('\n' + '='*60)
    print('Fig.5: Starlink Phase 2')
    print('='*60)

    gamma_db = np.arange(-25, 2, 1.0)
    gamma_lin = 10 ** (gamma_db / 10)
    mc_n = 8000

    Nm = cfg.STARLINK_PHASE2['Nm']
    Rm = cfg.STARLINK_PHASE2['Rm']

    print('  Selection...')
    P_sel = [coverage_selection_multi_shell(g, Nm, Rm) for g in gamma_lin]
    print('  Combining...')
    P_com = [monte_carlo_combining(g, Nm, Rm, n_samples=mc_n) for g in gamma_lin]

    fig, ax = plt.subplots(1, 1, figsize=FIG_SINGLE)
    setup_axes(ax, 'SINR Threshold $\\gamma_0$ (dB)', 'Coverage Probability')
    ax.semilogy(gamma_db, P_sel, color=COLORS[0], ls='-', marker=MARKERS[0], markevery=4, label='Selection Diversity')
    ax.semilogy(gamma_db, P_com, color=COLORS[1], ls='-', marker=MARKERS[1], markevery=4, label='Combining Diversity')
    ax.legend(loc='lower left', framealpha=0.9, edgecolor='none')
    ax.set_ylim(1e-3, 1.05)
    ax.set_title("Starlink Phase 2 Theoretical Model")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig5_starlink_comparison.png'))


def run_fig6():
    """Fig.6: 用户密度影响"""
    print('\n' + '='*60)
    print('Fig.6: User Intensity Effect (γo=-20dB)')
    print('='*60)

    gamma_o = cfg.GAMMA_FIG6
    lambda_o_sweep = np.logspace(-12, -6, 25)
    mc_n = 3000

    configs = [
        {'label': '1 Shell', 'Nm': [900], 'Rm': [600]},
        {'label': '2 Shells', 'Nm': [900, 900], 'Rm': [600, 900]},
        {'label': '3 Shells', 'Nm': [900, 900, 900], 'Rm': [600, 900, 1200]},
    ]

    P_sel, P_com = {}, {}
    for c in configs:
        label = c['label']
        Nm, Rm = c['Nm'], c['Rm']
        print(f'  {label}...')
        ps, pc = [], []
        for lam_o in lambda_o_sweep:
            lam_u = cfg.D_O * lam_o
            if len(Nm) == 1:
                ps.append(coverage_selection_single_shell(gamma_o, Nm[0], Rm[0], lam_u))
                pc.append(monte_carlo_combining(gamma_o, Nm[0], Rm[0], lam_u, n_samples=mc_n))
            else:
                ps.append(coverage_selection_multi_shell(gamma_o, Nm, Rm, lam_u))
                pc.append(monte_carlo_combining(gamma_o, Nm, Rm, lam_u, n_samples=mc_n))
        P_sel[label] = ps
        P_com[label] = pc

    x_vals = lambda_o_sweep * 1e8  # per 100 km²

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    ax = axes[0]
    setup_axes(ax, 'User Density (per $100\\,\\mathrm{km}^2$)', 'Coverage Probability', '(a) Satellite Selection')
    for i, c in enumerate(configs):
        ax.semilogx(x_vals, P_sel[c['label']], color=COLORS[i], ls='-', marker=MARKERS[i], markevery=4, label=c['label'])
    ax.legend(framealpha=0.9, fontsize=7)
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    setup_axes(ax, 'User Density (per $100\\,\\mathrm{km}^2$)', 'Coverage Probability', '(b) Combining Diversity')
    for i, c in enumerate(configs):
        ax.semilogx(x_vals, P_com[c['label']], color=COLORS[i], ls='-', marker=MARKERS[i], markevery=4, label=c['label'])
    ax.legend(framealpha=0.9, fontsize=7)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig6_user_intensity.png'))


if __name__ == '__main__':
    print('='*60)
    print('论文复现: Coverage Diversity in Mega Satellite Constellations')
    print('IEEE TWC, Vol. 24, No. 11, 2025')
    print('='*60)

    t_start = time.time()
    try:
        run_fig4()
        print(f'  Fig.4 done ({time.time()-t_start:.0f}s)')
    except Exception as e:
        print(f'  Fig.4 FAILED: {e}')

    t1 = time.time()
    try:
        run_fig5()
        print(f'  Fig.5 done ({time.time()-t1:.0f}s)')
    except Exception as e:
        print(f'  Fig.5 FAILED: {e}')

    t2 = time.time()
    try:
        run_fig6()
        print(f'  Fig.6 done ({time.time()-t2:.0f}s)')
    except Exception as e:
        print(f'  Fig.6 FAILED: {e}')

    print(f'\nTotal time: {time.time()-t_start:.0f}s')
    print('Check output/ for figures.')
