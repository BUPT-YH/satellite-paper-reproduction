"""
快速复现脚本 - 使用预计算和向量化加速
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as cfg
from stochastic_geometry import (
    coverage_selection_single_shell, coverage_selection_multi_shell,
    phi_max_m, avg_interference, avg_n_satellites, path_loss, p_los, pdf_B,
)
from plotting import FIG_DOUBLE, FIG_SINGLE, COLORS, MARKERS, setup_axes, save_fig
from scipy import integrate

OUTPUT = 'output'

# ========== 预计算参数 ==========
def precompute_shell(Rm, Nm, lambda_u=None):
    """预计算壳层参数"""
    if lambda_u is None:
        lambda_u = cfg.LAMBDA_U
    pm = phi_max_m(Rm)
    I = avg_interference(Rm, lambda_u)
    Nb = avg_n_satellites(Nm, Rm)
    # 预计算路径损耗查找表
    phis = np.linspace(1e-4, pm, 200)
    losses = np.array([path_loss(p, Rm) for p in phis])
    plos_vals = np.array([p_los(p, Rm) for p in phis])
    return {'Rm': Rm, 'Nm': Nm, 'phi_max': pm, 'I_bar': I, 'N_bar': Nb,
            'phis': phis, 'losses': losses, 'plos': plos_vals, 'lambda_u': lambda_u}


def fast_mc_selection(gamma_o, shells, n_samples=5000):
    """快速向量化 MC - 选择分集"""
    n_covered = 0
    for _ in range(n_samples):
        best = 0.0
        for s in shells:
            n_sat = np.random.poisson(s['N_bar'])
            if n_sat == 0:
                continue
            # 从 f_B 采样位置
            u = np.random.uniform(0, 1, n_sat)
            phis = np.arccos(1 - u * (1 - np.cos(s['phi_max'])))
            for phi in phis:
                l = path_loss(phi, s['Rm'])
                p_LoS = p_los(phi, s['Rm'])
                if np.random.uniform() < p_LoS:
                    zeta = 10 ** (np.random.normal(-cfg.MU_LOS, cfg.SIGMA_LOS) / 10)
                else:
                    zeta = 10 ** (np.random.normal(-cfg.MU_NLOS, cfg.SIGMA_NLOS) / 10)
                sinr = cfg.RHO_T * cfg.GT * cfg.GR * l * zeta / (s['I_bar'] + cfg.WS)
                if sinr > best:
                    best = sinr
        if best >= gamma_o:
            n_covered += 1
    return n_covered / n_samples


def fast_mc_combining(gamma_o, shells, n_samples=5000):
    """快速 MC - 合并分集 (按论文模型: Σ S_n/(I_m+Ws))"""
    n_covered = 0
    for _ in range(n_samples):
        total = 0.0
        for s in shells:
            n_sat = np.random.poisson(s['N_bar'])
            if n_sat == 0:
                continue
            u = np.random.uniform(0, 1, n_sat)
            phis = np.arccos(1 - u * (1 - np.cos(s['phi_max'])))
            for phi in phis:
                l = path_loss(phi, s['Rm'])
                p_LoS = p_los(phi, s['Rm'])
                if np.random.uniform() < p_LoS:
                    zeta = 10 ** (np.random.normal(-cfg.MU_LOS, cfg.SIGMA_LOS) / 10)
                else:
                    zeta = 10 ** (np.random.normal(-cfg.MU_NLOS, cfg.SIGMA_NLOS) / 10)
                total += cfg.RHO_T * cfg.GT * cfg.GR * l * zeta / (s['I_bar'] + cfg.WS)
        if total >= gamma_o:
            n_covered += 1
    return n_covered / n_samples


if __name__ == '__main__':
    t0 = time.time()
    print('论文复现: Coverage Diversity in Mega Satellite Constellations')
    print('IEEE TWC 2025 - 快速版\n')

    # 预计算
    ss = precompute_shell(600, 900)
    ms = [precompute_shell(600, 900), precompute_shell(900, 400), precompute_shell(1200, 100)]
    sl = [precompute_shell(335.9, 2493), precompute_shell(340.8, 2478), precompute_shell(345.6, 2547)]

    gamma_db = np.arange(-25, 2, 2.0)
    gamma_lin = 10 ** (gamma_db / 10)
    mc_n = 3000

    # ========== Fig.4 ==========
    print('Fig.4...')
    P_sel_ss = [coverage_selection_single_shell(g, 900, 600) for g in gamma_lin]
    P_sel_ms = [coverage_selection_multi_shell(g, [900,400,100], [600,900,1200]) for g in gamma_lin]
    P_com_ss = [fast_mc_combining(g, [ss], mc_n) for g in gamma_lin]
    P_com_ms = [fast_mc_combining(g, ms, mc_n) for g in gamma_lin]

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    ax = axes[0]
    setup_axes(ax, 'SINR Threshold $\\gamma_0$ (dB)', 'Coverage Probability', 'Satellite Selection')
    ax.semilogy(gamma_db, P_sel_ss, color=COLORS[0], ls='-', marker=MARKERS[0], label='1-Shell Analytical')
    ax.semilogy(gamma_db, P_sel_ms, color=COLORS[2], ls='-', marker=MARKERS[2], label='3-Shell Analytical')
    ax.legend(fontsize=7, framealpha=0.9, edgecolor='none')
    ax.set_ylim(1e-3, 1.05)
    ax = axes[1]
    setup_axes(ax, 'SINR Threshold $\\gamma_0$ (dB)', 'Coverage Probability', 'Combining Diversity')
    ax.semilogy(gamma_db, P_com_ss, color=COLORS[1], ls='-', marker=MARKERS[1], label='1-Shell MC')
    ax.semilogy(gamma_db, P_com_ms, color=COLORS[3], ls='-', marker=MARKERS[3], label='3-Shell MC')
    ax.legend(fontsize=7, framealpha=0.9, edgecolor='none')
    ax.set_ylim(1e-3, 1.05)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUTPUT, 'fig4_coverage_comparison.png'))
    print(f'  Fig.4 done ({time.time()-t0:.0f}s)')

    # ========== Fig.5 ==========
    print('Fig.5...')
    t1 = time.time()
    P_sel_sl = [coverage_selection_multi_shell(g, [2493,2478,2547], [335.9,340.8,345.6]) for g in gamma_lin]
    P_com_sl = [fast_mc_combining(g, sl, mc_n) for g in gamma_lin]

    fig, ax = plt.subplots(1, 1, figsize=FIG_SINGLE)
    setup_axes(ax, 'SINR Threshold $\\gamma_0$ (dB)', 'Coverage Probability')
    ax.semilogy(gamma_db, P_sel_sl, color=COLORS[0], ls='-', marker=MARKERS[0], label='Selection')
    ax.semilogy(gamma_db, P_com_sl, color=COLORS[1], ls='-', marker=MARKERS[1], label='Combining')
    ax.legend(framealpha=0.9, edgecolor='none')
    ax.set_ylim(1e-3, 1.05)
    ax.set_title('Starlink Phase 2')
    fig.tight_layout()
    save_fig(fig, os.path.join(OUTPUT, 'fig5_starlink_comparison.png'))
    print(f'  Fig.5 done ({time.time()-t1:.0f}s)')

    # ========== Fig.6 ==========
    print('Fig.6...')
    t2 = time.time()
    gamma_o = 10 ** (-20 / 10)
    lam_sweep = np.logspace(-12, -6, 20)
    configs_6 = [
        {'label': '1 Shell', 'Nm': [900], 'Rm': [600]},
        {'label': '2 Shells', 'Nm': [900, 900], 'Rm': [600, 900]},
        {'label': '3 Shells', 'Nm': [900, 900, 900], 'Rm': [600, 900, 1200]},
    ]
    P6_sel, P6_com = {}, {}
    for c in configs_6:
        ps, pc = [], []
        for lo in lam_sweep:
            lu = cfg.D_O * lo
            if len(c['Nm']) == 1:
                ps.append(coverage_selection_single_shell(gamma_o, c['Nm'][0], c['Rm'][0], lu))
                sh = [precompute_shell(c['Rm'][0], c['Nm'][0], lu)]
            else:
                ps.append(coverage_selection_multi_shell(gamma_o, c['Nm'], c['Rm'], lu))
                sh = [precompute_shell(c['Rm'][i], c['Nm'][i], lu) for i in range(len(c['Nm']))]
            pc.append(fast_mc_combining(gamma_o, sh, 2000))
        P6_sel[c['label']] = ps
        P6_com[c['label']] = pc
        print(f'  {c["label"]} done')

    xv = lam_sweep * 1e8
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE)
    for idx, (data, title) in enumerate([(P6_sel, '(a) Selection'), (P6_com, '(b) Combining')]):
        ax = axes[idx]
        setup_axes(ax, 'User Density (per $100\\,\\mathrm{km}^2$)', 'Coverage Probability', title)
        for i, c in enumerate(configs_6):
            ax.semilogx(xv, data[c['label']], color=COLORS[i], ls='-', marker=MARKERS[i], markevery=3, label=c['label'])
        ax.legend(fontsize=7, framealpha=0.9)
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUTPUT, 'fig6_user_intensity.png'))
    print(f'  Fig.6 done ({time.time()-t2:.0f}s)')

    print(f'\nAll done in {time.time()-t0:.0f}s. Check output/.')
