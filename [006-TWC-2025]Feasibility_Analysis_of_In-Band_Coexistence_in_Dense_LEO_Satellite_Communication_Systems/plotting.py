"""
绘图模块 — 统一样式, 适配新结果格式
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 150

COLORS = {
    'blue': '#0072B2', 'red': '#D55E00', 'green': '#009E73',
    'purple': '#CC79A7', 'orange': '#E69F00', 'black': '#000000',
    'gray': '#999999',
}


def ecdf(data):
    """经验 CDF"""
    s = np.sort(data)
    return s, np.arange(1, len(s) + 1) / len(s)


def savefig(fig, name, d='output'):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")


def plot_fig2(ant_mod):
    """Fig. 2: 波束方向图"""
    a, g64 = ant_mod.plot_beam_pattern(64, 64)
    _, g32 = ant_mod.plot_beam_pattern(32, 32)
    _, g16 = ant_mod.plot_beam_pattern(16, 16)
    _, g8 = ant_mod.plot_beam_pattern(8, 8)
    m = (a >= -30) & (a <= 30)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(a[m], g64[m], 'k-', lw=1.5, label='64×64 (Sat)')
    ax.plot(a[m], g32[m], color=COLORS['blue'], lw=1.5, label='32×32')
    ax.plot(a[m], g16[m], color=COLORS['red'], lw=1.5, label='16×16')
    ax.plot(a[m], g8[m], color=COLORS['green'], lw=1.5, label='8×8')
    ax.set_xlabel('Angle from Boresight (°)')
    ax.set_ylabel('Normalized Gain (dB)')
    ax.set_title('Fig. 2: Beam Patterns of Phased Array Antennas')
    ax.set_ylim([-50, 5]); ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    savefig(fig, 'fig2_beam_patterns.png')


def plot_fig3():
    """Fig. 3: 频谱效率损失"""
    inr = np.linspace(-30, 10, 500)
    inr_l = 10 ** (inr / 10)
    fig, ax = plt.subplots(figsize=(7, 5))
    for snr_db, c, lb in [(-15, COLORS['blue'], '-15 dB'),
                           (0, COLORS['green'], '0 dB'),
                           (10, COLORS['red'], '10 dB'),
                           (20, COLORS['black'], '20 dB')]:
        s = 10 ** (snr_db / 10)
        r = np.log2(1 + s / (1 + inr_l)) / np.log2(1 + s) * 100
        ax.plot(inr, r, color=c, lw=1.5, label=f'SNR = {lb}')
    for thr in [-12.2, -6, 0]:
        ax.axvline(thr, color='gray', ls=':', alpha=0.6)
    ax.set_xlabel('INR (dB)'); ax.set_ylabel('SE (% of capacity)')
    ax.set_title('Fig. 3: Spectral Efficiency Loss Due to Interference')
    ax.set_xlim([-30, 10]); ax.set_ylim([0, 105]); ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, 'fig3_se_loss.png')


def plot_fig4(R):
    """Fig. 4: 干扰上下界 CDF"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, lb, c, ls in [
        ('inr_abs_max', 'INR$_{max}$(u)', COLORS['red'], '-'),
        ('inr_abs_min', 'INR$_{min}$(u)', COLORS['blue'], '-'),
        ('inr_cond_max', 'INR$_{max}$(u,p*)', COLORS['red'], '--'),
        ('inr_cond_min', 'INR$_{min}$(u,p*)', COLORS['blue'], '--'),
    ]:
        d = np.array(R[key])
        if len(d) == 0: continue
        x, y = ecdf(d)
        ax.plot(x, y, color=c, ls=ls, lw=1.5, label=lb)
    ax.axvline(-12.2, color='gray', ls=':', alpha=0.6, label='-12.2 dB')
    ax.set_xlabel('INR (dB)'); ax.set_ylabel('CDF')
    ax.set_title('Fig. 4: CDFs of Interference Bounds')
    ax.set_xlim([-60, 20]); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    savefig(fig, 'fig4_interference_bounds.png')


def plot_fig5(R):
    """Fig. 5: 可行卫星数量"""
    fig, ax = plt.subplots(figsize=(8, 5))
    tc = {-15: COLORS['purple'], -12.2: COLORS['red'],
          -6: COLORS['orange'], 0: COLORS['green']}
    for thr in [-15, -12.2, -6, 0]:
        d = np.array(R['feasible_counts'][thr])
        if len(d) == 0: continue
        x, y = ecdf(d)
        ax.plot(x, y, color=tc[thr], lw=1.5, label=f'INR$_{{th}}$={thr} dB')
    ax.set_xlabel('$N_s$'); ax.set_ylabel('CDF')
    ax.set_title('Fig. 5: CDF of Feasible Secondary Satellites')
    ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, 'fig5_feasible_satellites.png')


def plot_fig6(R, cfg='32x32'):
    """Fig. 6: 贪心 SINR"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ub = np.array(R['pri_snr_ub'])
    if len(ub): x, y = ecdf(ub); ax.plot(x, y, 'k:', lw=1.5, label='Upper Bound')
    for k, lb, c in [('pri_sinr_gsnr', 'Max-SNR', COLORS['red']),
                      ('pri_sinr_gsinr', 'Max-SINR', COLORS['blue'])]:
        d = np.array(R[k])
        if len(d) == 0: continue
        x, y = ecdf(d); ax.plot(x, y, color=c, lw=1.5, label=lb)
    ax.set_xlabel('Primary SINR (dB)'); ax.set_ylabel('CDF')
    ax.set_title('(a) Primary'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for k, lb, c in [('sec_sinr_gsnr', 'Max-SNR', COLORS['red']),
                      ('sec_sinr_gsinr', 'Max-SINR', COLORS['blue'])]:
        d = np.array(R[k])
        if len(d) == 0: continue
        x, y = ecdf(d); ax.plot(x, y, color=c, lw=1.5, label=lb)
    if len(ub): x, y = ecdf(ub); ax.plot(x, y, 'k:', lw=1.5, label='Upper Bound')
    ax.set_xlabel('Secondary SINR (dB)'); ax.set_ylabel('CDF')
    ax.set_title('(b) Secondary'); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f'Fig. 6: Greedy Selection ({cfg})')
    plt.tight_layout()
    savefig(fig, 'fig6_greedy_sinr.png')


def plot_fig7(R):
    """Fig. 7: 贪心干扰"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, lb, c in [('inr_gsnr', 'Max-SNR', COLORS['red']),
                      ('inr_gsinr', 'Max-SINR', COLORS['blue'])]:
        d = np.array(R[k])
        if len(d) == 0: continue
        x, y = ecdf(d); ax.plot(x, y, color=c, lw=1.5, label=lb)
    for t in [-12.2, -6, 0]:
        ax.axvline(t, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('INR (dB)'); ax.set_ylabel('CDF')
    ax.set_title('Fig. 7: Interference under Greedy Selection')
    ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, 'fig7_greedy_interference.png')


def plot_fig8(R):
    """Fig. 8: 保护性 SINR"""
    tc = {-12.2: COLORS['black'], -6: COLORS['red'], 0: COLORS['green']}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_i, prefix, title in [(0, 'pri', 'Primary'), (1, 'sec', 'Secondary')]:
        ax = axes[ax_i]
        ub = np.array(R['pri_snr_ub'])
        if len(ub) and ax_i == 0:
            x, y = ecdf(ub); ax.plot(x, y, 'k:', lw=1.5, label='Upper Bound')

        for thr in [-12.2, -6.0, 0.0]:
            c = tc[thr]
            for suf, ls, pre in [('psnr', '--', 'Prot.Max-SNR'),
                                  ('psinr', '-', 'Prot.Max-SINR')]:
                k = f'{prefix}_sinr_{suf}'
                d = np.array(R[k][thr])
                if len(d) == 0: continue
                x, y = ecdf(d)
                ax.plot(x, y, color=c, ls=ls, lw=1.5,
                        label=f'{pre},{thr}dB')

        ax.set_xlabel(f'{title} SINR (dB)'); ax.set_ylabel('CDF')
        ax.set_title(f'({chr(97+ax_i)}) {title}'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle('Fig. 8: Protective Selection SINR (32×32)')
    plt.tight_layout()
    savefig(fig, 'fig8_protective_sinr.png')


def plot_fig9(R):
    """Fig. 9: 有用卫星数量"""
    fig, ax = plt.subplots(figsize=(8, 5))
    dc = {1: COLORS['blue'], 2: COLORS['red'], 3: COLORS['green']}
    for delta in [1, 2, 3]:
        for thr in [-12.2]:
            d = np.array(R['useful'].get((delta, thr), []))
            if len(d) == 0: continue
            x, y = ecdf(d)
            ax.plot(x, y, color=dc[delta], lw=1.5,
                    label=f'Δ={delta}dB, INR$_{{th}}$={thr}dB')
    ax.set_xlabel('$\\tilde{N}_s$'); ax.set_ylabel('CDF')
    ax.set_title('Fig. 9: Useful Secondary Satellites')
    ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, 'fig9_useful_satellites.png')


def plot_fig10(R):
    """Fig. 10: 角间距"""
    fig, ax = plt.subplots(figsize=(8, 5))
    tc = {-12.2: COLORS['red'], -6: COLORS['orange'], 0: COLORS['green']}
    for thr in [-12.2, -6, 0]:
        c = tc[thr]
        d1 = np.array(R['ang_sinr'].get(thr, []))
        d2 = np.array(R['ang_snr'].get(thr, []))
        if len(d1):
            x, y = ecdf(d1)
            ax.plot(x, y, color=c, ls='-', lw=1.5, label=f'∠(s*∞,s*), {thr}dB')
        if len(d2):
            x, y = ecdf(d2)
            ax.plot(x, y, color=c, ls='--', lw=1.5, label=f'∠(s†∞,s*), {thr}dB')
    ax.set_xlabel('Angular Separation (°)'); ax.set_ylabel('CDF')
    ax.set_title('Fig. 10: Angular Separation')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    savefig(fig, 'fig10_angular_separation.png')


def plot_fig11(R):
    """Fig. 11: 俯仰角和角间距"""
    fig, ax = plt.subplots(figsize=(8, 5))
    tc = {-12.2: COLORS['red'], -6: COLORS['orange'], 0: COLORS['green']}
    for thr in [-12.2, -6, 0]:
        c = tc[thr]
        d1 = np.array(R['el_sec'].get(thr, []))
        d2 = np.array(R['ang_pri'].get(thr, []))
        if len(d1):
            x, y = ecdf(d1)
            ax.plot(x, y, color=c, ls='-', lw=1.5, label=f'Elev(s*), {thr}dB')
        if len(d2):
            x, y = ecdf(d2)
            ax.plot(x, y, color=c, ls='--', lw=1.5, label=f'∠(s*,p*), {thr}dB')
    ax.set_xlabel('Angle (°)'); ax.set_ylabel('CDF')
    ax.set_title('Fig. 11: Elevation and Angular Separation')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    savefig(fig, 'fig11_elevation_separation.png')


def plot_fig13(R):
    """Fig. 13: 不确定性下可行卫星"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    gc = {0: COLORS['green'], 10: COLORS['blue'], 20: COLORS['orange'],
          30: COLORS['red'], 40: COLORS['purple'], 50: COLORS['black']}

    ax = axes[0]
    for g in [0, 10, 20, 30, 40, 50]:
        d = np.array(R['feas_unc'][g].get(-12.2, []))
        if len(d) == 0: continue
        x, y = ecdf(d)
        ax.plot(x, y, color=gc[g], lw=1.5, label=f'γ={g}°')
    ax.set_xlabel("$N'_s$"); ax.set_ylabel('CDF')
    ax.set_title("(a) CDF at INR$_{th}$=-12.2dB")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    gammas = [0, 10, 20, 30, 40, 50]
    thrs = [-15, -12.2, -6, 0]
    avg = np.zeros((len(gammas), len(thrs)))
    for gi, g in enumerate(gammas):
        for ti, thr in enumerate(thrs):
            d = np.array(R['feas_unc'][g].get(thr, []))
            avg[gi, ti] = np.mean(d) if len(d) else 0
    im = ax.imshow(avg, aspect='auto', origin='lower', cmap='YlOrRd')
    ax.set_xticks(range(len(thrs)))
    ax.set_xticklabels([str(t) for t in thrs])
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([f'{g}' for g in gammas])
    ax.set_xlabel('INR$_{th}$ (dB)'); ax.set_ylabel('γ (°)')
    ax.set_title("(b) Avg $N'_s$")
    plt.colorbar(im, ax=ax)

    fig.suptitle('Fig. 13: Feasible Satellites Under Uncertainty')
    plt.tight_layout()
    savefig(fig, 'fig13_uncertainty_feasible.png')


def plot_fig14(R):
    """Fig. 14: 保障 SINR"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    gc = {0: COLORS['green'], 10: COLORS['blue'], 20: COLORS['orange'],
          30: COLORS['red'], 40: COLORS['purple'], 50: COLORS['black']}

    ax = axes[0]
    ub = np.array(R['pri_snr_ub'])
    ref = np.median(ub) if len(ub) else 0
    for g in [0, 10, 20, 30, 40, 50]:
        d = np.array(R['guar_sinr'].get(g, []))
        if len(d) == 0: continue
        gap = d - ref
        x, y = ecdf(gap)
        ax.plot(x, y, color=gc[g], lw=1.5, label=f'γ={g}°')
    ax.set_xlabel('Guaranteed SINR Gap (dB)'); ax.set_ylabel('CDF')
    ax.set_title('(a) Guaranteed SINR (INR$_{th}$=-12.2dB)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    gs = [0, 10, 20, 30, 40, 50]
    meds = [np.median(R['guar_sinr'][g]) if len(R['guar_sinr'][g]) else np.nan
            for g in gs]
    ax.plot(gs, meds, 'o-', color=COLORS['blue'], lw=2)
    ax.set_xlabel('γ (°)'); ax.set_ylabel('Median Guaranteed SINR (dB)')
    ax.set_title('(b) Median vs Uncertainty'); ax.grid(alpha=0.3)

    fig.suptitle('Fig. 14: Guaranteed SINR Under Uncertainty')
    plt.tight_layout()
    savefig(fig, 'fig14_guaranteed_sinr.png')
