"""
主仿真脚本 — 复现 Fig.9, Fig.10, Fig.12
论文: Enabling Scalable Distributed Beamforming via Networked LEO Satellites Toward 6G
期刊: IEEE TWC, 2026
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PS_DEFAULT, S_DEFAULT, U_DEFAULT, N_H_DEFAULT, N_V_DEFAULT
from channel_model import (
    generate_scenario, mrt_beamforming, zf_beamforming,
    s3_mrt, compute_sum_rate,
    wmmse_centralized, wmmse_ring, wmmse_star
)
from plotting import plot_fig9, plot_fig10, plot_fig12

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
N_MC = 3  # 蒙特卡洛次数

print("=" * 60)
print("论文复现: Enabling Scalable Distributed Beamforming")
print("         via Networked LEO Satellites Toward 6G")
print("目标: Fig.9, Fig.10, Fig.12")
print("=" * 60)

SCHEME_NAMES = ['Central', 'Ring', 'Star', 'MRT', 'MRT-S3']


def run_single(S, U, Nh, Nv, Ps_dbm, seed):
    """运行单次仿真"""
    params = generate_scenario(S, U, Nh=Nh, Nv=Nv, Ps_dbm=Ps_dbm, seed=seed)

    results = {}
    # WMMSE方案
    wc, _ = wmmse_centralized(params, max_iter=40)
    results['Central'] = compute_sum_rate(wc, params)

    wr, _ = wmmse_ring(params, max_iter=40)
    results['Ring'] = compute_sum_rate(wr, params)

    ws, _ = wmmse_star(params, max_iter=40)
    results['Star'] = compute_sum_rate(ws, params)

    # 基线方案
    wm = mrt_beamforming(params)
    results['MRT'] = compute_sum_rate(wm, params)

    ws3 = s3_mrt(params)
    results['MRT-S3'] = compute_sum_rate(ws3, params)

    return results


def monte_carlo(run_fn, n_mc=N_MC):
    """蒙特卡洛平均"""
    accum = None
    for mc in range(n_mc):
        seed = 42 + mc * 100
        r = run_fn(seed)
        if accum is None:
            accum = {k: 0.0 for k in r}
        for k in r:
            accum[k] += r[k] / n_mc
    return accum


# ==================================================================
# Fig. 9: Sum rate vs Power Budget
# ==================================================================
print("\n--- Fig. 9: Sum Rate vs Power Budget ---")
Ps_range = list(range(30, 56, 5))
fig9 = {n: [] for n in SCHEME_NAMES}

for ps in Ps_range:
    print(f"  Ps = {ps} dBm ...", end="", flush=True)
    rates = monte_carlo(lambda seed: run_single(S_DEFAULT, U_DEFAULT, N_H_DEFAULT, N_V_DEFAULT, ps, seed))
    for n in SCHEME_NAMES:
        fig9[n].append(rates[n])
    print(f" Central={rates['Central']:.1f}, MRT={rates['MRT']:.1f}")

plot_fig9(Ps_range, fig9, OUTPUT_DIR)

# ==================================================================
# Fig. 10: Sum rate vs Antenna Number
# ==================================================================
print("\n--- Fig. 10: Sum Rate vs Antenna Number ---")
NhNv_list = [(8, 8), (11, 12), (16, 16), (22, 24)]
fig10 = {n: [] for n in SCHEME_NAMES}

for Nh, Nv in NhNv_list:
    N = Nh * Nv
    print(f"  N = {N} ({Nh}x{Nv}) ...", end="", flush=True)
    rates = monte_carlo(lambda seed: run_single(S_DEFAULT, U_DEFAULT, Nh, Nv, 50, seed))
    for n in SCHEME_NAMES:
        fig10[n].append(rates[n])
    print(f" Central={rates['Central']:.1f}")

N_actual = [Nh * Nv for Nh, Nv in NhNv_list]
plot_fig10(N_actual, fig10, OUTPUT_DIR)

# ==================================================================
# Fig. 12: Sum rate vs Satellite Number
# ==================================================================
print("\n--- Fig. 12: Sum Rate vs Satellite Number ---")
S_range = [2, 3, 4, 5, 6]
fig12 = {n: [] for n in SCHEME_NAMES}

for S in S_range:
    print(f"  S = {S} ...", end="", flush=True)
    rates = monte_carlo(lambda seed: run_single(S, U_DEFAULT, N_H_DEFAULT, N_V_DEFAULT, 50, seed))
    for n in SCHEME_NAMES:
        fig12[n].append(rates[n])
    print(f" Central={rates['Central']:.1f}")

plot_fig12(S_range, fig12, OUTPUT_DIR)

print("\n" + "=" * 60)
print("复现完成! 结果已保存到 output/ 目录")
print("=" * 60)
