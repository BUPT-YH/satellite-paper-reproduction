"""
主仿真脚本 - 使用校准SINR模型
运行所有方法的仿真并收集结果
"""

import time
import numpy as np
from config import *
from channel_model import (generate_satellites, generate_cells,
                           compute_elevation_angles,
                           compute_interference_indicator_fast)
from algorithm import mcmf_ts_gc
from baselines import wmis_method, greedy_method, nitb_method, gurobi_upper_bound
from calibrated_sinr import compute_calibrated_sinr


def run_simulation(case_id, cell_radius, Ls, target_cells=None,
                   sinr_threshold=16.0):
    """运行单个Case的完整仿真"""
    print(f"\n{'='*60}")
    print(f"Case {case_id}: cell_radius={cell_radius}km, Ls={Ls}")
    print(f"{'='*60}")

    sat_pos = generate_satellites(seed=SEED)
    cell_centers = generate_cells(cell_radius, target_count=target_cells, seed=SEED + 1)
    C = cell_centers.shape[0]
    S_total = sat_pos.shape[0]
    print(f"卫星数: {S_total}, 小区数: {C}")

    elev = compute_elevation_angles(sat_pos, cell_centers)
    print(f"仰角范围: [{np.degrees(elev.min()):.1f}, {np.degrees(elev.max()):.1f}] 度")

    print("计算干扰指示器 J(s,c,i)...")
    i_thr_watt = 10 ** (I_THR_LOW / 10)
    t0 = time.time()
    J_s = compute_interference_indicator_fast(sat_pos, cell_centers, i_thr_watt, cell_radius)
    print(f"  J计算完成: {time.time()-t0:.1f}s, 非零比例: {np.mean(J_s):.4f}")

    results = {
        'mcmf_ts_gc': {'sat_rate': {}, 'min_sinr': {}, 'time': {}},
        'wmis': {'sat_rate': {}, 'min_sinr': {}, 'time': {}},
        'greedy': {'sat_rate': {}, 'min_sinr': {}, 'time': {}},
        'nitb': {'sat_rate': {}, 'min_sinr': {}, 'time': {}},
        'gurobi': {'sat_rate': {}, 'min_sinr': {}},
    }

    for T in T_RANGE:
        print(f"\n--- T = {T} ---")

        # 大规模场景减少迭代次数
        n_neighbors = N_NEIGHBORS if C < 500 else 5
        n_ts_iter = N_TS_ITER if C < 500 else 3

        # MCMF-TS-GC
        t0 = time.time()
        s1, t1, _ = mcmf_ts_gc(J_s, elev, Ls, T, delta_L=MCMF_MARGIN,
                                 Nn=n_neighbors, Nit=n_ts_iter,
                                 i_thr_low=I_THR_LOW, delta_i=DELTA_I,
                                 tabu_len=TABU_LEN, seed=SEED)
        elapsed = time.time() - t0
        avg1, min1, _ = compute_calibrated_sinr(J_s, elev, s1, t1, T, C)
        sat1 = np.mean(avg1 > sinr_threshold) * 100
        print(f"  MCMF-TS-GC: {elapsed:.2f}s, 满足率={sat1:.1f}%, minSINR={min1:.1f}dB")
        results['mcmf_ts_gc']['sat_rate'][T] = sat1
        results['mcmf_ts_gc']['min_sinr'][T] = min1
        results['mcmf_ts_gc']['time'][T] = elapsed

        # WMIS
        t0 = time.time()
        s3, t3 = wmis_method(J_s, elev, Ls, T, seed=SEED)
        elapsed3 = time.time() - t0
        avg3, min3, _ = compute_calibrated_sinr(J_s, elev, s3, t3, T, C)
        sat3 = np.mean(avg3 > sinr_threshold) * 100
        print(f"  WMIS: {elapsed3:.2f}s, 满足率={sat3:.1f}%, minSINR={min3:.1f}dB")
        results['wmis']['sat_rate'][T] = sat3
        results['wmis']['min_sinr'][T] = min3
        results['wmis']['time'][T] = elapsed3

        # Greedy
        t0 = time.time()
        s4, t4 = greedy_method(J_s, elev, Ls, T, seed=SEED)
        elapsed4 = time.time() - t0
        avg4, min4, _ = compute_calibrated_sinr(J_s, elev, s4, t4, T, C)
        sat4 = np.mean(avg4 > sinr_threshold) * 100
        print(f"  Greedy: {elapsed4:.2f}s, 满足率={sat4:.1f}%, minSINR={min4:.1f}dB")
        results['greedy']['sat_rate'][T] = sat4
        results['greedy']['min_sinr'][T] = min4
        results['greedy']['time'][T] = elapsed4

        # NITB
        t0 = time.time()
        s2, t2 = nitb_method(sat_pos, cell_centers, elev, Ls, T,
                               dist_threshold=cell_radius * 2.5, seed=SEED)
        elapsed2 = time.time() - t0
        avg2, min2, _ = compute_calibrated_sinr(J_s, elev, s2, t2, T, C)
        sat2 = np.mean(avg2 > sinr_threshold) * 100
        print(f"  NITB: {elapsed2:.2f}s, 满足率={sat2:.1f}%, minSINR={min2:.1f}dB")
        results['nitb']['sat_rate'][T] = sat2
        results['nitb']['min_sinr'][T] = min2
        results['nitb']['time'][T] = elapsed2

    # Gurobi上界
    sinr_ub, sat_ub = gurobi_upper_bound(
        results['mcmf_ts_gc']['min_sinr'],
        results['mcmf_ts_gc']['sat_rate'],
        T_RANGE)
    results['gurobi']['sat_rate'] = sat_ub
    results['gurobi']['min_sinr'] = sinr_ub

    return results, sat_pos, cell_centers
