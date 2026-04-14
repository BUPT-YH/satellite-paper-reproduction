"""
主仿真模块 (高效版)
- 用 10x10 协方差矩阵 (N_indep=10 次独立采样)
- DTC 用特征值分解, 避免 sqrtm/logm/expm
- 所有图表仿真逻辑
"""

import numpy as np
from scipy.linalg import sqrtm
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from info_geometry import (
    latlon_to_cartesian, calc_link_distance, calc_off_axis_angle,
    calc_elevation_angle, antenna_gain_s465, antenna_gain_s1528,
    calc_received_power_dBW, calc_signal_amplitude,
    generate_cov_free, generate_cov_interf,
    center_matrix_AIRM, center_matrix_SKLD,
    calc_DTC_AIRM_fast, calc_DTC_SKLD_fast,
    calc_threshold, calc_JR_DTC, potential_function,
    get_signal_params,
)

# 协方差矩阵维度 = 独立采样次数
COV_DIM = N_independent_samples  # = 10


def compute_all_signal_params():
    """计算所有信号参数"""
    # 无干扰: OneWeb 卫星 -> OneWeb 地球站
    dist_OW = calc_link_distance(lat_OW_es, lon_OW_es, h_OW_es,
                                  lat_OW_sat, lon_OW_sat, h_OW_sat)
    Pr_OW_dBW = calc_received_power_dBW(
        Pt_OW, Gt_OW_peak, Gr_OW_es, freq, dist_OW,
        lat_OW_es, lon_OW_es, h_OW_es, lat_OW_sat, lon_OW_sat, h_OW_sat)
    A0 = calc_signal_amplitude(Pr_OW_dBW)
    sigma0 = np.sqrt(k_boltz * T_noise * bandwidth)

    print(f"  OneWeb 链路距离: {dist_OW:.1f} km")
    print(f"  无干扰接收功率: {Pr_OW_dBW:.2f} dBW")
    print(f"  A0 = {A0:.4e}, sigma0 = {sigma0:.4e}")

    # 干扰参数
    A_interf, sigma_interf = [], []
    sl_params = [
        (Pt_SL1, Gt_SL1_peak, HPBW_SL1, lat_SL1, lon_SL1, h_SL1),
        (Pt_SL2, Gt_SL2_peak, HPBW_SL2, lat_SL2, lon_SL2, h_SL2),
    ]
    for idx, (Pt, Gt, hpbw, lat_s, lon_s, h_s) in enumerate(sl_params):
        dist = calc_link_distance(lat_OW_es, lon_OW_es, h_OW_es, lat_s, lon_s, h_s)
        psi = calc_off_axis_angle(lat_OW_es, lon_OW_es, h_OW_es,
                                   lat_OW_sat, lon_OW_sat, h_OW_sat, lat_s, lon_s, h_s)
        Gr = antenna_gain_s465(Gr_OW_es, psi, D_OW_es)
        Gt_eff = antenna_gain_s1528(Gt, 0, hpbw)
        Pr_dBW = calc_received_power_dBW(Pt, Gt_eff, Gr, freq, dist,
                                          lat_OW_es, lon_OW_es, h_OW_es, lat_s, lon_s, h_s)
        Ai = calc_signal_amplitude(Pr_dBW)
        si = sigma0 * 0.1
        A_interf.append(Ai)
        sigma_interf.append(si)
        print(f"  干扰卫星{idx+1}: dist={dist:.1f}km, psi={psi:.2f}°, Pr={Pr_dBW:.2f}dBW, A={Ai:.4e}")

    return A0, sigma0, A_interf, sigma_interf


# ==================== Fig. 3 ====================

def simulate_fig3(A0, sigma0, Q=1000):
    """AIRM 平均 DTC vs 迭代次数"""
    print("\n=== Fig. 3 ===")
    results = {}

    for M in [15, 100]:
        # 生成 Q 个无干扰协方差矩阵
        covs = [generate_cov_free(A0, sigma0, COV_DIM) for _ in range(Q)]

        avg_dtc_list = []
        R_center = np.mean(covs, axis=0).copy()

        for it in range(AIRM_max_iter):
            dtc_vals = [calc_DTC_AIRM_fast(R_center, Rk) for Rk in covs]
            avg_dtc_list.append(np.mean(dtc_vals))

            # 迭代更新中心
            R_center_inv = np.linalg.inv(R_center)
            log_sum = np.zeros((COV_DIM, COV_DIM))
            for Rk in covs:
                inner = R_center_inv @ Rk
                eigv, eigvec = np.linalg.eigh(inner)
                eigv = np.maximum(eigv, 1e-15)
                log_sum += eigvec @ np.diag(np.log(eigv)) @ eigvec.T

            log_mean = log_sum / Q
            eigv_c, eigvec_c = np.linalg.eigh(R_center)
            eigv_c = np.maximum(eigv_c, 1e-15)
            R_half = eigvec_c @ np.diag(np.sqrt(eigv_c)) @ eigvec_c.T

            eigv_l, eigvec_l = np.linalg.eigh(log_mean)
            exp_log = eigvec_l @ np.diag(np.exp(AIRM_step * eigv_l)) @ eigvec_l.T
            R_center = R_half @ exp_log @ R_half
            R_center = (R_center + R_center.T) / 2

        results[M] = avg_dtc_list
        print(f"  M={M}: 最终 avg DTC = {avg_dtc_list[-1]:.4f}")

    return results[15], results[100]


# ==================== Fig. 4 & 5 ====================

def simulate_fig4_5(A0, sigma0, Q=1000):
    """AIRM 和 SKLD 的 DTC 散点图和阈值"""
    print("\n=== Fig. 4 & 5 ===")
    airm_r, skld_r = {}, {}

    for M in [15, 100]:
        covs = [generate_cov_free(A0, sigma0, COV_DIM) for _ in range(Q)]

        # AIRM
        R_ca = center_matrix_AIRM(covs, AIRM_max_iter, AIRM_step)
        dtc_a = np.array([calc_DTC_AIRM_fast(R_ca, Rk) for Rk in covs])
        th_a = calc_threshold(dtc_a, Pf)
        airm_r[M] = (dtc_a, th_a)
        print(f"  M={M} AIRM: range=[{dtc_a.min():.4f}, {dtc_a.max():.4f}], threshold={th_a:.4f}")

        # SKLD
        R_cs = center_matrix_SKLD(covs)
        dtc_s = np.array([calc_DTC_SKLD_fast(R_cs, Rk) for Rk in covs])
        th_s = calc_threshold(dtc_s, Pf)
        skld_r[M] = (dtc_s, th_s)
        print(f"  M={M} SKLD: range=[{dtc_s.min():.4f}, {dtc_s.max():.4f}], threshold={th_s:.4f}")

    return airm_r, skld_r


# ==================== Fig. 6 ====================

def simulate_fig6(A0, sigma0, A_interf, sigma_interf, Q=500):
    """JR-DTC 随采样点数变化"""
    print("\n=== Fig. 6 ===")
    sp = np.array(list(range(15, 51, 5)) + [60, 70, 80, 90, 100])

    jr_airm = {f'Case{i}': ([], []) for i in range(4)}
    jr_skld = {f'Case{i}': ([], []) for i in range(4)}

    for M in sp:
        print(f"  M={M}...", end='', flush=True)

        # 无干扰中心矩阵
        covs_free = [generate_cov_free(A0, sigma0, COV_DIM) for _ in range(Q)]
        R_ca = center_matrix_AIRM(covs_free, AIRM_max_iter, AIRM_step)
        R_cs = center_matrix_SKLD(covs_free)

        for ci in range(4):
            _, _, Ai, si = get_signal_params(ci, A0, sigma0, A_interf, sigma_interf)
            covs_case = [generate_cov_interf(A0, sigma0, Ai, si, COV_DIM) for _ in range(Q)]

            dtc_a = np.array([calc_DTC_AIRM_fast(R_ca, Rk) for Rk in covs_case])
            dtc_s = np.array([calc_DTC_SKLD_fast(R_cs, Rk) for Rk in covs_case])

            la, ua = calc_JR_DTC(dtc_a, Pf)
            ls, us = calc_JR_DTC(dtc_s, Pf)

            jr_airm[f'Case{ci}'][0].append(la)
            jr_airm[f'Case{ci}'][1].append(ua)
            jr_skld[f'Case{ci}'][0].append(ls)
            jr_skld[f'Case{ci}'][1].append(us)

        print(" done")

    for cn in jr_airm:
        jr_airm[cn] = (np.array(jr_airm[cn][0]), np.array(jr_airm[cn][1]))
        jr_skld[cn] = (np.array(jr_skld[cn][0]), np.array(jr_skld[cn][1]))

    return jr_airm, jr_skld, sp


# ==================== Fig. 7 ====================

def simulate_fig7(A0, sigma0, A_interf, sigma_interf, Q_train=500, Q_test=1000):
    """正确判断概率 vs 采样点数"""
    print("\n=== Fig. 7 ===")
    sp = np.array(list(range(15, 51, 5)) + [60, 70, 80, 90, 100])

    prob_airm = {f'Case{i}': [] for i in range(4)}
    prob_skld = {f'Case{i}': [] for i in range(4)}
    prob_energy = []

    for M in sp:
        print(f"  M={M}...", end='', flush=True)

        # 训练: 计算各场景的 JR-DTC
        jr_a, jr_s = {}, {}
        covs_free_train = [generate_cov_free(A0, sigma0, COV_DIM) for _ in range(Q_train)]
        R_ca = center_matrix_AIRM(covs_free_train, AIRM_max_iter, AIRM_step)
        R_cs = center_matrix_SKLD(covs_free_train)

        for ci in range(4):
            _, _, Ai, si = get_signal_params(ci, A0, sigma0, A_interf, sigma_interf)
            covs_case = [generate_cov_interf(A0, sigma0, Ai, si, COV_DIM) for _ in range(Q_train)]
            dtc_a = np.array([calc_DTC_AIRM_fast(R_ca, Rk) for Rk in covs_case])
            dtc_s = np.array([calc_DTC_SKLD_fast(R_cs, Rk) for Rk in covs_case])
            jr_a[ci] = calc_JR_DTC(dtc_a, Pf)
            jr_s[ci] = calc_JR_DTC(dtc_s, Pf)

        # 测试
        for ci in range(4):
            _, _, Ai, si = get_signal_params(ci, A0, sigma0, A_interf, sigma_interf)
            correct_a, correct_s, correct_e = 0, 0, 0
            n_test = min(Q_test, 500)

            for _ in range(n_test):
                R_test = generate_cov_interf(A0, sigma0, Ai, si, COV_DIM)

                da = calc_DTC_AIRM_fast(R_ca, R_test)
                if jr_a[ci][0] <= da <= jr_a[ci][1]:
                    correct_a += 1

                ds = calc_DTC_SKLD_fast(R_cs, R_test)
                if jr_s[ci][0] <= ds <= jr_s[ci][1]:
                    correct_s += 1

            prob_airm[f'Case{ci}'].append(correct_a / n_test)
            prob_skld[f'Case{ci}'].append(correct_s / n_test)

        # 能量检测法 (只判断有无干扰, 不区分具体场景)
        correct_e = 0
        n_test_e = min(Q_test, 500)
        for _ in range(n_test_e):
            R_test = generate_cov_free(A0, sigma0, COV_DIM)
            # 能量 = trace(R)
            energy_test = np.trace(R_test)
            R_free = generate_cov_free(A0, sigma0, COV_DIM)
            energy_free = np.trace(R_free)
            if abs(energy_test - energy_free) < energy_free * 0.5:
                correct_e += 1
        prob_energy.append(correct_e / n_test_e)

        vals_a = [prob_airm[f'Case{i}'][-1] for i in range(4)]
        vals_s = [prob_skld[f'Case{i}'][-1] for i in range(4)]
        print(f" AIRM={[f'{v:.2f}' for v in vals_a]}, SKLD={[f'{v:.2f}' for v in vals_s]}, Energy={prob_energy[-1]:.2f}")

    return prob_airm, prob_skld, prob_energy, sp


# ==================== Fig. 8 ====================

def simulate_fig8():
    """仿射嵌入 3D 曲面图"""
    print("\n=== Fig. 8 ===")
    lat_arr = np.arange(vis_lat_range[0], vis_lat_range[1] + 0.1, 0.1)
    lon_arr = np.arange(vis_lon_range[0], vis_lon_range[1] + 0.1, 0.1)
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
    Z = np.zeros_like(lon_grid)

    for i, lat_sl in enumerate(lat_arr):
        for j, lon_sl in enumerate(lon_arr):
            total_power = 0
            for (lat_es, lon_es) in earth_stations:
                dist = calc_link_distance(lat_es, lon_es, 0, lat_sl, lon_sl, h_SL1)
                psi = calc_off_axis_angle(lat_es, lon_es, 0,
                                          lat_OW_sat, lon_OW_sat, h_OW_sat,
                                          lat_sl, lon_sl, h_SL1)
                Gr = antenna_gain_s465(Gr_OW_es, psi, D_OW_es)
                Gt = antenna_gain_s1528(Gt_SL1_peak, 0, HPBW_SL1)
                Pr_dBW = calc_received_power_dBW(Pt_SL1, Gt, Gr, freq, dist)
                Ai = calc_signal_amplitude(Pr_dBW)
                total_power += Ai**2

            R = np.eye(N_independent_samples) * (total_power + 1e-20)
            Z[i, j] = potential_function(R)

        if (i + 1) % 10 == 0:
            print(f"    进度: {i+1}/{len(lat_arr)}")

    return lon_grid, lat_grid, Z
