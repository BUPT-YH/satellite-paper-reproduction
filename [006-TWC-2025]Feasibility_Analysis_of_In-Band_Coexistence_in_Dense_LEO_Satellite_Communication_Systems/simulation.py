"""
优化仿真模块: 向量化实现, 无 Python 内循环
复现论文 Fig. 4-14
"""

import numpy as np
from config import (MIN_ELEVATION, INR_THRESHOLDS, CITIES,
                    TIME_RESOLUTION, SIM_DURATION, NOISE_PSD,
                    USER_ANTENNA_CONFIGS, R_EARTH, MU_EARTH, OMEGA_EARTH,
                    PRIMARY_EIRP, SECONDARY_EIRP)
from antenna import max_upa_gain_dbi
from channel import total_path_loss_db, compute_snr_db, compute_sinr_db
from constellation import (build_constellation, get_satellite_positions,
                           get_user_position, get_local_frame, compute_geometry)


# ===================== 向量化 UPA 增益 =====================

def _array_factor_sq(psi, N):
    """向量化阵列因子 |AF|^2"""
    result = np.full_like(psi, float(N ** 2), dtype=float)
    large = np.abs(psi) >= 1e-10
    if np.any(large):
        p = psi[large]
        num = np.sin(N * p / 2) ** 2
        den = np.sin(p / 2) ** 2
        result[large] = num / np.maximum(den, 1e-30)
    return result


def vec_upa_gain(eval_dirs, steer_dir, nx, ny, east, north):
    """
    向量化 UPA 增益 (dBi)
    eval_dirs: (N, 3) or (3,), steer_dir: (3,)
    """
    if eval_dirs.ndim == 1:
        eval_dirs = eval_dirs[np.newaxis, :]

    du = eval_dirs @ east - np.dot(steer_dir, east)
    dv = eval_dirs @ north - np.dot(steer_dir, north)

    gx = _array_factor_sq(np.pi * du, nx)
    gy = _array_factor_sq(np.pi * dv, ny)

    max_g = 10 * np.log10(nx * ny)
    gain_lin = (gx * gy) / (nx * ny) ** 2
    return 10 * np.log10(np.maximum(gain_lin, 1e-30)) + max_g


def compute_inr_vec(eirp_w, gtx_dbi, grx_dbi, pl_db):
    """向量化 INR (dB)"""
    eirp_db = 10 * np.log10(eirp_w)
    noise_db = 10 * np.log10(NOISE_PSD)
    return eirp_db + gtx_dbi + grx_dbi - pl_db - noise_db


# ===================== 主仿真函数 =====================

def run_simulation(starlink_params, kuiper_params, user_config='32x32',
                   verbose=True):
    sl_raan, sl_ma0, sl_radius, sl_incl = starlink_params
    kp_raan, kp_ma0, kp_radius, kp_incl = kuiper_params

    user_nx, user_ny = USER_ANTENNA_CONFIGS[user_config]
    n_time_steps = int(SIM_DURATION / TIME_RESOLUTION)

    sat_g = 10 * np.log10(64 * 64)     # 36.1 dBi
    usr_g = 10 * np.log10(user_nx * user_ny)

    # EIRP 按高度调整
    sl_max_alt = np.max(sl_radius) - R_EARTH
    kp_max_alt = np.max(kp_radius) - R_EARTH
    sl_eirp = PRIMARY_EIRP * np.array([
        10 ** (-20 * np.log10(sl_max_alt / max(r - R_EARTH, 1)) / 10)
        for r in sl_radius])
    kp_eirp = SECONDARY_EIRP * np.array([
        10 ** (-20 * np.log10(kp_max_alt / max(r - R_EARTH, 1)) / 10)
        for r in kp_radius])

    # 结果容器
    R = {}
    R['inr_abs_max'] = []; R['inr_abs_min'] = []
    R['inr_cond_max'] = []; R['inr_cond_min'] = []
    R['feasible_counts'] = {thr: [] for thr in INR_THRESHOLDS}
    R['pri_sinr_gsnr'] = []; R['pri_sinr_gsinr'] = []
    R['sec_sinr_gsnr'] = []; R['sec_sinr_gsinr'] = []
    R['pri_snr_ub'] = []
    R['inr_gsnr'] = []; R['inr_gsinr'] = []
    R['pri_sinr_psnr'] = {t: [] for t in [-12.2, -6.0, 0.0]}
    R['sec_sinr_psnr'] = {t: [] for t in [-12.2, -6.0, 0.0]}
    R['pri_sinr_psinr'] = {t: [] for t in [-12.2, -6.0, 0.0]}
    R['sec_sinr_psinr'] = {t: [] for t in [-12.2, -6.0, 0.0]}
    R['useful'] = {(d, t): [] for d in [1, 2, 3] for t in [-12.2, -6.0, 0.0]}
    R['ang_sinr'] = {t: [] for t in [-12.2, -6.0, 0.0]}
    R['ang_snr'] = {t: [] for t in [-12.2, -6.0, 0.0]}
    R['el_sec'] = {t: [] for t in [-12.2, -6.0, 0.0]}
    R['ang_pri'] = {t: [] for t in [-12.2, -6.0, 0.0]}
    R['feas_unc'] = {g: {t: [] for t in INR_THRESHOLDS} for g in [0, 10, 20, 30, 40, 50]}
    R['guar_sinr'] = {g: [] for g in [0, 10, 20, 30, 40, 50]}

    # 采样城市 (不同纬度)
    city_ids = [0, 2, 5, 7, 12, 13]  # Austin, Bogota, Chicago, Moscow, Singapore, Sydney
    # 采样时间 (每 3 步取 1, ~240 步)
    t_indices = np.arange(0, n_time_steps, 3)

    if verbose:
        print(f"仿真: {len(t_indices)} 时间步, {len(city_ids)} 城市, "
              f"天线 {user_config}")

    for ti, t_idx in enumerate(t_indices):
        t = t_idx * TIME_RESOLUTION
        if verbose and ti % 30 == 0:
            print(f"  {t/3600:.1f}h/{SIM_DURATION/3600:.0f}h")

        sl_pos = get_satellite_positions(sl_raan, sl_ma0, sl_radius, sl_incl, t)
        kp_pos = get_satellite_positions(kp_raan, kp_ma0, kp_radius, kp_incl, t)

        for ci in city_ids:
            _, lat, lon = CITIES[ci]
            upos = get_user_position(lat, lon)
            east, north, up = get_local_frame(lat, lon)

            sl_el, _, sl_dist, sl_dir = compute_geometry(upos, sl_pos, east, north, up)
            kp_el, _, kp_dist, kp_dir = compute_geometry(upos, kp_pos, east, north, up)

            sl_ov = np.where(sl_el >= MIN_ELEVATION)[0]
            kp_ov = np.where(kp_el >= MIN_ELEVATION)[0]
            if len(sl_ov) == 0 or len(kp_ov) == 0:
                continue

            # 主服务卫星 (最高仰角)
            pi = sl_ov[np.argmax(sl_el[sl_ov])]
            pd = sl_dir[pi]; pdist = sl_dist[pi]; pel = sl_el[pi]

            # ===== 向量化: 所有次级卫星对主用户的 INR =====
            kp_dirs = kp_dir[kp_ov]  # (M, 3)
            kp_dists = kp_dist[kp_ov]
            kp_els = kp_el[kp_ov]

            # 用户接收增益: 波束指向主卫星, 评估各次级卫星方向
            grx_sec = vec_upa_gain(kp_dirs, pd, user_nx, user_ny, east, north)
            gtx_sec = np.full(len(kp_ov), sat_g)  # 共址 → 主瓣最大增益
            pl_sec = total_path_loss_db(kp_dists, kp_els)
            inr_all = compute_inr_vec(SECONDARY_EIRP, gtx_sec, grx_sec, pl_sec)

            # 主用户 SNR
            pl_pri = total_path_loss_db(np.array([pdist]), np.array([pel]))[0]
            pri_snr = compute_snr_db(PRIMARY_EIRP, np.array([sat_g]),
                                     np.array([usr_g]), np.array([pl_pri]))[0]

            # 次级用户 SNR
            snr_sec = compute_snr_db(SECONDARY_EIRP, gtx_sec,
                                     np.full(len(kp_ov), usr_g), pl_sec)

            # 次级用户受到主系统干扰的 INR
            grx_toward_pri = vec_upa_gain(
                np.broadcast_to(pd, kp_dirs.shape), kp_dirs,
                user_nx, user_ny, east, north)
            inr_from_pri = compute_inr_vec(PRIMARY_EIRP,
                                           np.full(len(kp_ov), sat_g),
                                           grx_toward_pri,
                                           np.full(len(kp_ov), pl_pri))
            sinr_sec = compute_sinr_db(snr_sec, inr_from_pri)

            # ===== 收集结果 =====
            R['inr_abs_max'].append(np.max(inr_all))
            R['inr_abs_min'].append(np.min(inr_all))
            R['inr_cond_max'].append(np.max(inr_all))
            R['inr_cond_min'].append(np.min(inr_all))
            R['pri_snr_ub'].append(pri_snr)

            for thr in INR_THRESHOLDS:
                R['feasible_counts'][thr].append(int(np.sum(inr_all <= thr)))

            # 贪心选择
            i_snr = np.argmax(snr_sec)
            i_sinr = np.argmax(sinr_sec)
            R['sec_sinr_gsnr'].append(sinr_sec[i_snr])
            R['inr_gsnr'].append(inr_all[i_snr])
            R['sec_sinr_gsinr'].append(sinr_sec[i_sinr])
            R['inr_gsinr'].append(inr_all[i_sinr])
            R['pri_sinr_gsnr'].append(compute_sinr_db(np.array([pri_snr]),
                                                       np.array([inr_all[i_snr]]))[0])
            R['pri_sinr_gsinr'].append(compute_sinr_db(np.array([pri_snr]),
                                                        np.array([inr_all[i_sinr]]))[0])

            # 保护性选择
            for thr in [-12.2, -6.0, 0.0]:
                mask = inr_all <= thr
                if not np.any(mask):
                    continue
                feas = np.where(mask)[0]

                b1 = feas[np.argmax(snr_sec[feas])]  # Prot. Max-SNR
                b2 = feas[np.argmax(sinr_sec[feas])]  # Prot. Max-SINR

                R['sec_sinr_psnr'][thr].append(sinr_sec[b1])
                R['pri_sinr_psnr'][thr].append(compute_sinr_db(
                    np.array([pri_snr]), np.array([inr_all[b1]]))[0])
                R['sec_sinr_psinr'][thr].append(sinr_sec[b2])
                R['pri_sinr_psinr'][thr].append(compute_sinr_db(
                    np.array([pri_snr]), np.array([inr_all[b2]]))[0])

                # 有用卫星
                for delta in [1, 2, 3]:
                    thr_s = snr_sec[i_snr] - delta
                    R['useful'][(delta, thr)].append(int(np.sum(mask & (sinr_sec >= thr_s))))

                # 角间距
                sd2 = kp_dirs[b2]
                c1 = np.clip(np.dot(sd2, kp_dirs[i_sinr]), -1, 1)
                c2 = np.clip(np.dot(sd2, kp_dirs[i_snr]), -1, 1)
                c3 = np.clip(np.dot(sd2, pd), -1, 1)
                R['ang_sinr'][thr].append(np.degrees(np.arccos(c1)))
                R['ang_snr'][thr].append(np.degrees(np.arccos(c2)))
                R['el_sec'][thr].append(kp_els[b2])
                R['ang_pri'][thr].append(np.degrees(np.arccos(c3)))

            # 不确定性 (每 3 个时间步中再隔 1 个)
            if ti % 2 == 0:
                _run_uncertainty(R, sl_dir, sl_el, sl_dist, sl_ov,
                                 kp_dir, kp_el, kp_dist, kp_ov,
                                 pd, pel, pdist, east, north,
                                 user_nx, user_ny, sat_g, usr_g)

    if verbose:
        print("仿真完成!")
    return R


def _run_uncertainty(R, sl_dir, sl_el, sl_dist, sl_ov,
                     kp_dir, kp_el, kp_dist, kp_ov,
                     pd, pel, pdist, east, north,
                     unx, uny, sat_g, usr_g):
    """不确定性分析 (Fig. 13/14)"""
    pi_idx = np.argmin(np.abs(sl_dir[sl_ov] - pd).sum(axis=1))
    # 更精确: 找到最近的 overhead primary sat index
    dots = sl_dir[sl_ov] @ pd
    pi_local = np.argmax(dots)

    for gamma in [0, 10, 20, 30, 40, 50]:
        gamma_rad = np.deg2rad(gamma)
        ang = np.arccos(np.clip(sl_dir[sl_ov] @ pd, -1, 1))
        p_prime = sl_ov[ang <= gamma_rad]
        if gamma == 0:
            p_prime = sl_ov[pi_local:pi_local+1]

        if len(p_prime) == 0:
            continue

        n_kp = len(kp_ov)
        p_dirs = sl_dir[p_prime]  # (P', 3)

        for thr in INR_THRESHOLDS:
            n_ok = 0
            for ki in kp_ov:
                # 对所有 p' 计算 INR
                sd = kp_dir[ki]
                grx = vec_upa_gain(
                    np.broadcast_to(sd, p_dirs.shape),
                    p_dirs, unx, uny, east, north)
                gtx = np.full(len(p_prime), sat_g)
                pl = total_path_loss_db(sl_dist[p_prime], sl_el[p_prime])
                inr_p = compute_inr_vec(SECONDARY_EIRP, gtx, grx, pl)
                if np.all(inr_p <= thr):
                    n_ok += 1
            R['feas_unc'][gamma][thr].append(n_ok)

        # 保障 SINR (INRth = -12.2 dB)
        best_guar = -100.0
        for ki in kp_ov:
            sd = kp_dir[ki]
            grx = vec_upa_gain(
                np.broadcast_to(sd, p_dirs.shape),
                p_dirs, unx, uny, east, north)
            gtx = np.full(len(p_prime), sat_g)
            pl = total_path_loss_db(sl_dist[p_prime], sl_el[p_prime])
            inr_p = compute_inr_vec(SECONDARY_EIRP, gtx, grx, pl)
            if not np.all(inr_p <= -12.2):
                continue

            # 对每个 p' 计算次级 SINR, 取最小 (保障值)
            min_sinr = 100.0
            for pj in p_prime:
                gtx_p = np.array([sat_g])
                grx_p = vec_upa_gain(
                    sd[np.newaxis, :], sl_dir[pj][np.newaxis, :],
                    unx, uny, east, north)
                pl_p = total_path_loss_db(np.array([kp_dist[ki]]),
                                          np.array([kp_el[ki]]))
                inr_p2 = compute_inr_vec(PRIMARY_EIRP, gtx_p, grx_p, pl_p)
                snr_s = compute_snr_db(SECONDARY_EIRP, np.array([sat_g]),
                                       np.array([usr_g]), pl_p)
                sinr_s = compute_sinr_db(snr_s, inr_p2)
                min_sinr = min(min_sinr, sinr_s[0])
            best_guar = max(best_guar, min_sinr)

        if best_guar > -100:
            R['guar_sinr'][gamma].append(best_guar)
