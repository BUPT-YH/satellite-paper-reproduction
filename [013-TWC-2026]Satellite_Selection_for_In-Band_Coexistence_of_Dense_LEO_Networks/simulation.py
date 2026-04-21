"""
仿真主逻辑模块 — 精确 FCC 参数版
  Fig. 4 — 基线 INR CDF (无保护)
  Fig. 5 — 提出方案 INR CDF (不同 INRmax_th)
"""

import numpy as np
from config import *
from satellite_selection import (
    propagate_constellation, get_ground_cluster_centers,
    compute_elevation, compute_single_beam_inr, compute_snr_db,
    sat_tx_gain_dbi, user_rx_gain_dbi, compute_off_axis_angle_deg,
    compute_free_space_path_loss_db, compute_noise_dbw, compute_eirp_dbw,
    TX_MAX_GAIN_DBI
)

BW_HZ = BANDWIDTH_MHZ * 1e6  # 带宽 (Hz)
SEC_EIRP_DBW = compute_eirp_dbw(SECONDARY_MAX_EIRP_DBW_PER_HZ, BW_HZ)  # 次系统总 EIRP
PRI_EIRP_DBW = compute_eirp_dbw(PRIMARY_MAX_EIRP_DBW_PER_HZ, BW_HZ)


def get_visible_sats(sat_positions, ground_pos, min_elev_deg=MIN_ELEVATION_DEG):
    """获取对地面点可见的卫星索引和仰角"""
    n = sat_positions.shape[0]
    el_arr = np.array([compute_elevation(sat_positions[i], ground_pos) for i in range(n)])
    mask = el_arr >= min_elev_deg
    return np.where(mask)[0], el_arr[mask]


def he_association(sat_positions, cluster_centers):
    """HE 策略: 每个簇选仰角最高的卫星"""
    nc = cluster_centers.shape[0]
    assoc = -np.ones(nc, dtype=int)
    used = set()
    for n in range(nc):
        vis, elevs = get_visible_sats(sat_positions, cluster_centers[n])
        avail = [(i, e) for i, e in zip(vis, elevs) if i not in used]
        if avail:
            avail.sort(key=lambda x: -x[1])
            assoc[n] = avail[0][0]
            used.add(avail[0][0])
    return assoc


def mct_association(sat_positions, cluster_centers):
    """MCT 策略: 选仰角最低的可见卫星 (刚进入覆盖, 剩余可见时间最长)"""
    nc = cluster_centers.shape[0]
    assoc = -np.ones(nc, dtype=int)
    used = set()
    for n in range(nc):
        vis, elevs = get_visible_sats(sat_positions, cluster_centers[n])
        avail = [(i, e) for i, e in zip(vis, elevs) if i not in used]
        if avail:
            avail.sort(key=lambda x: x[1])
            assoc[n] = avail[0][0]
            used.add(avail[0][0])
    return assoc


def compute_aggregate_inr(pri_pos, sec_pos, pri_assoc, sec_assoc,
                          cluster_centers, num_beams, user_positions):
    """
    计算所有主用户的聚合 INR (dB)
    每波束 EIRP = 总 EIRP (论文模型), NB 个波束叠加
    """
    all_inr = []
    for n in range(len(cluster_centers)):
        if pri_assoc[n] < 0:
            continue
        pri_sat = pri_pos[pri_assoc[n]]
        users = user_positions[n]

        for user_pos in users:
            agg = 0.0
            for m in range(len(cluster_centers)):
                if sec_assoc[m] < 0:
                    continue
                sec_sat = sec_pos[sec_assoc[m]]
                sec_cluster = cluster_centers[m]

                dist = np.linalg.norm(sec_sat - user_pos)
                if dist < 1:
                    continue
                fspl = compute_free_space_path_loss_db(dist)

                # 发射偏轴角
                off_tx = compute_off_axis_angle_deg(sec_sat, sec_cluster, user_pos)
                tx_g = sat_tx_gain_dbi(off_tx)
                eff_eirp = SEC_EIRP_DBW + (tx_g - TX_MAX_GAIN_DBI)

                # 接收偏轴角
                off_rx = compute_off_axis_angle_deg(user_pos, pri_sat, sec_sat)
                rx_g = user_rx_gain_dbi(off_rx)

                noise = compute_noise_dbw(BW_HZ)
                inr_1 = eff_eirp - fspl + rx_g - noise
                # NB 个波束叠加 (dB → linear sum → dB)
                inr_nb = inr_1 + 10 * np.log10(max(num_beams, 1))
                inr_nb = min(inr_nb, 40)

                if inr_nb > -999:
                    agg += 10 ** (inr_nb / 10)

            all_inr.append(10 * np.log10(agg) if agg > 0 else -999.0)
    return np.array(all_inr)


def setup_simulation():
    """初始化: 地面簇 + 用户"""
    cluster_centers, lats, lons = get_ground_cluster_centers(
        CENTER_LAT, CENTER_LON, NUM_CLUSTERS)
    np.random.seed(42)
    users_per_cluster = 20
    user_positions = {}
    for n in range(NUM_CLUSTERS):
        offsets = np.random.randn(users_per_cluster, 3) * CELL_RADIUS_KM * 0.8
        user_positions[n] = cluster_centers[n] + offsets
        for i in range(users_per_cluster):
            norm = np.linalg.norm(user_positions[n][i])
            user_positions[n][i] = user_positions[n][i] / norm * EARTH_RADIUS
    return cluster_centers, user_positions


# ===== Fig. 4: 基线 =====
def run_baseline_simulation(duration_sec=30, time_step=2.0):
    print("运行基线仿真 (无保护, 精确FCC参数)...")
    cluster_centers, user_positions = setup_simulation()
    time_steps = np.arange(0, duration_sec, time_step)
    results = {}

    for num_beams in BEAM_CONFIGS:
        for sec_policy in ['HE', 'MCT']:
            key = f"NB{num_beams}_PHE_S{sec_policy}"
            all_inr = []

            for t in time_steps:
                pri_pos = propagate_constellation(STARLINK_SHELLS, t)
                sec_pos = propagate_constellation(KUIPER_SHELLS, t)

                pri_assoc = he_association(pri_pos, cluster_centers)
                sec_assoc = (he_association if sec_policy == 'HE' else mct_association)(
                    sec_pos, cluster_centers)

                inr_vals = compute_aggregate_inr(
                    pri_pos, sec_pos, pri_assoc, sec_assoc,
                    cluster_centers, num_beams, user_positions)
                valid = inr_vals[inr_vals > -900]
                if len(valid) > 0:
                    all_inr.extend(valid.tolist())

            results[key] = np.array(all_inr)
            print(f"  {key}: {len(all_inr)} samples, "
                  f"median={np.median(all_inr):.1f}, "
                  f"@30%={np.percentile(all_inr, 30):.1f} dB")
    return results


# ===== Fig. 5/10: 提出方案 =====
def secondary_selection(pri_pos, sec_pos, pri_assoc, cluster_centers,
                        num_beams, user_positions, inr_th_db, inr_max_th_db):
    """
    次系统卫星选择: 拉格朗日松弛 + 绝对约束硬排除
    """
    nc = len(cluster_centers)

    # 每簇可用卫星
    available = []
    for n in range(nc):
        vis, _ = get_visible_sats(sec_pos, cluster_centers[n])
        available.append(vis)

    ns = sec_pos.shape[0]
    cap = np.full((ns, nc), -999.0)
    intf = np.full((ns, nc), -999.0)

    for n in range(nc):
        for m in available[n]:
            ss = sec_pos[m]
            # 容量 (SNR)
            elev = compute_elevation(ss, cluster_centers[n])
            snr = compute_snr_db(ss, cluster_centers[n], SEC_EIRP_DBW, BW_HZ)
            cap[m, n] = max(snr, -10)

            # 干扰: 该卫星对所有主用户的最大 INR
            max_inr = -999.0
            for pn in range(nc):
                if pri_assoc[pn] < 0:
                    continue
                ps = pri_pos[pri_assoc[pn]]
                for up in user_positions[pn][:5]:
                    d = np.linalg.norm(ss - up)
                    if d < 1:
                        continue
                    fspl = compute_free_space_path_loss_db(d)
                    otx = compute_off_axis_angle_deg(ss, cluster_centers[n], up)
                    txg = sat_tx_gain_dbi(otx)
                    eff = SEC_EIRP_DBW + (txg - TX_MAX_GAIN_DBI)
                    orx = compute_off_axis_angle_deg(up, ps, ss)
                    rxg = user_rx_gain_dbi(orx)
                    noise = compute_noise_dbw(BW_HZ)
                    i1 = eff - fspl + rxg - noise
                    inb = i1 + 10 * np.log10(max(num_beams, 1))
                    max_inr = max(max_inr, min(inb, 40))
            intf[m, n] = max_inr

    # 拉格朗日松弛
    lam = 1.0
    best_assoc = -np.ones(nc, dtype=int)

    for k in range(MAX_SUBGRADIENT_ITERS):
        used = set()
        assoc = -np.ones(nc, dtype=int)
        for n in range(nc):
            best_s, best_m = -np.inf, -1
            for m in available[n]:
                if m in used:
                    continue
                c, iv = cap[m, n], intf[m, n]
                if c < -900 or iv < -900:
                    continue
                # 绝对约束: 硬排除
                if inr_max_th_db < 1e6 and iv > inr_max_th_db:
                    continue
                score = c - lam * max(0, iv - inr_th_db)
                if score > best_s:
                    best_s, best_m = score, m
            if best_m >= 0:
                assoc[n] = best_m
                used.add(best_m)

        max_int = max((intf[assoc[n], n] for n in range(nc) if assoc[n] >= 0), default=-999)
        step = STEP_SIZE_INIT / (1 + k * 0.05)
        lam = max(0, lam + step * (inr_th_db - max_int))
        best_assoc = assoc.copy()

    return best_assoc


def run_proposed_simulation(duration_sec=30, time_step=2.0,
                            inr_th_db=-6.0, inr_max_th_list=None):
    if inr_max_th_list is None:
        inr_max_th_list = [-6.0, 0.0, 3.0, float('inf')]
    print(f"运行提出方案仿真 (INR_th={inr_th_db} dB)...")

    cluster_centers, user_positions = setup_simulation()
    num_beams = 16
    time_steps = np.arange(0, duration_sec, time_step)
    results = {}

    for inr_max_th in inr_max_th_list:
        label = f"{inr_max_th:.0f}" if inr_max_th < 1e6 else "inf"
        key = f"INRmax_{label}"
        all_inr, util_list = [], []

        for t in time_steps:
            pri_pos = propagate_constellation(STARLINK_SHELLS, t)
            sec_pos = propagate_constellation(KUIPER_SHELLS, t)
            pri_assoc = he_association(pri_pos, cluster_centers)
            sec_assoc = secondary_selection(
                pri_pos, sec_pos, pri_assoc, cluster_centers,
                num_beams, user_positions, inr_th_db, inr_max_th)

            inr_vals = compute_aggregate_inr(
                pri_pos, sec_pos, pri_assoc, sec_assoc,
                cluster_centers, num_beams, user_positions)
            valid = inr_vals[inr_vals > -900]
            if len(valid) > 0:
                all_inr.extend(valid.tolist())
            util_list.append(np.sum(sec_assoc >= 0) / NUM_CLUSTERS)

        results[key] = {'inr': np.array(all_inr), 'utilization': np.array(util_list),
                        'inr_th': inr_th_db, 'inr_max_th': inr_max_th}
        valid = [x for x in all_inr if x > -900]
        print(f"  {key}: {len(valid)} samples, "
              f"median={np.median(valid):.1f} dB, util={np.mean(util_list):.1%}" if valid else f"  {key}: no data")
    return results
