"""
统计分析模块
实现论文 Section III-E 的INR概率分布和中断概率计算
"""
import numpy as np
from config import (
    RE, BANDWIDTH, C0, PT_W, NOISE_POWER,
    GT_MAX_DBI, GR_MAX_DBI,
)


def compute_inr_pdf_monte_carlo(gs_lat, gs_lon, h_orbit, inclination, n_planes,
                                 n_sats_per_plane, elev_com_deg, int_radius,
                                 max_terminals, n_mc=5000, theta_min=25.0):
    """
    Monte Carlo仿真计算INR概率密度函数 - 用于验证Fig. 9

    对每个Monte Carlo样本:
    1. 随机生成干扰终端分布
    2. 计算聚合INR
    3. 统计INR分布

    参数:
        elev_com_deg: 通信卫星仰角 (度), 可以为25S/90/25N

    返回:
        inr_samples: (n_mc,) INR样本 (dB)
    """
    from constellation import (
        walker_satellite_positions, coarse_filtering, fine_filtering,
        communication_satellite_position, get_interfering_terminals,
    )
    from interference import compute_aggregate_interference

    gs_lat_rad = np.radians(gs_lat)
    gs_lon_rad = np.radians(gs_lon)
    interfered_gs_pos = RE * np.array([
        np.cos(gs_lat_rad) * np.cos(gs_lon_rad),
        np.cos(gs_lat_rad) * np.sin(gs_lon_rad),
        np.sin(gs_lat_rad)
    ])

    # 取一个固定时刻计算 (干扰卫星位置固定)
    t = 0.0
    all_pos, all_lats, all_lons = walker_satellite_positions(
        t, h_orbit, inclination, n_planes, n_sats_per_plane, 1
    )

    # 可见卫星
    coarse_vis = coarse_filtering(gs_lat, gs_lon, all_lats, all_lons, h_orbit, theta_min)
    if len(coarse_vis) > 0:
        coarse_pos = all_pos[coarse_vis]
        fine_vis_local = fine_filtering(gs_lat, gs_lon, coarse_pos, h_orbit, theta_min)
        visible = [coarse_vis[i] for i in fine_vis_local]
    else:
        visible = []

    com_sat_pos = communication_satellite_position(gs_lat, gs_lon, elev_com_deg, h_orbit)

    inr_samples = np.full(n_mc, -30.0)

    for mc in range(n_mc):
        # 随机生成干扰终端
        int_terminals, _, _ = get_interfering_terminals(
            gs_lat, gs_lon, int_radius, max_terminals, seed=mc
        )

        if len(visible) == 0:
            continue

        _, inr_w, _ = compute_aggregate_interference(
            visible, all_pos, int_terminals,
            interfered_gs_pos, com_sat_pos, gs_lat, gs_lon,
            h_orbit, theta_min
        )

        inr_samples[mc] = 10 * np.log10(max(inr_w, 1e-30))

    return inr_samples


def compute_outage_probability(teis_db, gs_lat, gs_lon, h_orbit, elev_array,
                                c0=C0, bandwidth=BANDWIDTH):
    """
    计算中断概率 - Eq. (13), (49)

    PT=38 dBW按EIRP解读, 通信链路不再乘GT_max

    参数:
        teis_db: (T, E) TEIS矩阵 (INR, dB)
        gs_lat, gs_lon: 地面站位置
        h_orbit: 轨道高度
        elev_array: 仰角数组

    返回:
        outage_prob: (T, E) 中断概率矩阵 [0, 1]
    """
    from constellation import communication_satellite_position
    from interference import total_path_loss

    s0_db = 10 * np.log10(max(2 ** (c0 / bandwidth) - 1, 1e-30))

    gs_lat_rad = np.radians(gs_lat)
    gs_lon_rad = np.radians(gs_lon)
    gs_pos = RE * np.array([
        np.cos(gs_lat_rad) * np.cos(gs_lon_rad),
        np.cos(gs_lat_rad) * np.sin(gs_lon_rad),
        np.sin(gs_lat_rad)
    ])

    outage = np.zeros_like(teis_db)
    transition_width = 3.0

    for ei, elev in enumerate(elev_array):
        com_sat_pos = communication_satellite_position(gs_lat, gs_lon, elev, h_orbit)
        d_com = np.linalg.norm(com_sat_pos - gs_pos)

        if elev <= 90:
            actual_elev = elev
        else:
            actual_elev = 180 - elev

        # EIRP解读: 不再乘GT_max
        pl_com = total_path_loss(d_com, max(actual_elev, 1.0))
        eirp_w = PT_W
        gr_com = 10 ** (GR_MAX_DBI / 10)
        pr = eirp_w * gr_com * 10 ** (-pl_com / 10)

        for ti in range(teis_db.shape[0]):
            inr_db = teis_db[ti, ei]
            inr_linear = 10 ** (inr_db / 10)
            sinr_db = 10 * np.log10(max(pr / (inr_linear * NOISE_POWER + NOISE_POWER), 1e-30))
            outage[ti, ei] = 1.0 / (1.0 + np.exp((sinr_db - s0_db) / transition_width))

    return outage


def compute_throughput(teis_db, gs_lat, gs_lon, h_orbit, elev_array,
                       bandwidth=BANDWIDTH, c0=C0):
    """
    计算吞吐量 - Eq. (14)

    THU = B * (1 - Poutage) * log2(1 + SINR)

    参数:
        teis_db: (T, E) TEIS矩阵 (INR, dB)

    返回:
        throughput: (T, E) 吞吐量矩阵 (bps)
    """
    from constellation import communication_satellite_position
    from interference import total_path_loss

    gs_lat_rad = np.radians(gs_lat)
    gs_lon_rad = np.radians(gs_lon)
    gs_pos = RE * np.array([
        np.cos(gs_lat_rad) * np.cos(gs_lon_rad),
        np.cos(gs_lat_rad) * np.sin(gs_lon_rad),
        np.sin(gs_lat_rad)
    ])

    throughput = np.zeros_like(teis_db)

    for ei, elev in enumerate(elev_array):
        com_sat_pos = communication_satellite_position(gs_lat, gs_lon, elev, h_orbit)
        d_com = np.linalg.norm(com_sat_pos - gs_pos)

        if elev <= 90:
            actual_elev = elev
        else:
            actual_elev = 180 - elev

        pl_com = total_path_loss(d_com, max(actual_elev, 1.0))
        gt_com = 10 ** (GT_MAX_DBI / 10)
        gr_com = 10 ** (GR_MAX_DBI / 10)
        pr = PT_W * gt_com * gr_com * 10 ** (-pl_com / 10)

        for ti in range(teis_db.shape[0]):
            inr_w = 10 ** (teis_db[ti, ei] / 10)
            interference = inr_w * NOISE_POWER
            sinr = pr / (interference + NOISE_POWER)

            if sinr > 0:
                throughput[ti, ei] = bandwidth * np.log2(1 + sinr)
            else:
                throughput[ti, ei] = 0.0

    return throughput
