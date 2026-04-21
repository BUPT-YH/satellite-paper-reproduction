"""
卫星星座建模与信道计算模块 — 精确 FCC 参数版
基于 Starlink (6壳层 6900颗) 和 Kuiper (3壳层 3236颗) 的 FCC 公开文件参数

关键改进:
  - 每个壳层独立的 (高度, 倾角, 轨道面数, 每面卫星数)
  - 使用 dBW/Hz 的 EIRP 密度 + 带宽计算总 EIRP
  - 精确的 3dB 波束宽度 (1.6° 卫星 / 3.2° 用户)
  - 接收最大增益 30 dBi (论文 Table III)
"""

import numpy as np
from config import (
    EARTH_RADIUS, C, BOLTZMANN, GM,
    STARLINK_SHELLS, KUIPER_SHELLS,
    PRIMARY_MAX_EIRP_DBW_PER_HZ, SECONDARY_MAX_EIRP_DBW_PER_HZ,
    NOISE_PSD_DBM_PER_HZ, NOISE_FIGURE_DB,
    TX_ANTENNA_ARRAY, TX_BEAMWIDTH_3DB_DEG, TX_MAX_GAIN_DBI,
    RX_ANTENNA_ARRAY, RX_BEAMWIDTH_3DB_DEG, RX_MAX_GAIN_DBI,
    CARRIER_FREQ_GHZ, BANDWIDTH_MHZ, MIN_ELEVATION_DEG
)


def generate_constellation_from_shells(shells):
    """
    按精确 FCC 壳层参数生成星座

    每个壳层: (高度km, 倾角°, 轨道面数, 每面卫星数)
    RAAN 间隔 = 360° / num_planes
    同面内卫星均匀分布

    返回:
        positions: (N, 3) 地固坐标 (km)
    """
    all_positions = []

    for shell in shells:
        alt = shell['altitude_km']
        inc = np.radians(shell['inclination_deg'])
        num_planes = shell['num_planes']
        sats_per_plane = shell['sats_per_plane']

        r = EARTH_RADIUS + alt
        period = 2 * np.pi * np.sqrt(r ** 3 / GM)  # 轨道周期 (s)

        for plane_idx in range(num_planes):
            raan = 2 * np.pi * plane_idx / num_planes

            for sat_idx in range(sats_per_plane):
                theta0 = 2 * np.pi * sat_idx / sats_per_plane
                # t=0 时刻位置
                nu = theta0
                x_orb = r * np.cos(nu)
                y_orb = r * np.sin(nu)

                # 轨道面 → ECI (不考虑地球自转, t=0)
                cos_raan = np.cos(raan)
                sin_raan = np.sin(raan)
                cos_inc = np.cos(inc)
                sin_inc = np.sin(inc)

                x = cos_raan * x_orb - sin_raan * cos_inc * y_orb
                y = sin_raan * x_orb + cos_raan * cos_inc * y_orb
                z = sin_inc * y_orb

                all_positions.append([x, y, z])

    return np.array(all_positions)


def propagate_constellation(shells, t_sec):
    """
    传播星座到 t_sec 时刻 (考虑地球自转)

    返回:
        positions: (N, 3) 地固坐标 (km)
    """
    all_positions = []
    omega_earth = 2 * np.pi / 86400.0

    for shell in shells:
        alt = shell['altitude_km']
        inc = np.radians(shell['inclination_deg'])
        num_planes = shell['num_planes']
        sats_per_plane = shell['sats_per_plane']

        r = EARTH_RADIUS + alt
        period = 2 * np.pi * np.sqrt(r ** 3 / GM)

        for plane_idx in range(num_planes):
            raan = 2 * np.pi * plane_idx / num_planes
            # 地球自转使 RAAN 有效减小
            raan_eff = raan - omega_earth * t_sec

            cos_raan = np.cos(raan_eff)
            sin_raan = np.sin(raan_eff)
            cos_inc = np.cos(inc)
            sin_inc = np.sin(inc)

            for sat_idx in range(sats_per_plane):
                theta0 = 2 * np.pi * sat_idx / sats_per_plane
                nu = theta0 + 2 * np.pi * t_sec / period

                x_orb = r * np.cos(nu)
                y_orb = r * np.sin(nu)

                x = cos_raan * x_orb - sin_raan * cos_inc * y_orb
                y = sin_raan * x_orb + cos_raan * cos_inc * y_orb
                z = sin_inc * y_orb

                all_positions.append([x, y, z])

    return np.array(all_positions)


def get_ground_cluster_centers(center_lat, center_lon, num_clusters, spacing_km=80):
    """生成地面簇中心的地固坐标"""
    lats, lons = [], []
    cols = int(np.ceil(np.sqrt(num_clusters)))
    for i in range(num_clusters):
        row = i // cols
        col = i % cols
        lat = center_lat + (row - (cols - 1) / 2) * spacing_km / 111.0
        offset = 0.5 * spacing_km / 111.0 if row % 2 else 0
        lon = center_lon + (col - (cols - 1) / 2) * spacing_km / 111.0 + offset
        lats.append(lat)
        lons.append(lon)

    centers = np.zeros((num_clusters, 3))
    for i in range(num_clusters):
        lat_r, lon_r = np.radians(lats[i]), np.radians(lons[i])
        centers[i] = [
            EARTH_RADIUS * np.cos(lat_r) * np.cos(lon_r),
            EARTH_RADIUS * np.cos(lat_r) * np.sin(lon_r),
            EARTH_RADIUS * np.sin(lat_r),
        ]
    return centers, lats, lons


def compute_elevation(sat_pos, ground_pos):
    """仰角 (度)"""
    diff = sat_pos - ground_pos
    dist = np.linalg.norm(diff)
    if dist < 1:
        return -90.0
    ground_norm = ground_pos / np.linalg.norm(ground_pos)
    sin_el = np.dot(diff, ground_norm) / dist
    return np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))


def compute_free_space_path_loss_db(dist_km, freq_ghz=CARRIER_FREQ_GHZ):
    """自由空间路径损耗 (dB)"""
    dist_m = dist_km * 1e3
    freq_hz = freq_ghz * 1e9
    if dist_m <= 0:
        return 200.0
    return 20 * np.log10(4 * np.pi * dist_m * freq_hz / C)


def compute_off_axis_angle_deg(from_pos, to_ref, to_target):
    """偏轴角 (度): from→ref 方向 与 from→target 方向的夹角"""
    ref = to_ref - from_pos
    tgt = to_target - from_pos
    rn, tn = np.linalg.norm(ref), np.linalg.norm(tgt)
    if rn < 1 or tn < 1:
        return 0.0
    cos_a = np.dot(ref, tgt) / (rn * tn)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def antenna_gain_dbi(off_axis_deg, beamwidth_3dB, max_gain):
    """
    通用天线方向图模型

    主瓣: 高斯近似, 3dB 点在 beamwidth_3dB
    旁瓣: max_gain - 25 dB floor
    """
    theta = np.abs(off_axis_deg)

    if isinstance(theta, np.ndarray):
        main = max_gain - 12 * (theta / beamwidth_3dB) ** 2
        side = max_gain - 25
        gain = np.where(theta < 2 * beamwidth_3dB, main, side)
        return np.maximum(gain, max_gain - 30)
    else:
        if theta < 2 * beamwidth_3dB:
            return max(max_gain - 12 * (theta / beamwidth_3dB) ** 2, max_gain - 30)
        else:
            return max(max_gain - 25, max_gain - 30)


def sat_tx_gain_dbi(off_axis_deg):
    """卫星发射增益 (64×64, 1.6° beamwidth, 36 dBi max)"""
    return antenna_gain_dbi(off_axis_deg, TX_BEAMWIDTH_3DB_DEG, TX_MAX_GAIN_DBI)


def user_rx_gain_dbi(off_axis_deg):
    """地面用户接收增益 (32×32, 3.2° beamwidth, 30 dBi max)"""
    return antenna_gain_dbi(off_axis_deg, RX_BEAMWIDTH_3DB_DEG, RX_MAX_GAIN_DBI)


def compute_noise_dbw(bandwidth_hz):
    """接收噪声功率 (dBW)"""
    # 噪声 PSD = -174 dBm/Hz = -204 dBW/Hz
    noise_psd_dbw = NOISE_PSD_DBM_PER_HZ - 30  # → -204 dBW/Hz
    noise_dbw = noise_psd_dbw + 10 * np.log10(bandwidth_hz) + NOISE_FIGURE_DB
    return noise_dbw


def compute_eirp_dbw(eirp_density_dbw_hz, bandwidth_hz):
    """从 EIRP 密度 (dBW/Hz) 和带宽计算总 EIRP (dBW)"""
    return eirp_density_dbw_hz + 10 * np.log10(bandwidth_hz)


def compute_single_beam_inr(sec_sat, sec_cluster, user_pos, pri_sat,
                            eirp_dbw, bandwidth_hz):
    """
    计算单个次系统波束对主用户的 INR (dB)

    INR = EIRP_beam + (Gtx_offaxis - Gtx_max) - FSPL + Grx_offaxis - Noise
    """
    dist = np.linalg.norm(sec_sat - user_pos)
    if dist < 1:
        return -999.0

    fspl = compute_free_space_path_loss_db(dist)

    # 发射偏轴角: 卫星→服务簇 方向 vs 卫星→主用户 方向
    off_tx = compute_off_axis_angle_deg(sec_sat, sec_cluster, user_pos)
    tx_g = sat_tx_gain_dbi(off_tx)
    effective_eirp = eirp_dbw + (tx_g - TX_MAX_GAIN_DBI)

    # 接收偏轴角: 用户→主服务卫星 vs 用户→干扰卫星
    off_rx = compute_off_axis_angle_deg(user_pos, pri_sat, sec_sat)
    rx_g = user_rx_gain_dbi(off_rx)

    noise = compute_noise_dbw(bandwidth_hz)
    inr = effective_eirp - fspl + rx_g - noise
    return inr


def compute_snr_db(sat_pos, user_pos, eirp_dbw, bandwidth_hz):
    """计算 SNR (dB)"""
    dist = np.linalg.norm(sat_pos - user_pos)
    if dist < 1:
        return -999.0
    fspl = compute_free_space_path_loss_db(dist)
    rx_g = user_rx_gain_dbi(0)
    noise = compute_noise_dbw(bandwidth_hz)
    return eirp_dbw - fspl + rx_g - noise
