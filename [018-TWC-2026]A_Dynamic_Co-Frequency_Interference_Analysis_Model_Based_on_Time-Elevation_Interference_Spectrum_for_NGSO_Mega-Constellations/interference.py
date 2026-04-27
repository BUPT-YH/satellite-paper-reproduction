"""
干扰计算模块
实现论文中的路径损耗、天线增益、TEIS和INR计算
"""
import numpy as np
from config import (
    RE, K_BOLTZMANN, FREQ_MHZ, PT_W, BANDWIDTH, T_NOISE,
    GT_MAX_DBI, GR_MAX_DBI, D_LAMBDA_TX, D_LAMBDA_RX,
    LS_TX, LF_TX, PHI_M_RX,
    GAMMA_R, RAIN_HEIGHT, PL_ZENITH,
    NOISE_POWER,
)


def free_space_path_loss(d_km):
    """自由空间路径损耗 (dB) - Eq. (2)"""
    return 32.45 + 20 * np.log10(d_km) + 20 * np.log10(FREQ_MHZ)


def rain_attenuation(elev_deg):
    """雨衰 (dB) - Eq. (3)"""
    elev_rad = np.radians(max(elev_deg, 1.0))  # 防止除零
    LE = RAIN_HEIGHT / np.sin(elev_rad)
    return GAMMA_R * LE


def atmospheric_attenuation(elev_deg):
    """大气吸收衰减 (dB) - Eq. (4, 5)"""
    elev_rad = np.radians(max(elev_deg, 1.0))
    return PL_ZENITH / np.sin(elev_rad)


def total_path_loss(d_km, elev_deg):
    """总路径损耗 (dB) - Eq. (1)"""
    fspl = free_space_path_loss(d_km)
    plr = rain_attenuation(elev_deg)
    pla = atmospheric_attenuation(elev_deg)
    return fspl + plr + pla


def tx_antenna_gain(off_axis_deg):
    """
    发射天线增益 (dBi) - ITU-R S.1528, Eq. (6)
    off_axis_deg: 偏轴角 (度)
    """
    gt_max = GT_MAX_DBI
    phi_b = 3.0 / D_LAMBDA_TX * 180.0 / np.pi  # 3dB波束宽度
    if phi_b < 1.0:
        phi_b = 1.0

    Y = 1.5 * phi_b
    Z = Y * 10 ** (0.04 * (gt_max + LS_TX - LF_TX))

    phi = abs(off_axis_deg)

    if phi <= phi_b:
        return gt_max
    elif phi < Y:
        return gt_max - 3.0 * (phi / phi_b) ** 2
    elif phi < Z:
        val = gt_max + LS_TX - 25 * np.log10(phi / Y)
        return max(val, LF_TX)
    else:
        return LF_TX


def rx_antenna_gain(off_axis_deg):
    """
    接收天线增益 (dBi) - ITU-R S.465-5, Eq. (7, 8)
    off_axis_deg: 偏轴角 (度)
    """
    phi = abs(off_axis_deg)

    if phi < PHI_M_RX:
        return GR_MAX_DBI
    elif phi <= 48.0:
        return 32.0 - 25 * np.log10(max(phi, 0.1))
    else:
        return -10.0


def compute_off_axis_angle_tx(sat_pos, target_gs_pos, interfered_gs_pos):
    """
    计算发射端偏轴角 (度)

    干扰卫星的波束指向其服务的干扰终端,
    偏轴角为干扰卫星→被干扰终端方向与主瓣方向的夹角
    """
    vec_main = target_gs_pos - sat_pos  # 主瓣方向: 指向关联终端
    vec_side = interfered_gs_pos - sat_pos  # 干扰方向: 指向被干扰终端

    cos_angle = np.dot(vec_main, vec_side) / (np.linalg.norm(vec_main) * np.linalg.norm(vec_side) + 1e-30)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def compute_off_axis_angle_rx(gs_pos, com_sat_pos, int_sat_pos):
    """
    计算接收端偏轴角 (度) - Eq. (40)

    被干扰终端的主瓣指向通信卫星,
    偏轴角为被干扰终端→干扰卫星方向与主瓣方向的夹角
    """
    vec_main = com_sat_pos - gs_pos  # 主瓣方向: 指向通信卫星
    vec_side = int_sat_pos - gs_pos  # 干扰方向: 指向干扰卫星

    cos_angle = np.dot(vec_main, vec_side) / (np.linalg.norm(vec_main) * np.linalg.norm(vec_side) + 1e-30)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def compute_slant_range(sat_pos, gs_pos):
    """计算星地距离 (km)"""
    return np.linalg.norm(sat_pos - gs_pos)


def compute_single_inr(sat_pos, target_gs_pos, interfered_gs_pos, com_sat_pos, elev_deg):
    """
    计算单个干扰卫星的INR - Eq. (9)

    参数:
        sat_pos: 干扰卫星ECEF位置
        target_gs_pos: 干扰终端ECEF位置 (卫星关联的终端)
        interfered_gs_pos: 被干扰终端ECEF位置
        com_sat_pos: 通信卫星ECEF位置
        elev_deg: 干扰链路的仰角 (用于路径损耗计算)

    返回:
        inr_db: INR (dB)
        inr_w: INR (线性值, 瓦特/瓦特噪声比)
    """
    # 路径损耗
    d = compute_slant_range(sat_pos, interfered_gs_pos)
    pl_db = total_path_loss(d, elev_deg)
    gh = 10 ** (-pl_db / 10)  # 信道增益 (线性)

    # 发射天线增益
    phi_t = compute_off_axis_angle_tx(sat_pos, target_gs_pos, interfered_gs_pos)
    gt_dbi = tx_antenna_gain(phi_t)
    gt = 10 ** (gt_dbi / 10)

    # 接收天线增益
    phi_r = compute_off_axis_angle_rx(interfered_gs_pos, com_sat_pos, sat_pos)
    gr_dbi = rx_antenna_gain(phi_r)
    gr = 10 ** (gr_dbi / 10)

    # INR (线性)
    inr_w = PT_W * gt * gr * gh / NOISE_POWER

    # INR (dB)
    inr_db = 10 * np.log10(max(inr_w, 1e-30))

    return inr_db, inr_w


def compute_aggregate_interference(visible_indices, all_positions, int_terminals,
                                   interfered_gs_pos, com_sat_pos, gs_lat, gs_lon,
                                   h_orbit, theta_min):
    """
    计算聚合干扰 INR

    遍历所有可见干扰卫星, 对每颗卫星找到其关联的干扰终端,
    计算该卫星对被干扰终端的干扰, 然后求和

    返回:
        total_inr_db: 聚合INR (dB)
        total_inr_w: 聚合INR (线性)
        interference_details: 各干扰源详情
    """
    total_inr_w = 0.0
    interference_details = []

    for idx in visible_indices:
        sat_pos = all_positions[idx]

        # 找到与该卫星关联的干扰终端 (最大仰角准则)
        for term_pos in int_terminals:
            diff = sat_pos - term_pos
            dist = np.linalg.norm(diff)
            r_term = np.linalg.norm(term_pos)
            elev = np.degrees(np.arcsin(np.clip(
                np.dot(diff, term_pos / r_term) / dist, -1, 1
            )))

            if elev >= theta_min:
                # 计算该干扰卫星对被干扰终端的INR
                inr_db, inr_w = compute_single_inr(
                    sat_pos, term_pos, interfered_gs_pos, com_sat_pos, elev
                )
                total_inr_w += inr_w
                interference_details.append({
                    'sat_idx': idx,
                    'elev': elev,
                    'inr_db': inr_db,
                    'inr_w': inr_w
                })
                break  # 每颗卫星只与一个终端关联 (最大仰角)

    total_inr_db = 10 * np.log10(max(total_inr_w, 1e-30))
    return total_inr_db, total_inr_w, interference_details


def compute_teis(gs_lat, gs_lon, h_orbit, inclination, n_planes, n_sats_per_plane,
                 t_array, elev_array, int_terminals, theta_min=25.0):
    """
    计算时-仰角干扰谱 (TEIS) - Section III-D

    参数:
        gs_lat, gs_lon: 被干扰终端经纬度
        h_orbit: 轨道高度
        inclination: 轨道倾角
        n_planes: 轨道面数
        n_sats_per_plane: 每面卫星数
        t_array: 时间数组 (s)
        elev_array: 通信卫星仰角数组 (度, 论文定义: 0=南地平线, 90=天顶, 180=北地平线)
        int_terminals: 干扰终端ECEF坐标 (N, 3)
        theta_min: 最小仰角

    返回:
        teis: (len(t_array), len(elev_array)) INR矩阵 (dB)
    """
    from constellation import (
        get_visible_satellites, communication_satellite_position
    )

    teis = np.full((len(t_array), len(elev_array)), -30.0)  # 初始化为极低INR
    gs_lat_rad = np.radians(gs_lat)
    gs_lon_rad = np.radians(gs_lon)

    # 被干扰终端ECEF坐标
    interfered_gs_pos = RE * np.array([
        np.cos(gs_lat_rad) * np.cos(gs_lon_rad),
        np.cos(gs_lat_rad) * np.sin(gs_lon_rad),
        np.sin(gs_lat_rad)
    ])

    for ti, t in enumerate(t_array):
        # 获取可见干扰卫星
        visible, all_pos, _, _ = get_visible_satellites(
            t, gs_lat, gs_lon, h_orbit, inclination, n_planes, n_sats_per_plane
        )

        if len(visible) == 0:
            continue

        for ei, elev_com in enumerate(elev_array):
            # 通信卫星位置
            com_sat_pos = communication_satellite_position(
                gs_lat, gs_lon, elev_com, h_orbit
            )

            # 计算聚合干扰
            inr_db, inr_w, _ = compute_aggregate_interference(
                visible, all_pos, int_terminals,
                interfered_gs_pos, com_sat_pos, gs_lat, gs_lon,
                h_orbit, theta_min
            )

            teis[ti, ei] = inr_db

    return teis
