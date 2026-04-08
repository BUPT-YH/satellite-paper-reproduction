"""
Walker-Delta 星座生成与轨道传播
使用开普勒轨道力学计算卫星位置
"""

import numpy as np
from config import (R_EARTH, MU_EARTH, OMEGA_EARTH,
                    STARLINK_SHELLS, KUIPER_SHELLS)


def generate_walker_delta(n_planes, sats_per_plane, altitude_km,
                          inclination_deg, phasing_F, raan_offset=0.0):
    """
    生成 Walker-Delta 星座的初始轨道参数

    参数:
        n_planes: 轨道面数
        sats_per_plane: 每面卫星数
        altitude_km: 轨道高度 (km)
        inclination_deg: 轨道倾角 (度)
        phasing_F: Walker 相位因子 (0 到 P-1)
        raan_offset: RAAN 偏移量 (弧度)

    返回:
        raan_arr: 每颗卫星的 RAAN (rad), shape (n_total,)
        mean_anomaly_0: 每颗卫星的初始平近点角 (rad), shape (n_total,)
        radius: 轨道半径 (km)
        incl: 倾角 (rad)
    """
    n_total = n_planes * sats_per_plane
    incl = np.deg2rad(inclination_deg)
    radius = R_EARTH + altitude_km

    raan_arr = np.zeros(n_total)
    mean_anomaly_0 = np.zeros(n_total)

    for j in range(n_planes):
        # Walker-Delta: 各轨道面的 RAAN 均匀分布
        raan_j = 2 * np.pi * j / n_planes + raan_offset
        for k in range(sats_per_plane):
            idx = j * sats_per_plane + k
            raan_arr[idx] = raan_j
            # 平近点角: 均匀分布 + Walker 相位
            mean_anomaly_0[idx] = (2 * np.pi * k / sats_per_plane
                                   + 2 * np.pi * phasing_F * j / n_total)

    return raan_arr, mean_anomaly_0, radius, incl


def propagate_orbit(raan, mean_anomaly_0, radius, incl, t):
    """
    传播圆形轨道，返回 ECI 和 ECEF 坐标

    参数:
        raan: 升交点赤经数组 (rad)
        mean_anomaly_0: 初始平近点角数组 (rad)
        radius: 轨道半径 (km)
        incl: 倾角 (rad)
        t: 时间 (秒), 标量

    返回:
        pos_ecef: ECEF 坐标 (km), shape (n_sats, 3)
    """
    n = np.sqrt(MU_EARTH / radius ** 3)  # 平均角速度 (rad/s)

    # 当前真近点角 (圆形轨道，真近点角 = 平近点角)
    nu = mean_anomaly_0 + n * t

    # ECI 坐标
    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_i = np.cos(incl)
    sin_i = np.sin(incl)

    x_eci = radius * (cos_raan * cos_nu - sin_raan * sin_nu * cos_i)
    y_eci = radius * (sin_raan * cos_nu + cos_raan * sin_nu * cos_i)
    z_eci = radius * sin_nu * sin_i

    # 转换到 ECEF (考虑地球自转)
    theta_e = OMEGA_EARTH * t
    cos_te = np.cos(theta_e)
    sin_te = np.sin(theta_e)

    x_ecef = x_eci * cos_te + y_eci * sin_te
    y_ecef = -x_eci * sin_te + y_eci * cos_te
    z_ecef = z_eci

    return np.column_stack([x_ecef, y_ecef, z_ecef])


def build_constellation(shells):
    """
    构建完整星座的轨道参数

    返回:
        raan_all, ma0_all, radius_all, incl_all: 所有卫星的轨道参数
    """
    all_raan = []
    all_ma0 = []
    all_radius = []
    all_incl = []

    for i, (n_planes, sats_per_plane, alt, incl_deg, phasing) in enumerate(shells):
        # 对于相同高度和倾角的壳层，偏移 RAAN 避免重叠
        raan_offset = 0.0
        if i > 0:
            prev_planes, _, prev_alt, prev_incl, _ = shells[i-1]
            if abs(alt - prev_alt) < 1.0 and abs(incl_deg - prev_incl) < 0.1:
                raan_offset = np.pi / n_planes  # 偏移半个面间隔

        raan, ma0, radius, incl = generate_walker_delta(
            n_planes, sats_per_plane, alt, incl_deg, phasing, raan_offset
        )
        all_raan.append(raan)
        all_ma0.append(ma0)
        all_radius.extend([radius] * len(raan))
        all_incl.extend([incl] * len(raan))

    return (np.concatenate(all_raan),
            np.concatenate(all_ma0),
            np.array(all_radius),
            np.array(all_incl))


def get_satellite_positions(raan, ma0, radius, incl, t):
    """
    分组传播所有卫星，返回 ECEF 坐标

    为提高效率，按唯一半径-倾角分组传播
    """
    # 按唯一轨道参数分组
    unique_params = set(zip(radius, incl))
    positions = np.zeros((len(raan), 3))

    for r, inc in unique_params:
        mask = (radius == r) & (incl == inc)
        if np.any(mask):
            pos = propagate_orbit(raan[mask], ma0[mask], r, inc, t)
            positions[mask] = pos

    return positions


def get_user_position(lat_deg, lon_deg):
    """地面用户的 ECEF 坐标"""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = R_EARTH * np.cos(lat) * np.cos(lon)
    y = R_EARTH * np.cos(lat) * np.sin(lon)
    z = R_EARTH * np.sin(lat)
    return np.array([x, y, z])


def get_local_frame(lat_deg, lon_deg):
    """地面用户的局部坐标系 (东、北、天)"""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon),
                      -np.sin(lat) * np.sin(lon),
                      np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon),
                   np.cos(lat) * np.sin(lon),
                   np.sin(lat)])

    return east, north, up


def compute_geometry(user_pos, sat_positions, east, north, up):
    """
    计算用户到所有卫星的几何关系

    返回:
        elevation: 仰角数组 (度)
        azimuth: 方位角数组 (度)
        distance: 距离数组 (km)
        direction: 单位方向向量 (n_sats, 3)
    """
    diff = sat_positions - user_pos[np.newaxis, :]  # (n_sats, 3)
    distance = np.linalg.norm(diff, axis=1)
    direction = diff / distance[:, np.newaxis]

    # 仰角 = arcsin(方向·天向)
    sin_el = np.dot(direction, up)
    elevation = np.rad2deg(np.arcsin(np.clip(sin_el, -1, 1)))

    # 方位角
    e_local = np.dot(direction, east)
    n_local = np.dot(direction, north)
    azimuth = np.rad2deg(np.arctan2(e_local, n_local))

    return elevation, azimuth, distance, direction
