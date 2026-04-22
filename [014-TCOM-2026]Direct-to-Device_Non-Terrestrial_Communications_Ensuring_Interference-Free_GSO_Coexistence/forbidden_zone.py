"""
论文复现 - 禁区 (Forbidden Zone) 几何计算模块
实现论文 Eq. (1)-(5) 的禁区判定算法
"""
import numpy as np
from config import (R_EARTH, GSO_RADIUS, SAT_HPBW, MIN_ELEVATION, BETA_MIN)


def latlon_to_ecef(lat_deg, lon_deg, alt_km):
    """
    经纬度转 ECEF 坐标
    lat_deg: 纬度 (度)
    lon_deg: 经度 (度)
    alt_km: 海拔高度 (km)
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = R_EARTH + alt_km
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])


def ecef_to_latlon(x, y, z):
    """ECEF 坐标转经纬度"""
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    alt = r - R_EARTH
    return lat, lon, alt


def generate_walker_delta(num_orbits, sats_per_orbit, altitude_km, inclination_deg,
                          t_since_raan=0):
    """
    生成 Walker-delta 星座卫星位置
    返回每个卫星的 ECEF 坐标 (N, 3)

    参数:
        num_orbits: 轨道面数
        sats_per_orbit: 每轨道卫星数
        altitude_km: 轨道高度 (km)
        inclination_deg: 轨道倾角 (度)
        t_since_raan: 升交点赤经参考时刻 (用于时间演化)
    """
    # 轨道周期
    mu_earth = 3.986e5  # km^3/s^2
    orbital_radius = R_EARTH + altitude_km
    period = 2 * np.pi * np.sqrt(orbital_radius**3 / mu_earth)
    omega_orbit = 2 * np.pi / period  # 角速度 (rad/s)

    total_sats = num_orbits * sats_per_orbit
    positions = np.zeros((total_sats, 3))
    sat_index = 0

    inc = np.radians(inclination_deg)

    for p in range(num_orbits):
        # 升交点赤经 (RAAN)
        raan = 2 * np.pi * p / num_orbits

        for s in range(sats_per_orbit):
            # 真近点角（初始均匀分布 + 时间演化）
            nu = 2 * np.pi * s / sats_per_orbit + omega_orbit * t_since_raan

            # 在轨道平面内的位置
            x_orb = orbital_radius * np.cos(nu)
            y_orb = orbital_radius * np.sin(nu)

            # 从轨道平面坐标转到 ECEF
            # 旋转: 先绕 z 轴旋转 RAAN，再绕 x 轴旋转 inclination
            cos_raan, sin_raan = np.cos(raan), np.sin(raan)
            cos_inc, sin_inc = np.cos(inc), np.sin(inc)

            x = cos_raan * x_orb - sin_raan * cos_inc * y_orb
            y = sin_raan * x_orb + cos_raan * cos_inc * y_orb
            z = sin_inc * y_orb

            positions[sat_index] = [x, y, z]
            sat_index += 1

    return positions


def compute_elevation_angle(user_pos, sat_pos):
    """
    计算卫星相对于用户的仰角
    user_pos, sat_pos: ECEF 坐标 (km)
    返回仰角 (度)
    """
    # 用户位置的单位向量（指向天顶方向）
    user_r = np.linalg.norm(user_pos)
    user_zenith = user_pos / user_r

    # 卫星相对于用户的向量
    diff = sat_pos - user_pos
    dist = np.linalg.norm(diff)
    diff_unit = diff / dist

    # 仰角 = 90° - 用户天顶与卫星方向的夹角
    cos_zenith_angle = np.dot(user_zenith, diff_unit)
    zenith_angle = np.degrees(np.arccos(np.clip(cos_zenith_angle, -1, 1)))
    elevation = 90.0 - zenith_angle

    return elevation


def find_cone_axis_point(leo_pos, gso_pos):
    """
    找到锥体轴线与地球表面的交点 O_{i,g}
    论文: 将 GSO-LEO 连线从 LEO 侧延伸到地球表面

    几何: 从 GSO 穿过 LEO 延伸到地球表面，交点 O_{i,g}
    方向向量 = leo_pos - gso_pos (从 GSO 指向 LEO，延伸过 LEO 到达地球)

    参数:
        leo_pos: LEO 卫星 ECEF 坐标 (km)
        gso_pos: GSO 卫星 ECEF 坐标 (km)
    返回:
        交点 ECEF 坐标
    """
    # 从 GSO 穿过 LEO 的方向（朝地球方向延伸）
    direction = leo_pos - gso_pos
    direction = direction / np.linalg.norm(direction)

    # 求解: ||leo_pos + t * direction|| = R_EARTH
    # 即: t^2 + 2*(leo_pos·direction)*t + (||leo_pos||^2 - R_EARTH^2) = 0
    a = 1.0
    b = 2.0 * np.dot(leo_pos, direction)
    c = np.dot(leo_pos, leo_pos) - R_EARTH**2

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None  # 不与地球相交

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2*a)  # 取较小值（近交点）
    t2 = (-b + sqrt_disc) / (2*a)

    # 取正的 t 值（在射线方向上）
    if t1 > 0:
        t = t1
    elif t2 > 0:
        t = t2
    else:
        return None

    return leo_pos + t * direction


def is_in_forbidden_zone(user_pos, leo_pos, gso_positions, cone_half_angle_deg):
    """
    判断用户是否在 LEO 卫星的禁区 (Forbidden Zone) 内
    论文 Eq. (3)-(5)

    参数:
        user_pos: 用户 ECEF 坐标 (km)
        leo_pos: LEO 卫星 ECEF 坐标 (km)
        gso_positions: GSO 卫星位置数组 (G, 3)
        cone_half_angle_deg: 锥体半角 (度)
    返回:
        (bool, int): (是否在禁区内, 触发的 GSO 数量)
    """
    alpha = np.radians(cone_half_angle_deg)
    user_r = np.linalg.norm(user_pos)

    # 检查用户是否在地球表面
    if abs(user_r - R_EARTH) > 10:  # 10 km 容差
        user_pos = user_pos * (R_EARTH / user_r)  # 投影到地球表面

    fz_count = 0
    for g in range(len(gso_positions)):
        # 找到轴线与地球交点 O_{i,g}
        O_ig = find_cone_axis_point(leo_pos, gso_positions[g])
        if O_ig is None:
            continue

        # 计算角 ∠O_{i,g} L_i M_{i,g} (Eq. 3)
        L_O = O_ig - leo_pos
        L_M = user_pos - leo_pos

        L_O_norm = np.linalg.norm(L_O)
        L_M_norm = np.linalg.norm(L_M)

        if L_O_norm < 1e-6 or L_M_norm < 1e-6:
            continue

        cos_angle = np.dot(L_O, L_M) / (L_O_norm * L_M_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        if angle <= alpha:
            fz_count += 1

    return fz_count > 0, fz_count


def compute_forbidden_zones_over_time(user_lat, user_lon, constellation_positions_func,
                                       gso_positions, cone_half_angle_deg, min_elev_deg,
                                       total_time, delta_t):
    """
    计算整个时间窗口内用户位置的禁区卫星数量

    参数:
        user_lat, user_lon: 用户经纬度 (度)
        constellation_positions_func: 函数(t) 返回卫星 ECEF 坐标 (M, 3)
        gso_positions: GSO 位置 (G, 3)
        cone_half_angle_deg: 锥体半角 (度)
        min_elev_deg: 最小仰角 (度)
        total_time: 总仿真时间 (秒)
        delta_t: 时间步长 (秒)

    返回:
        times: 时间数组 (秒)
        num_fz_sats: 每个时刻禁区卫星数量
        num_visible_sats: 每个时刻可见卫星数量
    """
    user_pos = latlon_to_ecef(user_lat, user_lon, 0)

    n_steps = int(total_time / delta_t)
    times = np.arange(n_steps) * delta_t
    num_fz_sats = np.zeros(n_steps, dtype=int)
    num_visible_sats = np.zeros(n_steps, dtype=int)

    for t_idx, t in enumerate(times):
        leo_positions = constellation_positions_func(t)

        for i in range(len(leo_positions)):
            # 检查仰角
            elev = compute_elevation_angle(user_pos, leo_positions[i])
            if elev < min_elev_deg:
                continue

            num_visible_sats[t_idx] += 1

            # 检查禁区
            in_fz, _ = is_in_forbidden_zone(user_pos, leo_positions[i],
                                            gso_positions, cone_half_angle_deg)
            if in_fz:
                num_fz_sats[t_idx] += 1

    return times, num_fz_sats, num_visible_sats
