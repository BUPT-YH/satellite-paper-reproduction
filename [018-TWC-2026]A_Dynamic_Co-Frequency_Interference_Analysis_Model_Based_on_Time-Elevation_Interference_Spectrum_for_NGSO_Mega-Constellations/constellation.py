"""
Walker星座建模与卫星位置计算
实现论文 Section III-A/B/C 的核心算法
"""
import numpy as np
from config import (
    RE, GM, TE, DT, T_SIM, THETA_MIN,
    N_PLANES, N_SATS_PER_PLANE, N_TOTAL, F_WALKER,
)


def compute_orbital_period(h_orbit):
    """计算轨道周期 (秒)"""
    rs = RE + h_orbit
    return 2 * np.pi * np.sqrt(rs ** 3 / GM)


def compute_angular_velocity(h_orbit):
    """计算卫星角速度 (rad/s)"""
    rs = RE + h_orbit
    return np.sqrt(GM / rs ** 3)


def walker_satellite_positions(t, h_orbit, inclination, n_planes, n_sats_per_plane, f_walker):
    """
    计算Walker星座所有卫星在ECEF坐标系下的位置

    参数:
        t: 仿真时间 (s)
        h_orbit: 轨道高度 (km)
        inclination: 轨道倾角 (度)
        n_planes: 轨道面数
        n_sats_per_plane: 每面卫星数
        f_walker: Walker phasing factor

    返回:
        positions: (n_planes * n_sats_per_plane, 3) ECEF坐标 (km)
        lats: 各卫星纬度 (度)
        lons: 各卫星经度 (度)
    """
    rs = RE + h_orbit
    inc = np.radians(inclination)
    omega_s = compute_angular_velocity(h_orbit)
    n_total = n_planes * n_sats_per_plane

    positions = np.zeros((n_total, 3))
    lats = np.zeros(n_total)
    lons = np.zeros(n_total)

    idx = 0
    for m in range(n_planes):
        for n in range(n_sats_per_plane):
            # RAAN (Eq. 15, 18)
            omega_ref = -2 * np.pi * t / TE
            Omega_mn = omega_ref + 2 * np.pi * m / n_planes

            # 平近点角 (Eq. 16, 17)
            theta_ref = omega_s * t
            theta_mn = theta_ref + 2 * np.pi * (f_walker * m / n_total + n_planes * n / n_total)

            # PQW坐标系下的位置 (Eq. 20)
            s_pqw = np.array([
                rs * np.cos(theta_mn),
                rs * np.sin(theta_mn),
                0.0
            ])

            # PQW → ECI 变换矩阵 (Eq. 21)
            cos_O = np.cos(Omega_mn)
            sin_O = np.sin(Omega_mn)
            cos_i = np.cos(inc)
            sin_i = np.sin(inc)

            R_pqw2eci = np.array([
                [cos_O, -sin_O * cos_i, sin_O * sin_i],
                [sin_O, cos_O * cos_i, -cos_O * sin_i],
                [0, sin_i, cos_i]
            ])

            # ECI坐标
            s_eci = R_pqw2eci @ s_pqw

            # 经纬度 (Eq. 23, 24)
            lon = np.degrees(np.arctan2(s_eci[1], s_eci[0])) % 360
            if lon > 180:
                lon -= 360
            lat = np.degrees(np.arcsin(s_eci[2] / rs))

            # ECEF坐标 (考虑地球自转, 经度修正已在RAAN中)
            positions[idx] = s_eci  # ECI与ECEF的区别通过RAAN修正已包含
            lats[idx] = lat
            lons[idx] = lon
            idx += 1

    return positions, lats, lons


def communication_satellite_position(gs_lat_deg, gs_lon_deg, elevation_deg, h_orbit):
    """
    计算通信卫星位置 (简化模型)

    通信卫星从南向北过境，给定仰角计算卫星ECEF位置

    参数:
        gs_lat_deg: 地面站纬度 (度)
        gs_lon_deg: 地面站经度 (度)
        elevation_deg: 通信卫星仰角 (度)
        h_orbit: 轨道高度 (km)

    返回:
        sat_pos: (3,) 卫星ECEF坐标 (km)
    """
    gs_lat = np.radians(gs_lat_deg)
    gs_lon = np.radians(gs_lon_deg)
    elev = np.radians(elevation_deg)
    rs = RE + h_orbit

    # 地面站ECEF坐标
    gs_pos = RE * np.array([
        np.cos(gs_lat) * np.cos(gs_lon),
        np.cos(gs_lat) * np.sin(gs_lon),
        np.sin(gs_lat)
    ])

    # 通信卫星沿从南到北的方向过境
    # 仰角 elevation_deg 定义为通信链路与南向地平线的夹角
    # 论文中仰角定义为: 与南向地平线的夹角, 0°=南方地平线, 90°=天顶, >90°=北向

    # 将仰角转换为与天顶的夹角
    # 论文中: elevation in figure = angle from south horizon
    # 实际仰角 = elevation (if <= 90) or 180 - elevation (if > 90)
    if elevation_deg <= 90:
        actual_elev = np.radians(elevation_deg)
        azimuth = np.pi  # 南向
    else:
        actual_elev = np.radians(180 - elevation_deg)
        azimuth = 0.0  # 北向

    # 从地面站到卫星的方向向量
    # 在ENU坐标系中
    az = azimuth
    el = actual_elev
    enu_dir = np.array([
        np.sin(az) * np.cos(el),
        np.cos(az) * np.cos(el),
        np.sin(el)
    ])

    # ENU → ECEF 变换
    R_enu2ecef = np.array([
        [-np.sin(gs_lon), -np.sin(gs_lat) * np.cos(gs_lon), np.cos(gs_lat) * np.cos(gs_lon)],
        [np.cos(gs_lon), -np.sin(gs_lat) * np.sin(gs_lon), np.cos(gs_lat) * np.sin(gs_lon)],
        [0, np.cos(gs_lat), np.sin(gs_lat)]
    ])

    ecef_dir = R_enu2ecef @ enu_dir

    # 计算卫星到地面站的距离
    cos_alpha = RE * np.cos(actual_elev) / rs
    alpha = np.arcsin(cos_alpha)  # 地心角
    d_slant = np.sqrt(rs ** 2 + RE ** 2 - 2 * rs * RE * np.cos(alpha))

    # 实际上用仰角直接计算距离
    sin_elev = np.sin(actual_elev)
    d_slant = np.sqrt(RE ** 2 * sin_elev ** 2 + 2 * RE * h_orbit + h_orbit ** 2) - RE * sin_elev

    sat_pos = gs_pos + d_slant * ecef_dir
    return sat_pos


def coarse_filtering(gs_lat_deg, gs_lon_deg, sat_lats, sat_lons, h_orbit, theta_min_deg):
    """
    粗筛选: 筛选星下点落在可见区域内的卫星 (Eq. 30-35)

    返回:
        visible_indices: 可见卫星索引列表
    """
    theta_min = np.radians(theta_min_deg)
    rs = RE + h_orbit

    # 纬度范围 (Eq. 30, 31)
    phi_south = gs_lat_deg - np.degrees(np.arccos(RE * np.cos(theta_min) / rs)) + theta_min_deg
    phi_north = gs_lat_deg + np.degrees(np.arccos(RE * np.cos(theta_min) / rs)) - theta_min_deg

    # 经度范围 (Eq. 32-35)
    psi_max = np.arccos(RE * np.cos(theta_min) / rs) - theta_min

    cos_psi_max = np.cos(psi_max)
    sin2_gs = np.sin(np.radians(gs_lat_deg)) ** 2
    cos2_gs = np.cos(np.radians(gs_lat_deg)) ** 2

    if cos2_gs > 1e-10:
        delta_lon_max = np.degrees(np.arccos(np.clip((cos_psi_max - sin2_gs) / cos2_gs, -1, 1)))
    else:
        delta_lon_max = 180.0

    lon_west = gs_lon_deg - delta_lon_max
    lon_east = gs_lon_deg + delta_lon_max

    visible = []
    for i in range(len(sat_lats)):
        lat_ok = phi_south <= sat_lats[i] <= phi_north
        # 经度比较需要处理跨越180度的情况
        dlon = sat_lons[i] - gs_lon_deg
        if dlon > 180:
            dlon -= 360
        if dlon < -180:
            dlon += 360
        lon_ok = abs(dlon) <= delta_lon_max

        if lat_ok and lon_ok:
            visible.append(i)

    return visible


def fine_filtering(gs_lat_deg, gs_lon_deg, sat_positions, h_orbit, theta_min_deg):
    """
    精筛选: 排除在可见圆域外的卫星 (Eq. 36, 37)

    参数:
        gs_lat_deg, gs_lon_deg: 地面站经纬度
        sat_positions: (N, 3) 卫星ECEF坐标
        h_orbit: 轨道高度
        theta_min_deg: 最小仰角

    返回:
        visible_indices: 精筛选后的可见卫星索引列表
    """
    gs_lat = np.radians(gs_lat_deg)
    gs_lon = np.radians(gs_lon_deg)
    theta_min = np.radians(theta_min_deg)
    rs = RE + h_orbit

    # 地面站ECEF坐标
    GS = RE * np.array([
        np.cos(gs_lat) * np.cos(gs_lon),
        np.cos(gs_lat) * np.sin(gs_lon),
        np.sin(gs_lat)
    ])

    # 天顶卫星位置 (Eq. 36)
    s_gs = rs * np.array([
        np.cos(gs_lat) * np.cos(gs_lon),
        np.cos(gs_lat) * np.sin(gs_lon),
        np.sin(gs_lat)
    ])

    threshold = np.pi / 2 - theta_min

    visible = []
    for i in range(len(sat_positions)):
        # Eq. 37
        vec1 = sat_positions[i] - GS
        vec2 = s_gs - GS
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-30)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        if angle <= threshold:
            visible.append(i)

    return visible


def get_visible_satellites(t, gs_lat, gs_lon, h_orbit, inclination, n_planes, n_sats_per_plane,
                          f_walker=F_WALKER, theta_min=THETA_MIN):
    """
    两阶段筛选获取可见干扰卫星

    返回:
        visible_indices: 可见卫星索引
        all_positions: 所有卫星位置
        all_lats, all_lons: 所有卫星经纬度
    """
    all_positions, all_lats, all_lons = walker_satellite_positions(
        t, h_orbit, inclination, n_planes, n_sats_per_plane, f_walker
    )

    # 粗筛选
    coarse_visible = coarse_filtering(gs_lat, gs_lon, all_lats, all_lons, h_orbit, theta_min)

    if len(coarse_visible) == 0:
        return [], all_positions, all_lats, all_lons

    # 精筛选
    coarse_positions = all_positions[coarse_visible]
    fine_visible_local = fine_filtering(gs_lat, gs_lon, coarse_positions, h_orbit, theta_min)

    # 映射回原始索引
    visible_indices = [coarse_visible[i] for i in fine_visible_local]

    return visible_indices, all_positions, all_lats, all_lons


def compute_elevation(gs_pos, sat_pos):
    """计算卫星相对于地面站的仰角"""
    diff = sat_pos - gs_pos
    dist = np.linalg.norm(diff)

    gs_lat = np.arcsin(gs_pos[2] / np.linalg.norm(gs_pos))
    r_gs = np.linalg.norm(gs_pos)

    # 仰角计算
    cos_alpha = np.dot(gs_pos, diff) / (r_gs * dist)
    alpha = np.arccos(np.clip(cos_alpha, -1, 1))

    elev = np.arccos(np.clip(r_gs * np.sin(alpha) / dist, -1, 1)) - np.pi / 2 + alpha
    # 简化计算
    elev = np.arcsin(np.clip(
        (np.dot(diff, gs_pos / r_gs)) / dist, -1, 1
    ))

    return np.degrees(elev)


def get_interfering_terminals(gs_lat_deg, gs_lon_deg, int_radius_km, max_terminals, seed=42):
    """
    在被干扰终端周围的圆形区域内生成干扰终端位置 (均匀分布近似PPP)

    返回:
        terminal_positions: (N, 3) ECEF坐标
    """
    rng = np.random.RandomState(seed)
    n_terminals = rng.poisson(max_terminals * 0.5)  # 期望50个终端
    n_terminals = max(1, min(n_terminals, max_terminals))

    gs_lat = np.radians(gs_lat_deg)
    gs_lon = np.radians(gs_lon_deg)

    # 在极坐标下均匀分布
    r = int_radius_km * np.sqrt(rng.uniform(0, 1, n_terminals))
    theta = rng.uniform(0, 2 * np.pi, n_terminals)

    # 转换为经纬度偏移
    # 近似: 1度纬度 ≈ 111km, 1度经度 ≈ 111*cos(lat)km
    lat_per_km = 1.0 / 111.0
    lon_per_km = 1.0 / (111.0 * np.cos(gs_lat))

    dlat = r * np.cos(theta) * lat_per_km
    dlon = r * np.sin(theta) * lon_per_km

    terminal_lats = gs_lat_deg + dlat
    terminal_lons = gs_lon_deg + dlon

    # 转ECEF
    terminal_positions = np.zeros((n_terminals, 3))
    for i in range(n_terminals):
        lat_r = np.radians(terminal_lats[i])
        lon_r = np.radians(terminal_lons[i])
        terminal_positions[i] = RE * np.array([
            np.cos(lat_r) * np.cos(lon_r),
            np.cos(lat_r) * np.sin(lon_r),
            np.sin(lat_r)
        ])

    return terminal_positions, terminal_lats, terminal_lons


def find_linked_satellite_max_elev(gs_lat_deg, gs_lon_deg, terminal_pos, all_positions,
                                   visible_indices, h_orbit, theta_min_deg):
    """
    为干扰终端找到最大仰角关联的卫星

    返回:
        best_idx: 可见卫星中最大仰角的索引 (在visible_indices中的位置)
        best_elev: 对应仰角
    """
    gs_lat = np.radians(gs_lat_deg)
    gs_lon = np.radians(gs_lon_deg)
    theta_min = np.radians(theta_min_deg)

    best_elev = -90
    best_idx = -1

    for i, idx in enumerate(visible_indices):
        sat_pos = all_positions[idx]
        diff = sat_pos - terminal_pos
        dist = np.linalg.norm(diff)
        r_term = np.linalg.norm(terminal_pos)

        elev_rad = np.arcsin(np.clip(
            np.dot(diff, terminal_pos / r_term) / dist, -1, 1
        ))
        elev = np.degrees(elev_rad)

        if elev > best_elev and elev >= theta_min_deg:
            best_elev = elev
            best_idx = i

    return best_idx, best_elev
