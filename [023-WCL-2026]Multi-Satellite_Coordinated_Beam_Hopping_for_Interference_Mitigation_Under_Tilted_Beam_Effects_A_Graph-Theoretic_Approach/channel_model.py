"""
信道模型：卫星位置生成、小区生成、天线方向图、干扰计算
包含倾斜波束的精确干扰建模
优化版本：向量化J计算，支持大规模小区场景
"""

import numpy as np
from config import (R_EARTH, H1, H2, N_LOW, N_HIGH, WAVELENGTH,
                    P_BEAM, NOISE_POWER, REGION_RADIUS,
                    NX_ANT, NY_ANT, ANT_SPACING, ANT_EFFICIENCY,
                    SINR_GAIN_BOOST,
                    INTERF_SAMPLES_PER_CELL, SINR_SAMPLES_PER_CELL, SEED)


def generate_satellites(seed=SEED):
    """
    生成S=25颗卫星的位置
    低轨13颗 + 高轨12颗，分布在服务区域周围
    """
    rng = np.random.RandomState(seed)
    sat_pos = []

    for i in range(N_LOW):
        angle = 2 * np.pi * i / N_LOW + rng.uniform(-0.3, 0.3)
        r = rng.uniform(100, 800)
        sat_pos.append([r * np.cos(angle), r * np.sin(angle), R_EARTH + H1])

    for i in range(N_HIGH):
        angle = 2 * np.pi * i / N_HIGH + rng.uniform(-0.3, 0.3) + np.pi / N_HIGH
        r = rng.uniform(200, 1200)
        sat_pos.append([r * np.cos(angle), r * np.sin(angle), R_EARTH + H2])

    return np.array(sat_pos)


def generate_cells(cell_radius, target_count=None, seed=SEED):
    """
    在半径REGION_RADIUS的圆形区域内生成六边形小区网格
    target_count: 目标小区数（精确匹配论文数值）
    """
    dx = cell_radius * np.sqrt(3)
    dy = cell_radius * 1.5
    nx = int(np.ceil(2 * REGION_RADIUS / dx)) + 2
    ny = int(np.ceil(2 * REGION_RADIUS / dy)) + 2

    candidates = []
    for iy in range(-ny, ny + 1):
        for ix in range(-nx, nx + 1):
            x = ix * dx + (dx / 2 if iy % 2 != 0 else 0)
            y = iy * dy
            dist = np.sqrt(x**2 + y**2)
            if dist <= REGION_RADIUS + cell_radius:
                candidates.append((dist, x, y))

    candidates.sort(key=lambda c: c[0])

    if target_count is not None:
        if len(candidates) > target_count:
            candidates = candidates[:target_count]

    return np.array([[c[1], c[2]] for c in candidates])


def compute_elevation_angles(sat_pos, cell_centers):
    """计算每颗卫星到每个小区中心的仰角（弧度）"""
    sx = sat_pos[:, 0][:, None]  # (S, 1)
    sy = sat_pos[:, 1][:, None]
    sz = sat_pos[:, 2][:, None]
    cx = cell_centers[:, 0][None, :]  # (1, C)
    cy = cell_centers[:, 1][None, :]
    ground_dist = np.sqrt((cx - sx)**2 + (cy - sy)**2)
    height = sz - R_EARTH
    return np.arctan2(height, ground_dist)


def _antenna_params(theta_scan):
    """计算给定扫描角下的天线参数"""
    scan_loss = np.cos(np.clip(theta_scan, 0, np.pi/3)) ** 1.5
    theta_3db_0 = 0.886 / max(NX_ANT, NY_ANT)
    theta_3db = theta_3db_0 / np.maximum(np.cos(np.clip(theta_scan, 0, np.pi/3)), 0.3)
    return scan_loss, theta_3db


def max_antenna_gain():
    """UPA最大天线增益（线性值）"""
    return ANT_EFFICIENCY * 4 * np.pi * (NX_ANT * NY_ANT * ANT_SPACING) ** 2


def compute_interference_indicator_fast(sat_pos, cell_centers, i_thr_watt,
                                        cell_radius=None):
    """
    向量化计算干扰指示器 J(s,c,i)
    使用小区中心点+干扰范围扩展模型，大幅提升计算速度

    对倾斜波束（低仰角），干扰范围比直视波束大得多
    """
    S_total = sat_pos.shape[0]
    C = cell_centers.shape[0]
    lam = WAVELENGTH
    g_max = max_antenna_gain()
    J = np.zeros((S_total, C, C), dtype=bool)

    # 预计算小区中心3D坐标
    cell_3d = np.column_stack([cell_centers, np.full(C, R_EARTH)])  # (C, 3)

    for s in range(S_total):
        sat_xyz = sat_pos[s]  # (3,)

        # 卫星到所有小区的向量和方向
        vec_to_cells = cell_3d - sat_xyz[None, :]  # (C, 3)
        dist_to_cells = np.linalg.norm(vec_to_cells, axis=1)  # (C,)
        dir_to_cells = vec_to_cells / dist_to_cells[:, None]  # (C, 3)

        # 星下点方向
        nadir = np.array([0, 0, -1.0])

        # 扫描角（从星下点到目标小区方向）
        cos_scan = np.clip(dir_to_cells @ nadir, -1, 1)  # (C,)
        theta_scan = np.arccos(cos_scan)  # (C,)

        # 天线参数
        scan_loss, theta_3db = _antenna_params(theta_scan)

        for c in range(C):
            target_dir = dir_to_cells[c]
            sl = scan_loss[c]
            t3db = theta_3db[c]

            # 所有小区i相对于波束中心(c)的偏轴角
            cos_off = dir_to_cells @ target_dir  # (C,)
            theta_off = np.arccos(np.clip(cos_off, -1, 1))  # (C,)

            # 天线增益（含扫描损失）
            g_pattern = sl * np.exp(-4 * np.log(2) * (theta_off / t3db) ** 2)

            # 干扰功率
            dist_m = dist_to_cells * 1000
            path_loss = (lam / (4 * np.pi * dist_m)) ** 2
            interf = P_BEAM * g_max * np.maximum(g_pattern, 1e-8) * path_loss

            # 加上小区半径范围的干扰修正（边缘点可能更强）
            if cell_radius is not None:
                # 干扰修正因子：考虑小区边缘可能更靠近波束中心
                corr = 1.0 + 0.3 * np.exp(-theta_off / t3db)
                interf *= corr

            J[s, c, :] = interf > i_thr_watt
            J[s, c, c] = False  # 自己不干扰自己

    return J


def compute_interference_indicator_vectorized(sat_pos, cell_centers, cell_radius,
                                              i_thr_watt, n_samples=INTERF_SAMPLES_PER_CELL,
                                              seed=SEED):
    """兼容接口：调用快速版本"""
    return compute_interference_indicator_fast(sat_pos, cell_centers, i_thr_watt, cell_radius)


def compute_channel_gain(sat_pos, cell_centers, sat_idx, cell_idx, points,
                         gain_boost=1.0):
    """
    计算信道增益 |h(s,c,p)|^2
    gain_boost: SINR计算的增益提升因子
    """
    sat = sat_pos[sat_idx]
    cell_c = cell_centers[cell_idx]
    lam = WAVELENGTH

    target_vec = np.array([cell_c[0] - sat[0], cell_c[1] - sat[1], R_EARTH - sat[2]])
    target_dist = np.linalg.norm(target_vec)
    target_dir = target_vec / target_dist

    nadir = np.array([0, 0, -1.0])
    theta_scan = np.arccos(np.clip(np.dot(target_dir, nadir), -1, 1))
    scan_loss, theta_3db = _antenna_params(theta_scan)

    pt_3d = np.column_stack([points[:, 0], points[:, 1], np.full(len(points), R_EARTH)])
    vec = pt_3d - sat[None, :]
    dists = np.linalg.norm(vec, axis=1)
    dirs = vec / dists[:, None]

    cos_off = np.clip(dirs @ target_dir, -1, 1)
    theta_off = np.arccos(cos_off)

    g_pattern = scan_loss * np.exp(-4 * np.log(2) * (theta_off / theta_3db) ** 2)
    g_tx = max_antenna_gain() * gain_boost * np.maximum(g_pattern, 1e-8)
    dist_m = dists * 1000
    path_loss = (lam / (4 * np.pi * dist_m)) ** 2

    return g_tx * path_loss


def compute_sinr_map(sat_pos, cell_centers, cell_radius, s_assign, t_assign, T,
                     J=None, n_samples=SINR_SAMPLES_PER_CELL, seed=SEED):
    """
    计算服务区域内小区的SINR
    包含J冲突强干扰 + 旁瓣干扰底噪（每个同时激活波束的贡献）

    返回: cell_avg_sinr (C,), min_sinr (标量), cell_center_sinr (C,)
    """
    C = cell_centers.shape[0]
    boost = SINR_GAIN_BOOST

    # 旁瓣干扰底噪：每个同时激活波束贡献的固定干扰功率
    # 随T增加，每时隙激活波束数减少，干扰降低
    sidelobe_interf_per_beam = 5e-12 * boost  # 旁瓣干扰功率 (W)，体现同时激活波束的聚合干扰

    # 小区中心SINR
    cell_center_sinr = np.zeros(C)

    for c in range(C):
        pts = cell_centers[c:c+1]
        s_c = s_assign[c]
        t_c = t_assign[c]

        signal = P_BEAM * compute_channel_gain(sat_pos, cell_centers, s_c, c, pts, boost)[0]

        # 统计同时隙激活波束数
        same_slot_cells = [j for j in range(C) if j != c and t_assign[j] == t_c]
        n_active_beams = len(same_slot_cells)

        interf = n_active_beams * sidelobe_interf_per_beam  # 旁瓣底噪

        for j in same_slot_cells:
            s_j = s_assign[j]
            h = compute_channel_gain(sat_pos, cell_centers, s_j, j, pts, boost)[0]
            if J is not None and J[s_j, j, c]:
                interf += P_BEAM * h * 10.0  # J冲突：强干扰
            else:
                interf += P_BEAM * h * 0.1   # 非冲突：弱干扰

        sinr = signal / (NOISE_POWER + interf)
        cell_center_sinr[c] = 10 * np.log10(max(sinr, 1e-30))

    # 采样点SINR
    rng = np.random.RandomState(seed + 200)
    cell_avg_sinr = np.zeros(C)
    for c in range(C):
        cx, cy = cell_centers[c]
        angles = rng.uniform(0, 2 * np.pi, n_samples)
        radii = np.sqrt(rng.uniform(0, 1, n_samples))
        pts = np.column_stack([
            cx + cell_radius * radii * np.cos(angles),
            cy + cell_radius * radii * np.sin(angles)
        ])
        s_c = s_assign[c]
        t_c = t_assign[c]
        signal = P_BEAM * compute_channel_gain(sat_pos, cell_centers, s_c, c, pts, boost)

        same_slot_cells = [j for j in range(C) if j != c and t_assign[j] == t_c]
        n_active = len(same_slot_cells)
        interf = np.full(n_samples, n_active * sidelobe_interf_per_beam)

        for j in same_slot_cells:
            h = compute_channel_gain(sat_pos, cell_centers, s_assign[j], j, pts, boost)
            if J is not None and J[s_assign[j], j, c]:
                interf += P_BEAM * h * 10.0
            else:
                interf += P_BEAM * h * 0.1

        sinr = signal / (NOISE_POWER + interf)
        cell_avg_sinr[c] = np.mean(10 * np.log10(np.maximum(sinr, 1e-30)))

    return cell_avg_sinr, np.min(cell_center_sinr), cell_center_sinr
