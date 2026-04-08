"""
干扰模型核心算法
包含: 单轨道、共面轨道、偏移轨道三种干扰场景的数学模型
"""

import numpy as np
from config import R_E, k_B, c


# ===================== 单轨道干扰模型 =====================

def single_orbit_N1(N, h, alpha_deg):
    """
    计算单轨道场景下干扰卫星数量 N1 (Eq. 5)
    N: 轨道中卫星总数
    h: 轨道高度 (m)
    alpha_deg: 波束宽度 (度)
    """
    alpha_rad = np.radians(alpha_deg)
    # 条件1: LOS可见性
    thresh1 = (N / np.pi) * np.arccos(R_E / (R_E + h))
    # 条件2/3: 天线对准
    thresh2 = 1 + N * alpha_rad / (2 * np.pi)
    N1 = max(0, int(np.floor(min(thresh1, thresh2))) - 1)
    return N1


def single_orbit_SIR(N, h, alpha_deg):
    """
    单轨道 SIR (Eq. 8)
    Gamma_1 = 1 / [(1-cos(2pi/N)) * sum_{i=2}^{N1+1} 1/(1-cos(2pi*i/N))]
    返回线性值
    """
    N1 = single_orbit_N1(N, h, alpha_deg)
    if N1 < 1:
        # 无干扰，SIR无穷大
        return np.inf

    numerator = 1.0 - np.cos(2 * np.pi / N)
    denominator = 0.0
    for i in range(2, N1 + 2):
        denominator += 1.0 / (1.0 - np.cos(2 * np.pi * i / N))

    if denominator == 0:
        return np.inf
    return numerator / denominator


def single_orbit_SINR(N, h, band_params, alpha_deg):
    """
    单轨道 SINR (Eq. 9)
    S1 = PRx / (E[I1] + PN)
    返回线性值
    """
    alpha_rad = np.radians(alpha_deg)
    lam = c / band_params['freq']
    P_Tx = band_params['P_Tx']
    G = 2.0 / (1.0 - np.cos(alpha_rad / 2))
    P_N = k_B * band_params['T_sys'] * band_params['bandwidth']

    N1 = single_orbit_N1(N, h, alpha_deg)

    # 信号功率 PRx = P_Tx * G^2 / (4*pi*d1)^2
    # d1^2 = 2*(R_E+h)^2 * (1 - cos(2*pi/N))
    d1_sq = 2 * (R_E + h)**2 * (1 - np.cos(2 * np.pi / N))
    P_Rx = P_Tx * G**2 * lam**2 / ((4 * np.pi)**2 * d1_sq)

    # 干扰功率 E[I1] (Eq. 7)
    E_I1 = 0.0
    if N1 >= 1:
        for i in range(2, N1 + 2):
            di_sq = 2 * (R_E + h)**2 * (1 - np.cos(2 * np.pi * i / N))
            E_I1 += lam**2 * P_Tx / (8 * np.pi**2 * (1 - np.cos(alpha_rad / 2))**2 * di_sq)

    total_interference = E_I1 + P_N
    if total_interference == 0:
        return np.inf
    return P_Rx / total_interference


def single_orbit_capacity(N, h, band_params, alpha_deg):
    """
    单轨道信道容量 C = B * log2(1 + SINR) (bps)
    """
    sinr = single_orbit_SINR(N, h, band_params, alpha_deg)
    if sinr == np.inf:
        sinr = 1e15
    return band_params['bandwidth'] * np.log2(1 + sinr)


# ===================== 偏移轨道干扰模型 =====================

def shifted_orbit_interference(N, N_S, h, gamma_deg, Delta_Omega_deg,
                                alpha_deg, Delta_beta_deg, band_params):
    """
    偏移轨道干扰模型 (Eq. 31-45)
    计算来自偏移轨道的干扰 E[I3]

    N: 本轨道卫星数
    N_S: 偏移轨道卫星数
    h: 轨道高度 (m)
    gamma_deg: 轨道倾角 (度)
    Delta_Omega_deg: RAAN偏移 (度)
    alpha_deg: 波束宽度 (度)
    Delta_beta_deg: 相对角偏移 (度)
    band_params: 频段参数字典
    """
    alpha_rad = np.radians(alpha_deg)
    gamma = np.radians(gamma_deg)
    Delta_Omega = np.radians(Delta_Omega_deg)
    Delta_beta = np.radians(Delta_beta_deg)
    lam = c / band_params['freq']
    P_Tx = band_params['P_Tx']
    G_factor = lam**2 * P_Tx / (4 * np.pi**2 * (1 - np.cos(alpha_rad / 2))**2)

    # 轨道1 (含目标链路) 的卫星位置
    # 发射卫星: phi=0, 接收卫星: phi=2*pi/N
    def orbital_to_GEC(phi, Omega_k, gamma_val):
        """轨道平面坐标 -> GEC坐标 (Eq. 32-33)"""
        # 轨道平面内位置
        r0 = np.array([(R_E + h) * np.cos(phi),
                        (R_E + h) * np.sin(phi),
                        0.0])
        # 转换矩阵 Mk (Eq. 33)
        Omega = Omega_k
        M = np.array([
            [np.cos(Omega), -np.sin(Omega)*np.cos(gamma_val), np.sin(Omega)*np.sin(gamma_val)],
            [np.sin(Omega),  np.cos(Omega)*np.cos(gamma_val),-np.cos(Omega)*np.sin(gamma_val)],
            [0,              np.sin(gamma_val),                 np.cos(gamma_val)]
        ])
        return M @ r0

    # 目标发射和接收卫星位置
    Omega1 = 0  # 轨道1 RAAN = 0
    r_Tx = orbital_to_GEC(0, Omega1, gamma)
    r_Rx = orbital_to_GEC(2 * np.pi / N, Omega1, gamma)

    # 接收指向方向
    r_Rx_to_Tx = r_Tx - r_Rx

    # 遍历偏移轨道(轨道2)中的每颗卫星
    Omega2 = Delta_Omega
    E_I3 = 0.0

    for j in range(N_S):
        phi_j = Delta_beta + 2 * np.pi * j / N_S
        r_j = orbital_to_GEC(phi_j, Omega2, gamma)

        # 检查条件1: 卫星间可见性 (Eq. 34-35)
        r_j_to_Rx = r_j - r_Rx
        r_Rx_to_j = r_j - r_Rx

        # Rise and set function R_j
        dot_rr = np.dot(r_j, r_Rx)
        norm_j_sq = np.dot(r_j, r_j)
        norm_Rx_sq = np.dot(r_Rx, r_Rx)
        R_j = (dot_rr**2 - norm_j_sq * norm_Rx_sq
               + (norm_j_sq + norm_Rx_sq) * R_E**2
               - 2 * R_E**2 * dot_rr)
        if R_j > 0:
            continue  # 被地球遮挡

        # 条件2: 干扰卫星在接收波束内 (Eq. 39, 41)
        r_Rx_to_j_vec = r_j - r_Rx
        norm_rx_j = np.linalg.norm(r_Rx_to_j_vec)
        norm_rx_tx = np.linalg.norm(r_Rx_to_Tx)
        if norm_rx_j == 0 or norm_rx_tx == 0:
            continue
        cos_psi_j = np.dot(r_Rx_to_j_vec, r_Rx_to_Tx) / (norm_rx_j * norm_rx_tx)
        cos_psi_j = np.clip(cos_psi_j, -1.0, 1.0)
        psi_j = np.arccos(cos_psi_j)
        if psi_j > alpha_rad / 2:
            continue

        # 条件3: 接收机在干扰卫星发射波束内 (Eq. 40, 42)
        # 干扰卫星的指向: 指向同一轨道的前一颗卫星 (j-1)
        phi_j_prev = Delta_beta + 2 * np.pi * ((j - 1) % N_S) / N_S
        r_j_prev = orbital_to_GEC(phi_j_prev, Omega2, gamma)
        r_j_to_prev = r_j_prev - r_j
        r_j_to_Rx = r_Rx - r_j
        norm_j_prev = np.linalg.norm(r_j_to_prev)
        norm_j_Rx = np.linalg.norm(r_j_to_Rx)
        if norm_j_prev == 0 or norm_j_Rx == 0:
            continue
        cos_psi_j_prime = np.dot(r_j_to_prev, r_j_to_Rx) / (norm_j_prev * norm_j_Rx)
        cos_psi_j_prime = np.clip(cos_psi_j_prime, -1.0, 1.0)
        psi_j_prime = np.arccos(cos_psi_j_prime)
        if psi_j_prime > alpha_rad / 2:
            continue

        # 所有条件满足，累加干扰功率 (Eq. 43)
        d_j_sq = norm_rx_j**2
        E_I3 += G_factor / d_j_sq

    return E_I3


def shifted_orbit_SINR(N, N_S, h, gamma_deg, Delta_Omega_deg,
                       alpha_deg, Delta_beta_deg, band_params):
    """
    偏移轨道 SINR (Eq. 45)
    S3 = PRx / (E[I1] + E[I3] + PN)
    """
    alpha_rad = np.radians(alpha_deg)
    lam = c / band_params['freq']
    P_Tx = band_params['P_Tx']
    G = 2.0 / (1.0 - np.cos(alpha_rad / 2))
    P_N = k_B * band_params['T_sys'] * band_params['bandwidth']

    # 信号功率
    d1_sq = 2 * (R_E + h)**2 * (1 - np.cos(2 * np.pi / N))
    P_Rx = P_Tx * G**2 * lam**2 / ((4 * np.pi)**2 * d1_sq)

    # 同轨道干扰
    N1 = single_orbit_N1(N, h, alpha_deg)
    E_I1 = 0.0
    if N1 >= 1:
        for i in range(2, N1 + 2):
            di_sq = 2 * (R_E + h)**2 * (1 - np.cos(2 * np.pi * i / N))
            E_I1 += lam**2 * P_Tx / (8 * np.pi**2 * (1 - np.cos(alpha_rad / 2))**2 * di_sq)

    # 偏移轨道干扰
    E_I3 = shifted_orbit_interference(N, N_S, h, gamma_deg, Delta_Omega_deg,
                                       alpha_deg, Delta_beta_deg, band_params)

    total = E_I1 + E_I3 + P_N
    if total == 0:
        return np.inf
    return P_Rx / total


# ===================== 共面偏移轨道干扰 =====================

def coplanar_shifted_interference(N, N_S, h, h_S, gamma_deg, Delta_Omega_deg,
                                   alpha_deg, Delta_beta_deg, band_params):
    """
    共面偏移轨道 (不同高度) 干扰模型
    偏移轨道在不同高度 h_S
    """
    alpha_rad = np.radians(alpha_deg)
    gamma = np.radians(gamma_deg)
    Delta_Omega = np.radians(Delta_Omega_deg)
    Delta_beta = np.radians(Delta_beta_deg)
    lam = c / band_params['freq']
    P_Tx = band_params['P_Tx']
    G_factor = lam**2 * P_Tx / (4 * np.pi**2 * (1 - np.cos(alpha_rad / 2))**2)

    def orbital_to_GEC_diff_h(phi, Omega_k, gamma_val, altitude):
        """不同高度的轨道平面坐标 -> GEC坐标"""
        r0 = np.array([(R_E + altitude) * np.cos(phi),
                        (R_E + altitude) * np.sin(phi),
                        0.0])
        Omega = Omega_k
        M = np.array([
            [np.cos(Omega), -np.sin(Omega)*np.cos(gamma_val), np.sin(Omega)*np.sin(gamma_val)],
            [np.sin(Omega),  np.cos(Omega)*np.cos(gamma_val),-np.cos(Omega)*np.sin(gamma_val)],
            [0,              np.sin(gamma_val),                 np.cos(gamma_val)]
        ])
        return M @ r0

    # 目标链路
    Omega1 = 0
    r_Tx = orbital_to_GEC_diff_h(0, Omega1, gamma, h)
    r_Rx = orbital_to_GEC_diff_h(2 * np.pi / N, Omega1, gamma, h)
    r_Rx_to_Tx = r_Tx - r_Rx

    E_I = 0.0
    for j in range(N_S):
        phi_j = Delta_beta + 2 * np.pi * j / N_S
        r_j = orbital_to_GEC_diff_h(phi_j, Delta_Omega, gamma, h_S)

        # 条件1: 可见性
        r_Rx_to_j = r_j - r_Rx
        dot_rr = np.dot(r_j, r_Rx)
        norm_j_sq = np.dot(r_j, r_j)
        norm_Rx_sq = np.dot(r_Rx, r_Rx)
        R_j = (dot_rr**2 - norm_j_sq * norm_Rx_sq
               + (norm_j_sq + norm_Rx_sq) * R_E**2
               - 2 * R_E**2 * dot_rr)
        if R_j > 0:
            continue

        # 条件2: 接收波束
        norm_rx_j = np.linalg.norm(r_Rx_to_j)
        norm_rx_tx = np.linalg.norm(r_Rx_to_Tx)
        if norm_rx_j == 0 or norm_rx_tx == 0:
            continue
        cos_psi = np.dot(r_Rx_to_j, r_Rx_to_Tx) / (norm_rx_j * norm_rx_tx)
        cos_psi = np.clip(cos_psi, -1.0, 1.0)
        if np.arccos(cos_psi) > alpha_rad / 2:
            continue

        # 条件3: 干扰卫星发射波束
        phi_j_prev = Delta_beta + 2 * np.pi * ((j - 1) % N_S) / N_S
        r_j_prev = orbital_to_GEC_diff_h(phi_j_prev, Delta_Omega, gamma, h_S)
        r_j_to_prev = r_j_prev - r_j
        r_j_to_Rx = r_Rx - r_j
        norm_j_prev = np.linalg.norm(r_j_to_prev)
        norm_j_Rx = np.linalg.norm(r_j_to_Rx)
        if norm_j_prev == 0 or norm_j_Rx == 0:
            continue
        cos_psi_prime = np.dot(r_j_to_prev, r_j_to_Rx) / (norm_j_prev * norm_j_Rx)
        cos_psi_prime = np.clip(cos_psi_prime, -1.0, 1.0)
        if np.arccos(cos_psi_prime) > alpha_rad / 2:
            continue

        E_I += G_factor / (norm_rx_j**2)

    return E_I


# ===================== 完整星座干扰 =====================

def full_constellation_SINR(N_per_orbit, n_orbits, h, h_S, gamma_deg,
                             alpha_deg, band_params, Delta_beta_deg=0):
    """
    完整星座干扰模型 (Figure 10)
    包含: 同轨道 + 共面轨道 + 所有偏移轨道 + 共面偏移轨道 干扰

    N_per_orbit: 每轨道卫星数
    n_orbits: 轨道面数
    h: 星座1高度
    h_S: 星座2高度
    gamma_deg: 倾角
    alpha_deg: 波束宽度
    band_params: 频段参数
    """
    alpha_rad = np.radians(alpha_deg)
    lam = c / band_params['freq']
    P_Tx = band_params['P_Tx']
    G = 2.0 / (1.0 - np.cos(alpha_rad / 2))
    P_N = k_B * band_params['T_sys'] * band_params['bandwidth']

    # 信号功率 (轨道1中的链路)
    d1_sq = 2 * (R_E + h)**2 * (1 - np.cos(2 * np.pi / N_per_orbit))
    P_Rx = P_Tx * G**2 * lam**2 / ((4 * np.pi)**2 * d1_sq)

    # 1) 同轨道干扰 E[I1]
    N1 = single_orbit_N1(N_per_orbit, h, alpha_deg)
    E_I1 = 0.0
    if N1 >= 1:
        for i in range(2, N1 + 2):
            di_sq = 2 * (R_E + h)**2 * (1 - np.cos(2 * np.pi * i / N_per_orbit))
            E_I1 += lam**2 * P_Tx / (8 * np.pi**2 * (1 - np.cos(alpha_rad / 2))**2 * di_sq)

    # 2) 共面轨道干扰 (星座2, 同一轨道面RAAN)
    E_I_coplanar = coplanar_shifted_interference(
        N_per_orbit, N_per_orbit, h, h_S, gamma_deg, 0.0,
        alpha_deg, Delta_beta_deg, band_params)

    # 3) 偏移轨道干扰 (星座1, 其他轨道面)
    E_I_shifted = 0.0
    Omega_spacing = 360.0 / n_orbits
    for orbit_idx in range(1, n_orbits):
        Delta_Omega = orbit_idx * Omega_spacing
        E_I_shifted += shifted_orbit_interference(
            N_per_orbit, N_per_orbit, h, gamma_deg, Delta_Omega,
            alpha_deg, Delta_beta_deg, band_params)

    # 4) 共面偏移轨道干扰 (星座2, 其他轨道面)
    E_I_coplanar_shifted = 0.0
    for orbit_idx in range(1, n_orbits):
        Delta_Omega = orbit_idx * Omega_spacing
        E_I_coplanar_shifted += coplanar_shifted_interference(
            N_per_orbit, N_per_orbit, h, h_S, gamma_deg, Delta_Omega,
            alpha_deg, Delta_beta_deg, band_params)

    E_I_total = E_I1 + E_I_coplanar + E_I_shifted + E_I_coplanar_shifted
    total = E_I_total + P_N
    if total == 0:
        return np.inf
    return P_Rx / total


def full_constellation_capacity(N_per_orbit, n_orbits, h, h_S, gamma_deg,
                                 alpha_deg, band_params, Delta_beta_deg=0):
    """完整星座信道容量"""
    sinr = full_constellation_SINR(N_per_orbit, n_orbits, h, h_S, gamma_deg,
                                    alpha_deg, band_params, Delta_beta_deg)
    if sinr == np.inf:
        sinr = 1e15
    return band_params['bandwidth'] * np.log2(1 + sinr)


def SNR_only(N, h, band_params, alpha_deg):
    """
    无干扰SNR (用于对比: "No interference" 曲线)
    SNR = PRx / PN
    """
    alpha_rad = np.radians(alpha_deg)
    lam = c / band_params['freq']
    P_Tx = band_params['P_Tx']
    G = 2.0 / (1.0 - np.cos(alpha_rad / 2))
    P_N = k_B * band_params['T_sys'] * band_params['bandwidth']

    d1_sq = 2 * (R_E + h)**2 * (1 - np.cos(2 * np.pi / N))
    P_Rx = P_Tx * G**2 * lam**2 / ((4 * np.pi)**2 * d1_sq)

    return P_Rx / P_N
