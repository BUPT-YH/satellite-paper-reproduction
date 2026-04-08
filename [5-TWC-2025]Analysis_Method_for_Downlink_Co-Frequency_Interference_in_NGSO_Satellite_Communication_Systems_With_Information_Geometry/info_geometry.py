"""
核心算法模块 - 信息几何干扰分析方法 (高效版)
关键优化:
1. DTC_AIRM 用特征值分解代替 sqrtm + logm
2. AIRM 中心矩阵迭代用批量特征值计算
3. 协方差矩阵用小维度 (10x10) 而非大维度匹配采样点数
"""

import numpy as np
from scipy.linalg import sqrtm, logm, expm
from config import *


# ==================== 坐标转换与几何计算 ====================

def latlon_to_cartesian(lat_deg, lon_deg, h):
    """大地坐标 -> ECEF"""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    e = e_earth
    N = a_earth / np.sqrt(1 - e**2 * np.sin(lat)**2)
    x = -(N + h) * np.cos(lat) * np.cos(lon)
    y = -(N + h) * np.cos(lat) * np.sin(lon)
    z = -((N * (1 - e**2)) + h) * np.sin(lat)
    return np.array([x, y, z])


def calc_off_axis_angle(lat_es, lon_es, h_es, lat_sat1, lon_sat1, h_sat1, lat_sat2, lon_sat2, h_sat2):
    """计算偏轴角 psi (度)"""
    pos_es = latlon_to_cartesian(lat_es, lon_es, h_es)
    pos_sat1 = latlon_to_cartesian(lat_sat1, lon_sat1, h_sat1)
    pos_sat2 = latlon_to_cartesian(lat_sat2, lon_sat2, h_sat2)
    vec_ESI = pos_sat1 - pos_es
    vec_ESII = pos_sat2 - pos_es
    cos_psi = np.dot(vec_ESII, vec_ESI) / (np.linalg.norm(vec_ESII) * np.linalg.norm(vec_ESI))
    return np.degrees(np.arccos(np.clip(cos_psi, -1, 1)))


def calc_link_distance(lat1, lon1, h1, lat2, lon2, h2):
    """链路距离 (km)"""
    pos1 = latlon_to_cartesian(lat1, lon1, h1)
    pos2 = latlon_to_cartesian(lat2, lon2, h2)
    return np.linalg.norm(pos2 - pos1) / 1e3


def calc_elevation_angle(lat_es, lon_es, h_es, lat_sat, lon_sat, h_sat):
    """仰角 (度)"""
    pos_es = latlon_to_cartesian(lat_es, lon_es, h_es)
    pos_sat = latlon_to_cartesian(lat_sat, lon_sat, h_sat)
    vec = pos_sat - pos_es
    dz = abs(vec[2])
    r_ground = np.sqrt(vec[0]**2 + vec[1]**2)
    return np.degrees(np.arctan2(dz, r_ground))


# ==================== 天线增益 ====================

def antenna_gain_s1528(G_max_dBi, psi_deg, hpbw_deg):
    """ITU-R S.1528 卫星天线方向图"""
    psi = np.abs(psi_deg)
    if psi <= 1.6 * hpbw_deg:
        return G_max_dBi - 12 * (psi / hpbw_deg)**2
    elif psi <= 3.2 * hpbw_deg:
        return G_max_dBi - 15
    else:
        return -10.0


def antenna_gain_s465(G_max_dBi, psi_deg, D_ant):
    """ITU-R S.465-6 地球站天线方向图"""
    psi = np.abs(psi_deg)
    if psi < 1.0:
        return G_max_dBi - 12 * psi**2
    elif psi < 48.0:
        return 32 - 25 * np.log10(psi)
    else:
        return -10.0


# ==================== 链路损耗 ====================

def free_space_loss_dB(freq_hz, dist_km):
    return 32.4 + 20 * np.log10(freq_hz / 1e6) + 20 * np.log10(dist_km)


def rain_attenuation_dB(freq_hz, dist_km, outage_rate=0.1):
    f_GHz = freq_hz / 1e9
    gamma_R = 0.01 * f_GHz**1.5
    L_E = min(dist_km, 5.0)
    return gamma_R * L_E * outage_rate


def cloud_attenuation_dB(freq_hz, elev_deg):
    f_GHz = freq_hz / 1e9
    elev = max(elev_deg, 5.0)
    L_red = liquid_water_density * cloud_thickness
    Kl = 0.001 * f_GHz**2
    return L_red * Kl / np.sin(np.radians(elev))


def calc_received_power_dBW(Pt_dBW, Gt_dBi, Gr_dBi, freq_hz, dist_km,
                             lat_es=None, lon_es=None, h_es=None,
                             lat_sat=None, lon_sat=None, h_sat=None):
    L_fs = free_space_loss_dB(freq_hz, dist_km)
    L_rain = rain_attenuation_dB(freq_hz, dist_km)
    L_cloud = 0
    if lat_es is not None:
        elev = calc_elevation_angle(lat_es, lon_es, h_es, lat_sat, lon_sat, h_sat)
        L_cloud = cloud_attenuation_dB(freq_hz, elev)
    return Pt_dBW + Gt_dBi + Gr_dBi - L_fs - L_rain - L_cloud


def calc_signal_amplitude(Pr_dBW):
    return np.sqrt(2 * 10 ** (Pr_dBW / 10))


# ==================== 协方差矩阵 (小维度高效版) ====================

def make_covariance_matrix(total_power, dim):
    """构建 dim x dim 的协方差矩阵
    对角元素为主, 加小随机扰动模拟实际采样噪声
    """
    R = np.eye(dim) * total_power
    noise = np.random.normal(0, total_power * 0.02, (dim, dim))
    noise = (noise + noise.T) / 2
    R += noise
    # 确保正定
    eigvals = np.linalg.eigvalsh(R)
    if np.min(eigvals) <= 0:
        R += np.eye(dim) * (abs(np.min(eigvals)) + 1e-10)
    return R


def generate_cov_free(A0, sigma0, dim):
    """无干扰场景协方差矩阵"""
    power = A0**2 + sigma0**2
    return make_covariance_matrix(power, dim)


def generate_cov_interf(A0, sigma0, A_interf, sigma_interf, dim):
    """有干扰场景协方差矩阵"""
    power = A0**2 + sigma0**2
    for Ai, si in zip(A_interf, sigma_interf):
        power += Ai**2 + si**2
    return make_covariance_matrix(power, dim)


# ==================== DTC 计算 (高效版) ====================

def calc_DTC_AIRM_fast(R_center, R_test):
    """AIRM DTC - 用特征值分解避免 sqrtm
    D = sqrt(sum(ln(lambda_i)^2)) 其中 lambda_i 是 R_test * inv(R_center) 的特征值
    (因为 AB 和 BA 有相同的非零特征值)
    """
    inner = np.linalg.solve(R_center, R_test)  # = inv(R_center) @ R_test
    eigvals = np.linalg.eigvalsh(inner)
    eigvals = np.maximum(eigvals.real, 1e-15)
    return np.sqrt(np.sum(np.log(eigvals)**2))


def calc_DTC_SKLD_fast(R_center, R_test):
    """SKLD DTC - 纯矩阵运算, 无需 sqrtm
    D = 0.5 * tr(inv(R_test)*R_center + inv(R_center)*R_test - 2I)
    """
    M = R_center.shape[0]
    R_test_inv = np.linalg.inv(R_test)
    R_center_inv = np.linalg.inv(R_center)
    return max(0.5 * np.trace(R_test_inv @ R_center + R_center_inv @ R_test - 2 * np.eye(M)), 0)


# ==================== 流形中心矩阵 ====================

def center_matrix_AIRM(cov_matrices, max_iter=30, tau=0.1):
    """AIRM 流形中心矩阵 - 用高效特征值迭代"""
    Q = len(cov_matrices)
    dim = cov_matrices[0].shape[0]
    R_center = np.mean(cov_matrices, axis=0).copy()

    for _ in range(max_iter):
        log_sum = np.zeros((dim, dim))
        R_center_inv = np.linalg.inv(R_center)

        for Rk in cov_matrices:
            inner = R_center_inv @ Rk
            eigvals, eigvecs = np.linalg.eigh(inner)
            eigvals = np.maximum(eigvals, 1e-15)
            log_inner = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T
            log_sum += log_inner

        log_mean = log_sum / Q
        # 更新: R_center = R_center^(1/2) @ exp(tau * log_mean) @ R_center^(1/2)
        eigvals_c, eigvecs_c = np.linalg.eigh(R_center)
        eigvals_c = np.maximum(eigvals_c, 1e-15)
        R_half = eigvecs_c @ np.diag(np.sqrt(eigvals_c)) @ eigvecs_c.T

        exp_term = eigvecs_c @ np.diag(np.exp(tau * np.log(eigvals_c))) @ eigvecs_c.T
        R_center = R_half @ exp_term @ R_half

        # 修正: 使用正确的迭代公式
        # R_new = R^{1/2} exp(tau * log_mean) R^{1/2}
        eigvals_e, eigvecs_e = np.linalg.eigh(log_mean)
        exp_log = eigvecs_e @ np.diag(np.exp(tau * eigvals_e)) @ eigvecs_e.T
        R_center = R_half @ exp_log @ R_half
        R_center = (R_center + R_center.T) / 2

    return R_center


def center_matrix_SKLD(cov_matrices):
    """SKLD 流形中心矩阵"""
    sum_R = np.zeros_like(cov_matrices[0])
    sum_R_inv = np.zeros_like(cov_matrices[0])

    for Rk in cov_matrices:
        sum_R += Rk
        sum_R_inv += np.linalg.inv(Rk)

    # sum_R_inv^(-1/2)
    eigvals, eigvecs = np.linalg.eigh(sum_R_inv)
    eigvals = np.maximum(eigvals, 1e-15)
    sum_R_inv_neg_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    sum_R_inv_half = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    inner = sum_R_inv_half @ sum_R @ sum_R_inv_half
    eigvals_i, eigvecs_i = np.linalg.eigh(inner)
    eigvals_i = np.maximum(eigvals_i, 1e-15)
    inner_sqrt = eigvecs_i @ np.diag(np.sqrt(eigvals_i)) @ eigvecs_i.T

    R_center = sum_R_inv_neg_half @ inner_sqrt @ sum_R_inv_neg_half
    return (R_center + R_center.T) / 2


# ==================== 阈值与 JR-DTC ====================

def calc_threshold(dtc_values, Pf=0.03):
    Q = len(dtc_values)
    sorted_dtc = np.sort(dtc_values)
    idx = min(max(int(Q * (1 - Pf)) - 1, 0), Q - 1)
    return sorted_dtc[idx]


def calc_JR_DTC(dtc_values, Pf=0.03):
    Q = len(dtc_values)
    sorted_dtc = np.sort(dtc_values)
    idx_l = min(max(int(Q * Pf) - 1, 0), Q - 1)
    idx_u = min(max(int(Q * (1 - Pf)) - 1, 0), Q - 1)
    return sorted_dtc[idx_l], sorted_dtc[idx_u]


# ==================== 势函数 ====================

def potential_function(R_cov):
    N = R_cov.shape[0]
    sign, logdet = np.linalg.slogdet(R_cov)
    return 0.5 * logdet + (N / 2) * np.log(2 * np.pi)


# ==================== 辅助 ====================

def get_signal_params(case_idx, A0, sigma0, A_interf, sigma_interf):
    if case_idx == 0:
        return A0, sigma0, [], []
    elif case_idx == 1:
        return A0, sigma0, [A_interf[0]], [sigma_interf[0]]
    elif case_idx == 2:
        return A0, sigma0, [A_interf[1]], [sigma_interf[1]]
    elif case_idx == 3:
        return A0, sigma0, A_interf, sigma_interf
    else:
        raise ValueError(f"Unknown case: {case_idx}")
