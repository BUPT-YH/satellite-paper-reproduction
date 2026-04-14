"""
相控阵天线 (UPA) 方向图计算
64×64 星载天线 + 8×8/16×16/32×32 用户天线
"""

import numpy as np
from config import WAVELENGTH, C_LIGHT, ANTENNA_SPACING


def upa_gain_db(direction_eval, direction_steer, nx, ny, east, north):
    """
    计算 UPA 天线增益 (dBi)

    天线面朝向天顶 (地面用户) 或天底 (卫星)
    电子波束赋形指向 direction_steer
    计算 direction_eval 方向的增益

    参数:
        direction_eval: 评估方向的单位向量, shape (n, 3) 或 (3,)
        direction_steer: 波束赋形方向的单位向量, shape (3,)
        nx, ny: 阵列维度
        east, north: 局部坐标系东和北方向向量

    返回:
        gain_dbi: 增益 (dBi), shape (n,)
    """
    if direction_eval.ndim == 1:
        direction_eval = direction_eval[np.newaxis, :]

    n_eval = direction_eval.shape[0]

    # 将方向向量投影到阵列面 (局部水平面)
    # u = sin(θ)·cos(φ) → 投影到 East 方向
    # v = sin(θ)·sin(φ) → 投影到 North 方向
    # 对于天顶方向的阵列: sin(θ) 是偏离天顶角的正弦
    u_eval = np.dot(direction_eval, east)
    v_eval = np.dot(direction_eval, north)

    u_steer = np.dot(direction_steer, east)
    v_steer = np.dot(direction_steer, north)

    # 归一化空间频率 (以波长为单位, d = λ/2)
    # ψ = π · Δu, 其中 Δu = u_eval - u_steer
    du = u_eval - u_steer
    dv = v_eval - v_steer

    psi_x = np.pi * du  # k·d·Δu, d/λ = 0.5
    psi_y = np.pi * dv

    # 阵列因子 (统一平面阵列)
    # |AF_x|² = sin²(Nx·ψx/2) / sin²(ψx/2)
    # 需要 handle ψx → 0 的情况

    gain_x = _array_factor_squared(psi_x, nx)
    gain_y = _array_factor_squared(psi_y, ny)

    # 归一化增益 (相对于最大值 N²)
    gain_linear = (gain_x * gain_y) / (nx * ny) ** 2

    # 转换为 dBi
    gain_dbi = 10 * np.log10(np.maximum(gain_linear, 1e-30))

    # 最大增益 = 10·log10(Nx·Ny)
    max_gain = 10 * np.log10(nx * ny)

    return gain_dbi + max_gain


def _array_factor_squared(psi, N):
    """
    计算 |AF|² = sin²(N·ψ/2) / sin²(ψ/2)
    当 ψ→0 时极限为 N²
    """
    result = np.zeros_like(psi)

    # 小角度情况: 使用泰勒展开避免数值问题
    small = np.abs(psi) < 1e-10
    large = ~small

    if np.any(small):
        result[small] = N ** 2

    if np.any(large):
        psi_l = psi[large]
        num = np.sin(N * psi_l / 2) ** 2
        den = np.sin(psi_l / 2) ** 2
        result[large] = num / np.maximum(den, 1e-30)

    return result


def max_upa_gain_dbi(nx, ny):
    """UPA 最大增益 (dBi)"""
    return 10 * np.log10(nx * ny)


def get_eirp_density(radius, is_primary, max_alt_primary=570.0, max_alt_secondary=630.0):
    """
    根据 FCC 填报的 EIRP 频谱密度，按高度调整

    高度最高的卫星使用最大功率，低轨道适当降低
    """
    from config import PRIMARY_EIRP, SECONDARY_EIRP

    if is_primary:
        max_alt = max_alt_primary
        eirp = PRIMARY_EIRP
    else:
        max_alt = max_alt_secondary
        eirp = SECONDARY_EIRP

    return eirp


def plot_beam_pattern(nx, ny, angles_deg=None):
    """
    生成 UPA 波束方向图数据 (方位截断)

    返回:
        angles: 角度数组 (度)
        gain_normalized: 归一化增益 (dB)
    """
    if angles_deg is None:
        angles_deg = np.linspace(-90, 90, 3601)

    angles_rad = np.deg2rad(angles_deg)

    # 波束指向正上方 (天顶), 评估偏离角
    # 偏离角 θ 对应 sin(θ) 在阵列面上的投影
    psi = np.pi * np.sin(angles_rad)  # k·d·sin(θ)

    gain_x = _array_factor_squared(psi, nx)
    gain_y = _array_factor_squared(psi, ny)

    # 归一化
    gain_total = (gain_x * gain_y) / (nx * ny) ** 2
    gain_db = 10 * np.log10(np.maximum(gain_total, 1e-10))

    return angles_deg, gain_db
