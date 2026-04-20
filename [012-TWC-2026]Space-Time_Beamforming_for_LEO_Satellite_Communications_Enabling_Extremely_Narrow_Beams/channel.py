"""
信道模型模块
- UPA 空间导向向量
- 多径信道生成 (Shadowed-Rician)
- 空时信道向量构造
- 多普勒频移生成
"""

import numpy as np
from config import (
    Nx_default, Ny_default, wavelength, altitude, RE, c,
    alpha, L_paths, delta_tap, doppler_range,
    sr_b, sr_m, sr_Omega
)


def upa_steering_vector(theta, phi, Nx=Nx_default, Ny=Ny_default):
    """
    UPA 空间导向向量 a(theta, phi) ∈ C^{Nx*Ny x 1}

    theta: 仰角 (elevation), rad
    phi: 方位角 (azimuth), rad
    返回: Nx*Ny 维导向向量
    """
    n = np.arange(Nx)
    m = np.arange(Ny)
    # 沿 x 轴: e^{-j 2pi * d/lambda * n * sin(theta)*cos(phi)}
    # 沿 y 轴: e^{-j 2pi * d/lambda * m * sin(theta)*sin(phi)}
    # 假设半波长间距 d = lambda/2
    ax = np.exp(-1j * np.pi * n * np.sin(theta) * np.cos(phi))
    ay = np.exp(-1j * np.pi * m * np.sin(theta) * np.sin(phi))
    # Kronecker 积: a = ay ⊗ ax
    a = np.kron(ay, ax)
    return a  # 不归一化，保留阵列增益 ||a||^2 = N


def generate_shadowed_rician(n_samples):
    """
    生成 Shadowed-Rician 衰落系数
    使用复高斯近似方法

    参数 (Average Shadowing):
        b = 0.126, m = 5.21, Omega = 0.835

    返回: 信道增益 |h|^2 样本
    """
    # LoS 分量: Nakagami-m 分布的幅度
    # 生成 Gamma 分布样本再取平方根
    shape = sr_m
    scale = sr_Omega / sr_m
    los_amplitude_sq = np.random.gamma(shape, scale, n_samples)

    # 多径分量: 复高斯
    multipath_power = np.random.exponential(sr_b, n_samples)

    # 总功率
    total_power = los_amplitude_sq + multipath_power
    return total_power


def generate_path_loss(d):
    """
    自由空间路径损耗 D = (c/(4*pi*fc*d))^alpha

    d: 卫星到用户距离 (km)
    返回: 路径损耗因子（线性值）
    """
    fc_hz = c * 1e3 / wavelength  # 恢复载频
    D = (c * 1e3 / (4 * np.pi * fc_hz * d * 1e3)) ** alpha
    return D


def generate_multipath_angles(theta_los, phi_los, L=3):
    """
    生成多径角度
    LoS 路径 (i=1) 沿主方向，其他路径随机偏移 ±1°

    theta_los, phi_los: LoS 方向 (rad)
    返回: (theta_array, phi_array) 各 L 维
    """
    thetas = np.zeros(L)
    phis = np.zeros(L)
    thetas[0] = theta_los
    phis[0] = phi_los
    # 其余多径在 LoS 方向 ±1° 范围内随机偏移
    for i in range(1, L):
        thetas[i] = theta_los + np.deg2rad(np.random.uniform(-1, 1))
        phis[i] = phi_los + np.random.uniform(-np.pi, np.pi)
    return thetas, phis


def generate_doppler_shifts(K):
    """
    生成 K 个用户的相对多普勒频移
    均匀分布 [-50kHz, 50kHz]

    返回: K 维多普勒频移数组 (Hz)
    """
    return np.random.uniform(-doppler_range, doppler_range, K)


def generate_distance(altitude_km=altitude):
    """
    生成卫星到用户的距离 (km)
    基于 530 km 高度，用户仰角 40°-80° 范围
    参考论文: 卫星方位角 10°, 用户仰角约 40-60°
    """
    sat_r = RE + altitude_km  # 卫星轨道半径
    # 用户仰角范围 [40°, 80°]
    elevation_deg = np.random.uniform(40, 80)
    elevation = np.deg2rad(elevation_deg)
    # 根据仰角计算距离
    # d = RE * sin(nadir_angle + elevation) / sin(elevation) -- 简化
    # 直接用几何关系: d = sqrt((RE+h)^2 - RE^2*cos^2(el)) - RE*sin(el)
    d = np.sqrt(sat_r**2 - (RE * np.cos(elevation))**2) - RE * np.sin(elevation)
    return max(d, altitude_km)  # 确保不小于轨道高度


def build_spatial_channel(theta, phi, beta, Nx=Nx_default, Ny=Ny_default, L=L_paths):
    """
    构建空间信道向量 h ∈ C^{N x 1}

    h = sum_{i=1}^{L} beta_i * a(theta_i, phi_i)

    theta, phi: L 维角度数组
    beta: L 维复增益
    """
    N = Nx * Ny
    h = np.zeros(N, dtype=complex)
    for i in range(L):
        a = upa_steering_vector(theta[i], phi[i], Nx, Ny)
        h += beta[i] * a
    return h


def build_spatial_channel_single_path(theta, phi, beta, Nx=Nx_default, Ny=Ny_default):
    """单径空间信道（LoS only）"""
    a = upa_steering_vector(theta, phi, Nx, Ny)
    return beta * a


def build_temporal_steering(f_doppler, tau, M):
    """
    构建时间导向向量 b(f, tau) ∈ C^{M x 1}

    b = [1, e^{-j2pi*f*tau}, ..., e^{-j2pi*f*(M-1)*tau}]
    """
    m = np.arange(M)
    return np.exp(-1j * 2 * np.pi * f_doppler * tau * m)


def build_space_time_channel(spatial_channel, f_doppler, tau, M):
    """
    构建空时信道向量 h_{M,tau} ∈ C^{MN x 1}

    h_{M,tau} = b(f, tau) ⊗ h_spatial
    单径简化版本
    """
    b = build_temporal_steering(f_doppler, tau, M)
    return np.kron(b, spatial_channel)


def build_space_time_channel_multipath(thetas, phis, betas, f_doppler, tau, M,
                                        Nx=Nx_default, Ny=Ny_default):
    """
    构建多径空时信道向量（公式 9）

    h_{M,tau} = sum_{i=1}^{L} beta_i * c_{M,tau,i}
    c_{M,tau,i} = b(f, tau) ⊗ a(theta_i, phi_i)

    注意：多普勒对所有路径相同
    """
    N = Nx * Ny
    h = np.zeros(M * N, dtype=complex)
    b = build_temporal_steering(f_doppler, tau, M)
    for i in range(len(betas)):
        a = upa_steering_vector(thetas[i], phis[i], Nx, Ny)
        c = np.kron(b, a)
        h += betas[i] * c
    return h


def generate_channel_realization(K, M, tau_list, Nx=Nx_default, Ny=Ny_default,
                                  L=L_paths, network='partial'):
    """
    生成一次完整的信道实现

    K: 用户/卫星数
    M: 重复次数
    tau_list: K 个卫星的重传间隔列表
    network: 'partial' 或 'full'

    返回:
        h_st: dict, h_st[(l,k)] = 空时信道向量 C^{MN x 1}
        h_spatial: dict, h_spatial[(l,k)] = 空间信道向量 C^{N x 1}
        f_doppler: K 维多普勒频移
    """
    N = Nx * Ny

    # 生成多普勒频移
    f_doppler = generate_doppler_shifts(K)

    # 为每个 (user, satellite) 对生成信道
    h_st = {}
    h_spatial = {}

    for k in range(K):
        for l in range(K):
            # 卫星到用户距离
            d = generate_distance()

            # 路径损耗
            path_loss = generate_path_loss(d)

            # 多径角度
            # 参考仰角和方位角
            theta_los = np.random.uniform(np.deg2rad(30), np.deg2rad(80))
            phi_los = np.random.uniform(-np.pi, np.pi)
            thetas, phis = generate_multipath_angles(theta_los, phi_los, L)

            # 信道增益（含路径损耗、衰落、tap gain）
            betas = np.zeros(L, dtype=complex)
            fading_powers = generate_shadowed_rician(L)
            for i in range(L):
                tap_gain = delta_tap ** i
                D_i = generate_path_loss(d)  # 简化：所有路径同距离
                betas[i] = np.sqrt(tap_gain * path_loss * fading_powers[i]) * \
                           np.exp(1j * 2 * np.pi * np.random.random())

            # 空间信道
            h_sp = np.zeros(N, dtype=complex)
            for i in range(L):
                a = upa_steering_vector(thetas[i], phis[i], Nx, Ny)
                h_sp += betas[i] * a

            h_spatial[(l, k)] = h_sp

            # 空时信道
            tau_k = tau_list[k] if tau_list is not None else 0
            if M > 1 and tau_k > 0:
                b = build_temporal_steering(f_doppler[l], tau_k, M)
                h_st[(l, k)] = np.kron(b, h_sp)
            else:
                h_st[(l, k)] = h_sp.copy()

    return h_st, h_spatial, f_doppler
