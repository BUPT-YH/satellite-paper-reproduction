"""
Monte Carlo仿真模块
Coverage and Rate Analysis of Follower-Based LEO Satellite Networks

用于验证解析结果的正确性
- 中断概率MC仿真
- 平均速率MC仿真
"""

import numpy as np
from scipy.stats import gamma as gamma_dist
import config as cfg


def sample_leader_contact_angle(size=1):
    """
    采样Leader-user的接触角

    N_L个球面均匀分布点中最近一个的接触角
    CDF: F(θ) = 1 - ((1+cos(θ))/2)^(N_L)
    逆变换采样

    参数:
        size: 采样数量
    返回:
        接触角数组 (弧度)
    """
    u = np.random.uniform(0, 1, size)
    # θ = arccos(1 - 2 * (1 - u)^(1/N_L))  由 CDF 反推
    # F(θ) = 1 - ((1+cos(θ))/2)^N_L = u
    # ((1+cos(θ))/2)^N_L = 1-u
    # (1+cos(θ))/2 = (1-u)^(1/N_L)
    # cos(θ) = 2*(1-u)^(1/N_L) - 1
    cos_theta = 2.0 * (1.0 - u) ** (1.0 / cfg.N_L) - 1.0
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(cos_theta)


def sample_follower_offset(size=1):
    """
    采样Follower在球冠内的偏角

    均匀分布在球冠内: CDF: F(ψ) = (1-cos(ψ))/(1-cos(θ_cap))
    逆变换: cos(ψ) = 1 - u*(1-cos(θ_cap))

    参数:
        size: 采样数量
    返回:
        偏角数组 (弧度)
    """
    u = np.random.uniform(0, 1, size)
    cos_psi = 1.0 - u * (1.0 - np.cos(cfg.theta_cap))
    cos_psi = np.clip(cos_psi, -1, 1)
    return np.arccos(cos_psi)


def sample_follower_azimuth(size=1):
    """
    采样Follower的方位角 (均匀分布0~2π)

    参数:
        size: 采样数量
    返回:
        方位角数组 (弧度)
    """
    return np.random.uniform(0, 2 * np.pi, size)


def sample_channel_gain(size=1):
    """
    采样Shadowed-Rician信道功率增益 (Gamma近似)

    参数:
        size: 采样数量
    返回:
        信道增益数组
    """
    return gamma_dist.rvs(a=cfg.m1_gamma, scale=cfg.m2_gamma, size=size)


def mc_outage_leader(gamma_th_dB, n_samples=None):
    """
    Leader中断概率的Monte Carlo仿真

    参数:
        gamma_th_dB: SNR阈值 (dB)
        n_samples: 仿真次数
    返回:
        中断概率估计值
    """
    if n_samples is None:
        n_samples = cfg.MC_samples

    gamma_th = 10 ** (gamma_th_dB / 10)

    # 采样接触角
    theta = sample_leader_contact_angle(n_samples)

    # 限制在theta_max范围内
    valid = theta <= cfg.theta_max
    theta = theta[valid]
    n_valid = len(theta)

    if n_valid == 0:
        return 1.0

    # 计算距离
    r = np.sqrt(cfg.R_sat**2 + cfg.R_earth**2 - 2 * cfg.R_sat * cfg.R_earth * np.cos(theta))

    # 采样信道增益
    w = sample_channel_gain(n_valid)

    # 计算SNR
    snr = cfg.xi_LU * w / r**2

    # 统计中断
    outage = np.sum(snr < gamma_th)

    # 注意: 超出theta_max的也计为中断（无法通信）
    total_outage = outage + (n_samples - n_valid)

    return total_outage / n_samples


def mc_outage_cluster(gamma_th_dB, N_F, n_samples=None):
    """
    Cluster中断概率的Monte Carlo仿真

    Cluster中断 = 所有Follower都中断 AND Leader也中断

    参数:
        gamma_th_dB: SNR阈值 (dB)
        N_F: Follower数量
        n_samples: 仿真次数
    返回:
        中断概率估计值
    """
    if n_samples is None:
        n_samples = cfg.MC_samples

    if N_F == 0:
        return mc_outage_leader(gamma_th_dB, n_samples)

    gamma_th = 10 ** (gamma_th_dB / 10)

    # 采样Leader接触角
    theta_LU = sample_leader_contact_angle(n_samples)
    valid = theta_LU <= cfg.theta_max

    total_outage = n_samples - np.sum(valid)  # 超出范围的直接中断

    # 对有效样本计算
    theta_valid = theta_LU[valid]
    n_valid = len(theta_valid)

    if n_valid == 0:
        return 1.0

    # Leader距离和SNR
    r_LU = np.sqrt(cfg.R_sat**2 + cfg.R_earth**2 - 2 * cfg.R_sat * cfg.R_earth * np.cos(theta_valid))
    w_LU = sample_channel_gain(n_valid)
    snr_LU = cfg.xi_LU * w_LU / r_LU**2
    leader_outage = snr_LU < gamma_th

    # Follower中断
    # 每个follower独立采样偏角和方位角
    all_follower_outage = np.ones(n_valid, dtype=bool)  # 默认全部中断

    for _ in range(N_F):
        psi = sample_follower_offset(n_valid)
        phi = sample_follower_azimuth(n_valid)

        # Follower到用户距离
        cos_sep = np.cos(theta_valid) * np.cos(psi) + np.sin(theta_valid) * np.sin(psi) * np.cos(phi)
        cos_sep = np.clip(cos_sep, -1, 1)
        angular_sep = np.arccos(cos_sep)
        r_FU = np.sqrt(cfg.R_sat**2 + cfg.R_earth**2 - 2 * cfg.R_sat * cfg.R_earth * np.cos(angular_sep))

        w_FU = sample_channel_gain(n_valid)
        snr_FU = cfg.xi_FU * w_FU / r_FU**2

        this_follower_ok = snr_FU >= gamma_th
        all_follower_outage = all_follower_outage & ~this_follower_ok

    # Cluster中断 = leader中断 AND 所有follower都中断
    cluster_outage = leader_outage & all_follower_outage
    total_outage += np.sum(cluster_outage)

    return total_outage / n_samples


def mc_avg_rate_leader(n_samples=None):
    """
    Leader平均速率的Monte Carlo仿真

    参数:
        n_samples: 仿真次数
    返回:
        平均速率 (bps)
    """
    if n_samples is None:
        n_samples = cfg.MC_samples

    # 采样接触角
    theta = sample_leader_contact_angle(n_samples)
    valid = theta <= cfg.theta_max
    theta = theta[valid]
    n_valid = len(theta)

    if n_valid == 0:
        return 0.0

    # 计算距离
    r = np.sqrt(cfg.R_sat**2 + cfg.R_earth**2 - 2 * cfg.R_sat * cfg.R_earth * np.cos(theta))

    # 采样信道增益
    w = sample_channel_gain(n_valid)

    # 计算SNR和速率
    snr = cfg.xi_LU * w / r**2
    rates = cfg.B_LU * np.log2(1.0 + snr)

    # 超出theta_max的速率为0
    total_rate = np.sum(rates)

    return total_rate / n_samples


def mc_avg_rate_cluster(N_F, n_samples=None):
    """
    Cluster平均速率的Monte Carlo仿真

    R_Cluster = R_LU + Σ_i min{R_LF_i, R_FU_i}

    参数:
        N_F: Follower数量
        n_samples: 仿真次数
    返回:
        平均速率 (bps)
    """
    if n_samples is None:
        n_samples = cfg.MC_samples

    # Leader速率
    R_LU_mc = mc_avg_rate_leader(n_samples)

    if N_F == 0:
        return R_LU_mc

    # 采样Leader接触角
    theta_LU = sample_leader_contact_angle(n_samples)
    valid = theta_LU <= cfg.theta_max
    theta_valid = theta_LU[valid]
    n_valid = len(theta_valid)

    if n_valid == 0:
        return 0.0

    follower_rate_total = 0.0

    for _ in range(N_F):
        # 采样follower偏角和方位角
        psi = sample_follower_offset(n_valid)
        phi = sample_follower_azimuth(n_valid)

        # Leader到Follower距离
        r_LF = np.sqrt(cfg.R_sat**2 + cfg.R_earth**2 - 2 * cfg.R_sat * cfg.R_earth * np.cos(psi))

        # Follower到用户距离
        cos_sep = np.cos(theta_valid) * np.cos(psi) + np.sin(theta_valid) * np.sin(psi) * np.cos(phi)
        cos_sep = np.clip(cos_sep, -1, 1)
        angular_sep = np.arccos(cos_sep)
        r_FU = np.sqrt(cfg.R_sat**2 + cfg.R_earth**2 - 2 * cfg.R_sat * cfg.R_earth * np.cos(angular_sep))

        # 采样信道增益
        w_LF = sample_channel_gain(n_valid)
        w_FU = sample_channel_gain(n_valid)

        # 各链路速率
        snr_LF = cfg.xi_LF * w_LF / r_LF**2
        rate_LF = cfg.B_LF * np.log2(1.0 + snr_LF)

        snr_FU = cfg.xi_FU * w_FU / r_FU**2
        rate_FU = cfg.B_FU * np.log2(1.0 + snr_FU)

        # 端到端速率取瓶颈
        rate_follower_i = np.minimum(rate_LF, rate_FU)
        follower_rate_total += np.sum(rate_follower_i)

    return R_LU_mc + follower_rate_total / n_samples


def mc_avg_rate_non_follower(rho_LU_total_dBW, n_samples=None):
    """
    Non-follower方案的平均速率MC仿真

    参数:
        rho_LU_total_dBW: 总发射功率 (dBW)
        n_samples: 仿真次数
    返回:
        平均速率 (bps)
    """
    if n_samples is None:
        n_samples = cfg.MC_samples

    rho_total = 10 ** (rho_LU_total_dBW / 10)
    xi_lu_new = rho_total * cfg.G * cfg.zeta_U * (cfg.nu / (4 * np.pi))**2 / cfg.sigma_U_sq / 1e6

    theta = sample_leader_contact_angle(n_samples)
    valid = theta <= cfg.theta_max
    theta = theta[valid]
    n_valid = len(theta)

    if n_valid == 0:
        return 0.0

    r = np.sqrt(cfg.R_sat**2 + cfg.R_earth**2 - 2 * cfg.R_sat * cfg.R_earth * np.cos(theta))
    w = sample_channel_gain(n_valid)
    snr = xi_lu_new * w / r**2
    rates = cfg.B_LU * np.log2(1.0 + snr)

    return np.sum(rates) / n_samples


if __name__ == "__main__":
    print("Testing Monte Carlo module...")

    theta_samples = sample_leader_contact_angle(10000)
    print(f"Mean contact angle: {np.rad2deg(np.mean(theta_samples)):.4f} deg")
    print(f"Range: [{np.rad2deg(np.min(theta_samples)):.4f}, {np.rad2deg(np.max(theta_samples)):.4f}] deg")

    print(f"\nMC Leader outage (gamma_th = {cfg.gamma_th_dB} dB)...")
    p_out_mc = mc_outage_leader(cfg.gamma_th_dB, 100000)
    print(f"P_out_LU (MC) = {p_out_mc:.6e}")
