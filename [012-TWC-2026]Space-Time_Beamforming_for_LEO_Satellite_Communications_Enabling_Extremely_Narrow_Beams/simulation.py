"""
仿真模块（重构版）
核心改进：基于实际几何生成信道，用户共址导致 AoA 相近，体现空时波束赋形价值
- 卫星在不同轨道位置
- 用户在地面上聚集（相同波束覆盖区内）
- 从同一卫星看不同用户 AoA 几乎相同
- 不同卫星到不同用户的 Doppler 不同
"""

import numpy as np
from config import (
    Nx_default, Ny_default, N_default, wavelength, altitude, RE, c,
    Ts, sigma2, L_paths, delta_tap, doppler_range, n_channels,
    P_range_dBm, K_partial, K_full, M_full,
    delay_error_max, dBm_to_W
)
from channel import (
    upa_steering_vector, generate_shadowed_rician, generate_path_loss,
    build_temporal_steering
)
from beamforming import (
    mrt_beamforming, zf_beamforming, slnr_beamforming,
    compute_st_zf, compute_st_slnr
)


def setup_geometry(K, network='partial'):
    """
    建立卫星-用户几何关系

    关键：用户在地面上聚集（共址），从同一卫星看不同用户 AoA 几乎相同
    不同卫星到用户的 Doppler 不同（卫星运动方向不同）
    """
    # 卫星位置：沿轨道弧分布，间隔约 5° 方位角
    sat_azimuths = np.deg2rad(np.linspace(5, 5 + (K-1) * 5, K))
    sat_zeniths = np.deg2rad(np.random.uniform(40, 60, K))

    # 用户位置：全部在地面同一区域附近（共址，±1km）
    # 从卫星角度看，用户间角度差极小
    user_center_theta = np.deg2rad(50)  # 参考仰角
    user_center_phi = np.deg2rad(0)     # 参考方位角

    return sat_azimuths, sat_zeniths, user_center_theta, user_center_phi


def generate_channel_geom(K, M, tau_list, f_doppler, Nx=Nx_default, Ny=Ny_default,
                           L=L_paths, network='partial', add_delay_error=False):
    """
    基于几何关系生成一次信道实现

    关键设计：
    - 从同一卫星 k 看不同用户 l, l' 的 AoA 几乎相同（用户共址）
    - f_doppler 由外部传入（确保与 tau 计算一致）
    """
    N = Nx * Ny

    sat_azimuths, sat_zeniths, user_theta, user_phi = setup_geometry(K, network)

    h_st = {}
    h_spatial = {}

    for k in range(K):
        # 卫星 k 的参考 AoA（从卫星 k 看所有用户的大致方向）
        ref_theta = np.pi / 2 - sat_zeniths[k]  # 转换为卫星坐标系中的角度
        ref_phi = sat_azimuths[k]

        # 卫星到用户的距离
        elevation_rad = np.pi / 2 - sat_zeniths[k]
        sat_r = RE + altitude
        d = np.sqrt(sat_r**2 - (RE * np.cos(elevation_rad))**2) - RE * np.sin(elevation_rad)
        d = max(d, altitude)

        # 路径损耗
        path_loss = generate_path_loss(d)

        for l in range(K):
            # 从卫星 k 看用户 l 的 AoA
            # 用户共址：不同用户 AoA 差异极小（< 0.5°）
            theta_offset = np.random.uniform(-0.003, 0.003)  # ~0.17°
            phi_offset = np.random.uniform(-0.003, 0.003)

            # LoS 方向
            thetas = np.zeros(L)
            phis = np.zeros(L)
            thetas[0] = ref_theta + theta_offset
            phis[0] = ref_phi + phi_offset

            # 多径分量偏离 LoS ±1°
            for i in range(1, L):
                thetas[i] = thetas[0] + np.deg2rad(np.random.uniform(-1, 1))
                phis[i] = phis[0] + np.random.uniform(-np.pi, np.pi)

            # 信道增益
            betas = np.zeros(L, dtype=complex)
            fading_powers = generate_shadowed_rician(L)
            for i in range(L):
                tap_gain = delta_tap ** i
                betas[i] = np.sqrt(tap_gain * path_loss * fading_powers[i]) * \
                           np.exp(1j * 2 * np.pi * np.random.random())

            # 延迟误差（不完美 CSIT）
            if add_delay_error:
                delay_err = np.random.uniform(0, delay_error_max)
                betas *= np.exp(-1j * 2 * np.pi * f_doppler[l] * delay_err)

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

    return h_st, h_spatial


# ============================================================
# 部分连接网络仿真
# ============================================================

def simulate_partial_network(K, P_dBm, M=2, n_ch=n_channels,
                              Nx=Nx_default, Ny=Ny_default, add_delay_error=False):
    """仿真部分连接网络"""
    P = dBm_to_W(P_dBm)
    N = Nx * Ny

    se_mrt = np.zeros(n_ch)
    se_zf = np.zeros(n_ch)
    se_slnr = np.zeros(n_ch)
    se_tdma = np.zeros(n_ch)
    se_stzf = np.zeros(n_ch)

    for ch in range(n_ch):
        f_doppler = np.zeros(K)
        # 生成不同用户的多普勒
        for k in range(K):
            f_doppler[k] = np.random.uniform(-doppler_range, doppler_range)

        # 计算最优 tau（精确值，不量化到 Ts）
        tau_list = np.zeros(K)
        for k in range(K):
            k_prime = (k + 1) % K  # 卫星 k 不应泄漏到的用户
            delta_f = np.abs(f_doppler[k] - f_doppler[k_prime])
            if delta_f > 10:
                tau_list[k] = 1.0 / (2 * delta_f)
            else:
                tau_list[k] = 1e-3  # 默认值

        h_st, h_spatial = generate_channel_geom(
            K, M, tau_list, f_doppler, Nx, Ny, network='partial', add_delay_error=add_delay_error
        )

        # ===== MRT =====
        f_dict = {}
        for k in range(K):
            f_dict[k] = mrt_beamforming(h_spatial[(k, k)])

        sum_se = 0
        for l in range(K):
            signal = np.abs(h_spatial[(l, l)].conj().T @ f_dict[l]) ** 2 * P
            interf = sum(
                np.abs(h_spatial.get((l, q), np.zeros(N)).conj().T @ f_dict[q]) ** 2 * P
                for q in range(K) if q != l
            )
            sinr = signal / (interf + sigma2)
            sum_se += np.log2(1 + sinr)
        se_mrt[ch] = sum_se

        # ===== ZF =====
        H_all = np.column_stack([h_spatial[(k, k)] for k in range(K)])
        for k in range(K):
            f_dict[k] = zf_beamforming(H_all, k)

        sum_se = 0
        for l in range(K):
            signal = np.abs(h_spatial[(l, l)].conj().T @ f_dict[l]) ** 2 * P
            interf = sum(
                np.abs(h_spatial.get((l, q), np.zeros(N)).conj().T @ f_dict[q]) ** 2 * P
                for q in range(K) if q != l
            )
            sinr = signal / (interf + sigma2)
            sum_se += np.log2(1 + sinr)
        se_zf[ch] = sum_se

        # ===== SLNR (MMSE) =====
        for k in range(K):
            h_d = h_spatial[(k, k)]
            H_interf = np.column_stack([h_spatial[(l, k)] for l in range(K) if l != k])
            f_dict[k] = slnr_beamforming(h_d, H_interf, sigma2, P)

        sum_se = 0
        for l in range(K):
            signal = np.abs(h_spatial[(l, l)].conj().T @ f_dict[l]) ** 2 * P
            interf = sum(
                np.abs(h_spatial.get((l, q), np.zeros(N)).conj().T @ f_dict[q]) ** 2 * P
                for q in range(K) if q != l
            )
            sinr = signal / (interf + sigma2)
            sum_se += np.log2(1 + sinr)
        se_slnr[ch] = sum_se

        # ===== TDMA =====
        K_odd = list(range(0, K, 2))
        K_even = list(range(1, K, 2))
        sum_se = 0
        for group in [K_odd, K_even]:
            for k in group:
                h = h_spatial[(k, k)]
                f = mrt_beamforming(h)
                signal = np.abs(h.conj().T @ f) ** 2 * P
                sinr = signal / sigma2
                sum_se += 0.5 * np.log2(1 + sinr)
        se_tdma[ch] = sum_se

        # ===== ST-ZF =====
        sum_se, _ = compute_st_zf(h_st, K, M, P, sigma2)
        se_stzf[ch] = sum_se

    return {
        'MRT': np.mean(se_mrt),
        'ZF': np.mean(se_zf),
        'SLNR': np.mean(se_slnr),
        'TDMA': np.mean(se_tdma),
        'ST-ZF': np.mean(se_stzf),
    }


# ============================================================
# 全连接网络仿真
# ============================================================

def simulate_full_network(K, P_dBm, M=M_full, n_ch=n_channels,
                           Nx=Nx_default, Ny=Ny_default, add_delay_error=False):
    """仿真全连接网络"""
    P = dBm_to_W(P_dBm)
    N = Nx * Ny

    se_mrt = np.zeros(n_ch)
    se_slnr = np.zeros(n_ch)
    se_tdma = np.zeros(n_ch)
    se_stslnr = np.zeros(n_ch)

    for ch in range(n_ch):
        f_doppler = np.random.uniform(-doppler_range, doppler_range, K)

        tau_list = np.zeros(K)
        for k in range(K):
            if M > 1:
                delta_f_min = min(
                    np.abs(f_doppler[k] - f_doppler[l]) for l in range(K) if l != k
                )
                if delta_f_min > 10:
                    tau_list[k] = 1.0 / (2 * delta_f_min)
                else:
                    tau_list[k] = 1e-3

        h_st, h_spatial = generate_channel_geom(
            K, M, tau_list, f_doppler, Nx, Ny, network='full', add_delay_error=add_delay_error
        )

        # ===== MRT =====
        f_dict = {}
        for k in range(K):
            f_dict[k] = mrt_beamforming(h_spatial[(k, k)])

        sum_se = 0
        for l in range(K):
            signal = np.abs(h_spatial[(l, l)].conj().T @ f_dict[l]) ** 2 * P
            interf = sum(
                np.abs(h_spatial.get((l, q), np.zeros(N)).conj().T @ f_dict[q]) ** 2 * P
                for q in range(K) if q != l
            )
            sinr = signal / (interf + sigma2)
            sum_se += np.log2(1 + sinr)
        se_mrt[ch] = sum_se

        # ===== SLNR =====
        for k in range(K):
            h_d = h_spatial[(k, k)]
            H_interf = np.column_stack([h_spatial[(l, k)] for l in range(K) if l != k])
            f_dict[k] = slnr_beamforming(h_d, H_interf, sigma2, P)

        sum_se = 0
        for l in range(K):
            signal = np.abs(h_spatial[(l, l)].conj().T @ f_dict[l]) ** 2 * P
            interf = sum(
                np.abs(h_spatial.get((l, q), np.zeros(N)).conj().T @ f_dict[q]) ** 2 * P
                for q in range(K) if q != l
            )
            sinr = signal / (interf + sigma2)
            sum_se += np.log2(1 + sinr)
        se_slnr[ch] = sum_se

        # ===== TDMA =====
        sum_se = 0
        for k in range(K):
            h = h_spatial[(k, k)]
            f = mrt_beamforming(h)
            signal = np.abs(h.conj().T @ f) ** 2 * P
            sinr = signal / sigma2
            sum_se += (1.0 / K) * np.log2(1 + sinr)
        se_tdma[ch] = sum_se

        # ===== ST-SLNR =====
        sum_se, _ = compute_st_slnr(h_st, K, M, P, sigma2, Nx, Ny)
        se_stslnr[ch] = sum_se

    return {
        'MRT': np.mean(se_mrt),
        'SLNR': np.mean(se_slnr),
        'TDMA': np.mean(se_tdma),
        'ST-SLNR': np.mean(se_stslnr),
    }


# ============================================================
# Fig. 8: ST-SLNR vs K and M
# ============================================================

def simulate_fig8(K_range, M_range, P_dBm=40, n_ch=n_channels,
                   Nx=Nx_default, Ny=Ny_default):
    """ST-SLNR 和频谱效率 vs K 和 M"""
    P = dBm_to_W(P_dBm)
    N = Nx * Ny
    results = {}

    for K in K_range:
        results[K] = {}
        for M in M_range:
            se_arr = np.zeros(n_ch)
            for ch in range(n_ch):
                f_doppler = np.random.uniform(-doppler_range, doppler_range, K)

                tau_list = np.zeros(K)
                if M > 1:
                    for k in range(K):
                        delta_f_min = min(
                            np.abs(f_doppler[k] - f_doppler[l])
                            for l in range(K) if l != k
                        )
                        if delta_f_min > 10:
                            tau_list[k] = 1.0 / (2 * delta_f_min)
                        else:
                            tau_list[k] = 1e-3

                h_st, h_spatial = generate_channel_geom(
                    K, M, tau_list, f_doppler, Nx, Ny, network='full'
                )

                if M == 1:
                    # 纯空间 SLNR
                    f_dict = {}
                    for k in range(K):
                        h_d = h_spatial[(k, k)]
                        H_interf = np.column_stack(
                            [h_spatial[(l, k)] for l in range(K) if l != k]
                        )
                        f_dict[k] = slnr_beamforming(h_d, H_interf, sigma2, P)

                    sum_se = 0
                    for l in range(K):
                        signal = np.abs(h_spatial[(l, l)].conj().T @ f_dict[l]) ** 2 * P
                        interf = sum(
                            np.abs(h_spatial.get((l, q), np.zeros(N)).conj().T @ f_dict[q]) ** 2 * P
                            for q in range(K) if q != l
                        )
                        sinr = signal / (interf + sigma2)
                        sum_se += np.log2(1 + sinr)
                    se_arr[ch] = sum_se
                else:
                    sum_se, _ = compute_st_slnr(h_st, K, M, P, sigma2, Nx, Ny)
                    se_arr[ch] = sum_se

            results[K][M] = np.mean(se_arr)
        print(f"  K={K} done: " + ", ".join(f"M={m}: {results[K][m]:.2f}" for m in M_range))

    return results
