# -*- coding: utf-8 -*-
"""
信道模型: Rician 衰落信道、UPA 阵列响应、大尺度路径损耗
修复: 统一归一化策略, RIS 每跳用 √NF
"""

import numpy as np
import config as cfg


def compute_positions(M=12):
    """
    计算所有节点位置 (3D 笛卡尔坐标)
    Returns:
        pos_leo: (J, 3) LEO 卫星位置
        pos_geo: (3,) GEO 卫星位置
        pos_lu:  (U, 3) 地面用户位置
        pos_gt:  (K, 3) GEO 终端位置
        pos_ris: (U, 3) RIS 位置 (与 LU 共址)
    """
    pos_geo = cfg.geo_to_cartesian(cfg.geo_lat, cfg.geo_lon, cfg.h_GEO)
    pos_leo = np.zeros((cfg.J, 3))
    for j in range(cfg.J):
        pos_leo[j] = cfg.geo_to_cartesian(
            cfg.geo_lat + cfg.leo_lat_offset[j], cfg.leo_lon, cfg.h_LEO)
    pos_lu = np.zeros((cfg.U, 3))
    for u in range(cfg.U):
        pos_lu[u] = cfg.geo_to_cartesian(
            cfg.geo_lat + cfg.lu_lat_offset[u],
            cfg.geo_lon + cfg.lu_lon_offset[u], 0)
    pos_gt = np.zeros((cfg.K, 3))
    for k in range(cfg.K):
        pos_gt[k] = cfg.geo_to_cartesian(
            cfg.geo_lat + cfg.gt_lat_offset[k],
            cfg.geo_lon + cfg.gt_lon_offset[k], 0)
    # RIS 位置: 与 LU 共址, 有微小高度偏移 (10m)
    pos_ris = pos_lu.copy()
    for u in range(cfg.U):
        direction = pos_leo[0] - pos_lu[u]
        direction = direction / np.linalg.norm(direction)
        pos_ris[u] = pos_lu[u] + direction * 0.01  # 10m 偏移
    return pos_leo, pos_geo, pos_lu, pos_gt, pos_ris


def compute_distances(pos_leo, pos_geo, pos_lu, pos_gt, pos_ris):
    """计算所有链路距离 (km)"""
    dist = {}
    # LEO-LU: (J, U)
    dist['leo_lu'] = np.zeros((cfg.J, cfg.U))
    for j in range(cfg.J):
        for u in range(cfg.U):
            dist['leo_lu'][j, u] = np.linalg.norm(pos_leo[j] - pos_lu[u])
    # LEO-RIS: (J, U) — 卫星到 RIS 距离接近卫星到 LU (RIS 在 LU 附近)
    dist['leo_ris'] = np.zeros((cfg.J, cfg.U))
    for j in range(cfg.J):
        for u in range(cfg.U):
            dist['leo_ris'][j, u] = np.linalg.norm(pos_leo[j] - pos_ris[u])
    # RIS-LU: (U,) 近距离 (~10m)
    dist['ris_lu'] = np.zeros(cfg.U)
    for u in range(cfg.U):
        dist['ris_lu'][u] = np.linalg.norm(pos_ris[u] - pos_lu[u])
    # LEO-GT: (J, K)
    dist['leo_gt'] = np.zeros((cfg.J, cfg.K))
    for j in range(cfg.J):
        for k in range(cfg.K):
            dist['leo_gt'][j, k] = np.linalg.norm(pos_leo[j] - pos_gt[k])
    # GEO-LU: (U,)
    dist['geo_lu'] = np.zeros(cfg.U)
    for u in range(cfg.U):
        dist['geo_lu'][u] = np.linalg.norm(pos_geo - pos_lu[u])
    # GEO-GT: (K,)
    dist['geo_gt'] = np.zeros(cfg.K)
    for k in range(cfg.K):
        dist['geo_gt'][k] = np.linalg.norm(pos_geo - pos_gt[k])
    # GEO-RIS: (U,)
    dist['geo_ris'] = np.zeros(cfg.U)
    for u in range(cfg.U):
        dist['geo_ris'][u] = np.linalg.norm(pos_geo - pos_ris[u])
    return dist


def upa_array_response(Nr, Nc, theta, psi, d_lambda=0.5):
    """UPA 阵列响应向量 (归一化, ||a|| = 1)
    论文中 h̄ 和 Ḡ 均使用归一化阵列响应
    MR 预编码的阵列增益通过 ||v||^4 中的 N 因子体现"""
    N_total = Nr * Nc
    a = np.zeros(N_total, dtype=complex)
    idx = 0
    for nr in range(Nr):
        for nc in range(Nc):
            phase = 2 * np.pi * d_lambda * (
                nr * np.cos(theta) * np.sin(psi) +
                nc * np.sin(theta) * np.sin(psi)
            )
            a[idx] = np.exp(1j * phase)
            idx += 1
    return a / np.sqrt(N_total)  # 归一化: ||a|| = 1


def compute_angles(pos_from, pos_to):
    """计算从 pos_from 到 pos_to 的方位角和俯仰角"""
    diff = pos_to - pos_from
    x, y, z = diff
    r = np.linalg.norm(diff)
    if r == 0:
        return 0, 0
    theta = np.arctan2(y, x)
    psi = np.arcsin(np.clip(z / r, -1, 1))
    return theta, psi


def compute_channel_statistics(dist, kappa_N_dB, kappa_R_dB, kappa_LR_dB,
                               M=12, Nr_sat=None, Nc_sat=None):
    """
    计算信道统计量 (α, β 系数) 和 LoS 分量
    归一化策略:
    - 卫星-地面直连链路: 用 NF (channel_norm_factor)
    - 卫星-RIS 链路: 用 NF (与直连链路相同, 卫星-RIS 距离≈卫星-用户距离)
    - RIS-LU 近距离链路: 用 NF_RIS (独立校准, 使级联路径≈直连路径)
    - GEO-地面链路: 用 NF
    """
    if Nr_sat is None:
        Nr_sat = cfg.N_r
    if Nc_sat is None:
        Nc_sat = cfg.N_c
    N = Nr_sat * Nc_sat
    NF = cfg.channel_norm_factor
    NF_RIS = cfg.ris_norm_factor  # RIS-LU 近距离链路的独立归一化

    # Rician 因子转线性
    kappa_N = cfg.get_kappa_linear(kappa_N_dB)
    kappa_R = cfg.get_kappa_linear(kappa_R_dB)
    kappa_LR = cfg.get_kappa_linear(kappa_LR_dB)
    kappa_LG = cfg.get_kappa_linear(cfg.kappa_LG)
    kappa_GL = cfg.get_kappa_linear(cfg.kappa_GL)
    kappa_GR = cfg.get_kappa_linear(cfg.kappa_GR)

    # RIS UPA 维度
    Mr = int(np.sqrt(M))
    while Mr > 0 and M % Mr != 0:
        Mr -= 1
    Mc = M // Mr

    sigma2 = cfg.get_noise_power() + cfg.system_interference_margin  # 含系统干扰裕量

    # ---- 路径损耗系数 ----
    def path_loss_sat2gnd(d_km):
        """卫星→地面直连链路, 归一化因子 NF"""
        if d_km < 1e-3:
            d_km = 1e-3
        fspl_db = cfg.compute_path_loss_db(d_km)
        total_loss_db = fspl_db - cfg.sat_antenna_gain_dBi - cfg.terminal_gain_dBi
        return 10 ** (-total_loss_db / 10) * NF

    def path_loss_sat2ris(d_km):
        """卫星→RIS, 归一化因子 NF (与直连链路相同)"""
        if d_km < 1e-3:
            d_km = 1e-3
        fspl_db = cfg.compute_path_loss_db(d_km)
        total_loss_db = fspl_db - cfg.sat_antenna_gain_dBi - cfg.ris_gain_dBi
        return 10 ** (-total_loss_db / 10) * NF

    def path_loss_ris2lu(d_km):
        """RIS→LU 近距离, 归一化因子 NF_RIS (独立校准)"""
        d_m = max(d_km * 1e3, 1.0)
        fspl_db = 20 * np.log10(4 * np.pi * d_m * cfg.f_carrier / cfg.c)
        total_loss_db = fspl_db - cfg.ris_gain_dBi - cfg.terminal_gain_dBi
        return 10 ** (-total_loss_db / 10) * NF_RIS

    def path_loss_geo2gnd(d_km, gain_dBi=None):
        """GEO→地面直连链路, 归一化因子 NF"""
        if gain_dBi is None:
            gain_dBi = cfg.geo_antenna_gain_dBi
        if d_km < 1e-3:
            d_km = 1e-3
        fspl_db = cfg.compute_path_loss_db(d_km)
        total_loss_db = fspl_db - gain_dBi - cfg.terminal_gain_dBi
        return 10 ** (-total_loss_db / 10) * NF

    # ---- α, β 系数 ----
    # LEO-LU: α_LL,ju, β_LL,ju
    alpha_LL = np.zeros((cfg.J, cfg.U))
    beta_LL = np.zeros((cfg.J, cfg.U))
    for j in range(cfg.J):
        for u in range(cfg.U):
            mu = path_loss_sat2gnd(dist['leo_lu'][j, u])
            alpha_LL[j, u] = np.sqrt(mu * kappa_N / (1 + kappa_N))
            beta_LL[j, u] = np.sqrt(mu / (1 + kappa_N))

    # RIS-LU: α_R,u, β_R,u (近距离, 用 √NF)
    alpha_R = np.zeros(cfg.U)
    beta_R = np.zeros(cfg.U)
    for u in range(cfg.U):
        mu = path_loss_ris2lu(dist['ris_lu'][u])
        alpha_R[u] = np.sqrt(mu * kappa_R / (1 + kappa_R))
        beta_R[u] = np.sqrt(mu / (1 + kappa_R))

    # LEO-RIS: α_LR,ju, β_LR,ju (用 √NF)
    alpha_LR = np.zeros((cfg.J, cfg.U))
    beta_LR = np.zeros((cfg.J, cfg.U))
    for j in range(cfg.J):
        for u in range(cfg.U):
            mu = path_loss_sat2ris(dist['leo_ris'][j, u])
            alpha_LR[j, u] = np.sqrt(mu * kappa_LR / (1 + kappa_LR))
            beta_LR[j, u] = np.sqrt(mu / (1 + kappa_LR))

    # LEO-GT: α_LG,jk, β_LG,jk (卫星-地面, 用 NF)
    alpha_LG = np.zeros((cfg.J, cfg.K))
    beta_LG = np.zeros((cfg.J, cfg.K))
    for j in range(cfg.J):
        for k in range(cfg.K):
            mu = path_loss_sat2gnd(dist['leo_gt'][j, k])
            alpha_LG[j, k] = np.sqrt(mu * kappa_LG / (1 + kappa_LG))
            beta_LG[j, k] = np.sqrt(mu / (1 + kappa_LG))

    # GEO-LU: α_GL,u, β_GL,u (用 NF)
    alpha_GL = np.zeros(cfg.U)
    beta_GL = np.zeros(cfg.U)
    for u in range(cfg.U):
        mu = path_loss_geo2gnd(dist['geo_lu'][u])
        alpha_GL[u] = np.sqrt(mu * kappa_GL / (1 + kappa_GL))
        beta_GL[u] = np.sqrt(mu / (1 + kappa_GL))

    # GEO-RIS: α_GR,u, β_GR,u (GEO→RIS, 用 NF × NF_RIS / NF = NF_RIS)
    alpha_GR = np.zeros(cfg.U)
    beta_GR = np.zeros(cfg.U)
    for u in range(cfg.U):
        mu = path_loss_geo2gnd(dist['geo_ris'][u], cfg.geo_antenna_gain_dBi)
        # GEO-RIS→LU 级联: GEO→RIS 用 NF, RIS→LU 用 NF_RIS
        # 这里 α_GR 是 GEO→RIS 部分, 只需 NF
        alpha_GR[u] = np.sqrt(mu * kappa_GR / (1 + kappa_GR))
        beta_GR[u] = np.sqrt(mu / (1 + kappa_GR))

    # ---- LoS 阵列响应向量 ----
    pos_leo, pos_geo, pos_lu, pos_gt, pos_ris = compute_positions(M)

    # h̄_LL,ju: 卫星 j -> LU u 的阵列响应 (N,)
    h_bar_LL = np.zeros((cfg.J, cfg.U, N), dtype=complex)
    for j in range(cfg.J):
        for u in range(cfg.U):
            theta, psi = compute_angles(pos_leo[j], pos_lu[u])
            h_bar_LL[j, u] = upa_array_response(Nr_sat, Nc_sat, theta, psi, 0.5)

    # Ḡ_LR,ju: 卫星 j -> RIS u 的阵列响应 (M, N)
    G_bar_LR = np.zeros((cfg.J, cfg.U, M, N), dtype=complex)
    for j in range(cfg.J):
        for u in range(cfg.U):
            theta_aoa, psi_aoa = compute_angles(pos_leo[j], pos_ris[u])
            a_M = upa_array_response(Mr, Mc, theta_aoa, psi_aoa, 0.5)
            theta_aod, psi_aod = compute_angles(pos_leo[j], pos_ris[u])
            a_N = upa_array_response(Nr_sat, Nc_sat, theta_aod, psi_aod, 0.5)
            G_bar_LR[j, u] = np.outer(a_M, a_N.conj())

    # h̄_R,u: RIS u 附近 LU 的阵列响应 (M,)
    h_bar_R = np.zeros((cfg.U, M), dtype=complex)
    for u in range(cfg.U):
        theta, psi = compute_angles(pos_ris[u], pos_lu[u])
        h_bar_R[u] = upa_array_response(Mr, Mc, theta, psi, 0.5)

    # h̄_LG,jk: 卫星 j -> GT k 的阵列响应 (N,)
    h_bar_LG = np.zeros((cfg.J, cfg.K, N), dtype=complex)
    for j in range(cfg.J):
        for k in range(cfg.K):
            theta, psi = compute_angles(pos_leo[j], pos_gt[k])
            h_bar_LG[j, k] = upa_array_response(Nr_sat, Nc_sat, theta, psi, 0.5)

    # ḡ_GR,u: GEO -> RIS u 的阵列响应 (M,)
    g_bar_GR = np.zeros((cfg.U, M), dtype=complex)
    for u in range(cfg.U):
        theta, psi = compute_angles(pos_geo, pos_ris[u])
        g_bar_GR[u] = upa_array_response(Mr, Mc, theta, psi, 0.5)

    # R̄_LL,ju = Ḡ_LR,ju * diag(h̄_R,u)
    R_bar_LL = np.zeros((cfg.J, cfg.U, M, N), dtype=complex)
    for j in range(cfg.J):
        for u in range(cfg.U):
            R_bar_LL[j, u] = G_bar_LR[j, u] * h_bar_R[u][:, None]

    # r̄_GL,u = ḡ_GR,u * diag(h̄_R,u) (用于 GEO 干扰)
    r_bar_GL = np.zeros((cfg.U, M), dtype=complex)
    for u in range(cfg.U):
        r_bar_GL[u] = g_bar_GR[u] * h_bar_R[u]

    chan_stat = {
        'N': N, 'M': M, 'Nr_sat': Nr_sat, 'Nc_sat': Nc_sat,
        'Mr': Mr, 'Mc': Mc,
        'sigma2': sigma2,
        # α, β 系数
        'alpha_LL': alpha_LL, 'beta_LL': beta_LL,
        'alpha_R': alpha_R, 'beta_R': beta_R,
        'alpha_LR': alpha_LR, 'beta_LR': beta_LR,
        'alpha_LG': alpha_LG, 'beta_LG': beta_LG,
        'alpha_GL': alpha_GL, 'beta_GL': beta_GL,
        'alpha_GR': alpha_GR, 'beta_GR': beta_GR,
        # LoS 分量
        'h_bar_LL': h_bar_LL,
        'G_bar_LR': G_bar_LR,
        'h_bar_R': h_bar_R,
        'h_bar_LG': h_bar_LG,
        'g_bar_GR': g_bar_GR,
        'R_bar_LL': R_bar_LL,
        'r_bar_GL': r_bar_GL,
    }

    return chan_stat


def channel_statistics_no_ris(cs_with_ris):
    """从含 RIS 的信道统计量创建 NoRIS 版本
    NoRIS = RIS 物理上不存在, 所有 RIS 相关系数严格置零"""
    import copy
    cs = copy.deepcopy(cs_with_ris)

    # RIS 相关系数严格置零
    cs['alpha_R'] = np.zeros_like(cs['alpha_R'])
    cs['beta_R'] = np.zeros_like(cs['beta_R'])
    cs['alpha_LR'] = np.zeros_like(cs['alpha_LR'])
    cs['beta_LR'] = np.zeros_like(cs['beta_LR'])
    cs['alpha_GR'] = np.zeros_like(cs['alpha_GR'])
    cs['beta_GR'] = np.zeros_like(cs['beta_GR'])

    # LoS 分量置零
    cs['h_bar_R'] = np.zeros_like(cs['h_bar_R'])
    cs['R_bar_LL'] = np.zeros_like(cs['R_bar_LL'])
    cs['G_bar_LR'] = np.zeros_like(cs['G_bar_LR'])
    cs['r_bar_GL'] = np.zeros_like(cs['r_bar_GL'])
    cs['g_bar_GR'] = np.zeros_like(cs['g_bar_GR'])

    return cs
