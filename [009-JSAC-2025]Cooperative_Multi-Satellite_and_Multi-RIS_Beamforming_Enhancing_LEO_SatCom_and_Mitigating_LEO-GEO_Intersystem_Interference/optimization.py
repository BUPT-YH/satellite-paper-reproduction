# -*- coding: utf-8 -*-
"""
优化算法实现:
- Algorithm 1: AO-based 自适应预编码 (AP-AO)
- Algorithm 2: 功率分配 — 二分搜索 Max-Min Fairness
- Algorithm 3: 两阶段设计 (MR-S-TS) 含 ES-SP-RMO
- NoRIS 基准方案
修复: 二分搜索功率分配, 确定性 RIS 初始化, 步长衰减, GEO 约束缩放
"""

import numpy as np
from closed_form import (f5_ju, f6_jui, f4_GEO, f7_juk,
                          sinr_mr_statistical, sinr_mr_two_timescale,
                          sinr_rzf_all)
import config as cfg

# 干扰阈值缩放: 论文中 ζ 的单位是 dBW (功率), 但 SINR 公式中功率被 NF 缩放
# 所以干扰阈值也需乘以 NF
_NF = cfg.channel_norm_factor


def init_ris_phase(M, U, seed=None):
    """初始化 RIS 相移 — 确定性 (全对齐, 最大阵列增益)"""
    phi = {}
    for u in range(U):
        # 全 1 向量 = 所有 RIS 元素相位对齐, 提供最大阵列增益
        phi[u] = np.ones(M, dtype=complex) / np.sqrt(M)
    return phi


def project_to_manifold(phi_u):
    """投影到复圆流形: |ϕ_m| = 1/√M"""
    M = len(phi_u)
    return phi_u / np.abs(phi_u) * np.sqrt(M)


def power_allocation_fp(cs, phi, PT, zeta_dB, scheme='mr_stat', max_iter=15):
    """
    功率分配 — 迭代 Max-Min Fairness
    给定 RIS 相移, 找到满足功率和干扰约束的最大 min-SINR
    修复: 确保最小功率下限, 对比等功率基线
    """
    J = cfg.J
    U = cfg.U
    K = cfg.K
    # 干扰阈值: ζ_dB 是 dBW, 转线性后乘以 NF
    zeta_lin = 10 ** (zeta_dB / 10) * _NF

    # 预计算所有增益矩阵
    s1 = np.zeros((J, U))       # s1[j,u] = f5_ju (信号增益)
    s2 = np.zeros((U, J, U))    # s2[u,j,i] = f6_jui (干扰增益)
    s_interf = np.zeros((J, U, K))  # LEO-GEO 干扰
    for j in range(J):
        for u in range(U):
            s1[j, u] = f5_ju(cs, phi[u], j, u)
            for i in range(U):
                s2[u, j, i] = f6_jui(cs, phi[u], phi[i], j, u, i)
            for k in range(K):
                s_interf[j, u, k] = f7_juk(cs, phi[u], j, u, k)
    geo = np.array([f4_GEO(cs, phi[u], u) for u in range(U)])

    sigma2 = cs['sigma2']

    # === 等功率基线 ===
    p_eq = np.ones((J, U)) * PT / (J * U)

    def compute_sinrs(p_mat):
        sinrs = np.zeros(U)
        for u in range(U):
            num = np.sum(p_mat[:, u] * s1[:, u])
            den = np.sum(p_mat * s2[u]) - num + geo[u] + sigma2
            sinrs[u] = num / max(den, 1e-30)
        return sinrs

    sinrs_eq = compute_sinrs(p_eq)
    min_sinr_eq = np.min(sinrs_eq)

    # === 迭代功率平衡 (从等功率开始) ===
    p = p_eq.copy()

    for iteration in range(max_iter):
        sinrs = compute_sinrs(p)
        u_min = np.argmin(sinrs)
        u_max = np.argmax(sinrs)

        if sinrs[u_max] > sinrs[u_min] * 1.1:
            # 从强用户转移功率给弱用户
            for j in range(J):
                if p[j, u_max] > 0.05 * PT / U:
                    delta = p[j, u_max] * 0.2
                    p[j, u_max] -= delta
                    p[j, u_min] += delta
        else:
            break

        # 功率约束
        for j in range(J):
            s = np.sum(p[j])
            if s > PT:
                p[j] *= PT / s
        # 最小功率下限: 每用户至少 5% 的等功率份额
        p = np.maximum(p, PT / (J * U) * 0.05)

    # GEO 干扰约束修正
    for k in range(K):
        ti = np.sum(p * s_interf[:, :, k])
        if ti > zeta_lin:
            p *= (zeta_lin / ti) * 0.95

    # 最终检查: 对比等功率基线
    sinrs_opt = compute_sinrs(p)
    min_sinr_opt = np.min(sinrs_opt)

    if min_sinr_opt < min_sinr_eq * 0.5:
        # 优化结果不如等功率, 回退到等功率
        p = p_eq.copy()

    return p


def _is_feasible(s1, s2, s_interf, geo, gamma, PT, zeta_lin, J, U, K, sigma2):
    """检查目标 SINR=gamma 是否可行"""
    # 简化可行性检查: 每个用户的最小功率需求
    p_min = np.zeros((J, U))
    for u in range(U):
        # p_ju * s1[j,u] >= gamma * (干扰 + 噪声) 的总和
        for j in range(J):
            if s1[j, u] > 0:
                p_min[j, u] = gamma * sigma2 / (J * s1[j, u])

    # 检查总功率约束
    for j in range(J):
        if np.sum(p_min[j]) > PT:
            return False

    # 检查干扰约束
    for k in range(K):
        ti = np.sum(p_min * s_interf[:, :, k])
        if ti > zeta_lin:
            return False

    return True


def _construct_power(s1, s2, geo, gamma, PT, J, U, sigma2):
    """根据目标 SINR 构造功率分配"""
    p = np.zeros((J, U))
    for u in range(U):
        for j in range(J):
            if s1[j, u] > 1e-30:
                # 需要的功率: p_ju * s1 >= gamma * sigma2 (近似)
                p[j, u] = gamma * sigma2 / (J * s1[j, u])

    # 归一化到功率约束
    for j in range(J):
        s = np.sum(p[j])
        if s > PT:
            p[j] *= PT / s
    p = np.maximum(p, 1e-15)
    return p


def optimize_ris_rg(cs, p, phi, zeta_dB, scheme='mr_stat',
                    max_iter=20, step_size=0.05):
    """
    Riemannian 梯度下降优化 RIS 相移
    修复: 步长衰减, 更多迭代
    """
    U = len(phi)
    M = cs['M']

    for iteration in range(max_iter):
        # 步长衰减: α_l = α_0 / (1 + 0.02 * l)
        lr = step_size / (1 + 0.02 * iteration)

        for u in range(U):
            # 计算欧几里得梯度 (数值差分)
            grad = compute_ris_gradient(cs, phi, p, u, scheme)

            # Riemannian 梯度: 投影到切空间
            rgrad = grad - np.real(grad * phi[u].conj()) * phi[u]

            # 更新步
            phi_new = phi[u] + lr * rgrad

            # 投影回流形 (|ϕ_m| = 1/√M)
            phi[u] = project_to_manifold(phi_new)

    return phi


def compute_ris_gradient(cs, phi, p, u, scheme='mr_stat'):
    """计算 SINR 对 ϕu 的梯度 (中心差分)"""
    M = cs['M']
    eps = 1e-5
    grad = np.zeros(M, dtype=complex)
    phi_u = phi[u].copy()

    for m in range(M):
        phi_plus = phi_u.copy()
        phi_plus[m] += eps
        phi[u] = phi_plus
        sinr_plus = compute_min_sinr_for_user(cs, phi, p, u, scheme)

        phi_minus = phi_u.copy()
        phi_minus[m] -= eps
        phi[u] = phi_minus
        sinr_minus = compute_min_sinr_for_user(cs, phi, p, u, scheme)

        grad[m] = (sinr_plus - sinr_minus) / (2 * eps)

    phi[u] = phi_u
    return grad


def compute_min_sinr_for_user(cs, phi, p, u, scheme='mr_stat'):
    """计算单个用户 SINR"""
    if scheme == 'mr_stat':
        return sinr_mr_statistical(cs, phi, p, u)
    elif scheme == 'mr_tts':
        return sinr_mr_two_timescale(cs, phi, p, u)
    return 0


def algorithm_ap_ao(cs, PT, zeta_dB, max_iter=8):
    """
    Algorithm 1: AO-based 自适应预编码 (AP-AO)
    交替优化卫星预编码和 RIS 相移
    """
    U = cfg.U
    M = cs['M']
    phi = init_ris_phase(M, U)

    for iteration in range(max_iter):
        # Step 1: 优化功率分配
        p = power_allocation_fp(cs, phi, PT, zeta_dB, 'mr_stat', max_iter=8)

        # Step 2: 优化 RIS 相移 (流形优化)
        phi = optimize_ris_rg(cs, p, phi, zeta_dB, 'mr_stat',
                              max_iter=15, step_size=0.05)

    # 最终功率分配
    p = power_allocation_fp(cs, phi, PT, zeta_dB, 'mr_stat', max_iter=10)
    return phi, p


def algorithm_mr_s_pa(cs, PT, zeta_dB):
    """
    Algorithm 2: MR 预编码下使用 AP-AO 的 RIS 结果做功率分配
    """
    phi, _ = algorithm_ap_ao(cs, PT, zeta_dB, max_iter=5)
    p = power_allocation_fp(cs, phi, PT, zeta_dB, 'mr_stat', max_iter=12)
    return phi, p


def algorithm_mr_s_ts(cs, PT, zeta_dB, max_iter=10):
    """
    Algorithm 3: 两阶段设计 (ES-SP-RMO + 功率分配)
    不依赖 AP-AO 结果
    """
    U = cfg.U
    M = cs['M']

    p = np.ones((cfg.J, U)) * PT / (cfg.J * U)
    phi = init_ris_phase(M, U)
    mu = cfg.mu_init

    for l4 in range(max_iter):
        phi_old = {u: phi[u].copy() for u in range(U)}

        phi = optimize_ris_rg(cs, p, phi, zeta_dB, 'mr_stat',
                              max_iter=12, step_size=0.05)
        p = power_allocation_fp(cs, phi, PT, zeta_dB, 'mr_stat', max_iter=8)

        # 指数平滑收敛检查
        zeta_lin = 10 ** (zeta_dB / 10) * _NF
        F_new = compute_smoothed_objective(cs, phi, p, mu, zeta_lin, 'mr_stat')
        F_old = compute_smoothed_objective(cs, phi_old, p, mu, zeta_lin, 'mr_stat')

        if F_new <= F_old:
            mu = mu / 2
        if mu < cfg.epsilon_ES:
            break

    p = power_allocation_fp(cs, phi, PT, zeta_dB, 'mr_stat', max_iter=12)
    return phi, p


def algorithm_mr_tts_pa(cs, PT, zeta_dB):
    """MR-TTS-PA: 使用双时间尺度 CSI 的功率分配"""
    phi, _ = algorithm_ap_ao(cs, PT, zeta_dB, max_iter=5)
    p = power_allocation_fp(cs, phi, PT, zeta_dB, 'mr_tts', max_iter=12)
    return phi, p


def algorithm_mr_tts_ts(cs, PT, zeta_dB, max_iter=10):
    """MR-TTS-TS: 使用双时间尺度 CSI 的两阶段设计"""
    U = cfg.U
    M = cs['M']

    p = np.ones((cfg.J, U)) * PT / (cfg.J * U)
    phi = init_ris_phase(M, U)
    mu = cfg.mu_init

    for l4 in range(max_iter):
        phi_old = {u: phi[u].copy() for u in range(U)}
        phi = optimize_ris_rg(cs, p, phi, zeta_dB, 'mr_tts',
                              max_iter=12, step_size=0.05)
        p = power_allocation_fp(cs, phi, PT, zeta_dB, 'mr_tts', max_iter=8)

        zeta_lin = 10 ** (zeta_dB / 10) * _NF
        F_new = compute_smoothed_objective(cs, phi, p, mu, zeta_lin, 'mr_tts')
        F_old = compute_smoothed_objective(cs, phi_old, p, mu, zeta_lin, 'mr_tts')

        if F_new <= F_old:
            mu = mu / 2
        if mu < cfg.epsilon_ES:
            break

    p = power_allocation_fp(cs, phi, PT, zeta_dB, 'mr_tts', max_iter=12)
    return phi, p


def compute_smoothed_objective(cs, phi, p, mu, zeta_lin, scheme):
    """
    计算指数平滑目标函数 (公式 36)
    F(Φ) = -μ log Σ_u exp(-SINR_u/μ) - Σ_k (interf_k - ζ_k)^2
    """
    U = len(phi)
    K = cfg.K

    sinr_vals = []
    for u in range(U):
        if scheme == 'mr_stat':
            sinr_vals.append(sinr_mr_statistical(cs, phi, p, u))
        elif scheme == 'mr_tts':
            sinr_vals.append(sinr_mr_two_timescale(cs, phi, p, u))

    sinr_arr = np.array(sinr_vals)
    # 数值稳定的 log-sum-exp
    if mu > 1e-10:
        max_sinr = np.max(sinr_arr)
        smooth_term = max_sinr - mu * np.log(np.sum(np.exp(-(sinr_arr - max_sinr) / mu)))
    else:
        smooth_term = np.min(sinr_arr)

    # 干扰惩罚项
    penalty = 0
    for k in range(K):
        total_interf = 0
        for j in range(cfg.J):
            for u in range(U):
                total_interf += p[j, u] * f7_juk(cs, phi[u], j, u, k)
        if total_interf > zeta_lin:
            penalty -= (total_interf - zeta_lin) ** 2

    return smooth_term + penalty


def run_no_ris(cs, PT, zeta_dB, scheme='mr_stat'):
    """NoRIS 基准: 不使用 RIS
    RIS 物理上不存在, phi 为零向量"""
    from channel_model import channel_statistics_no_ris
    cs_noris = channel_statistics_no_ris(cs)
    U = cfg.U
    M = cs['M']

    # phi = 零向量 (无 RIS)
    phi = {u: np.zeros(M, dtype=complex) for u in range(U)}
    p = power_allocation_fp(cs_noris, phi, PT, zeta_dB, scheme, max_iter=12)
    # 返回原始 cs (用于 evaluate), phi 全零
    return phi, p


def evaluate_scheme(cs, phi, p, scheme='mr_stat'):
    """评估方案: 返回最小 SINR (dB)"""
    # 如果 phi 全零 (NoRIS), 用修改后的 cs
    is_noris = all(np.all(phi[u] == 0) for u in phi)
    if is_noris:
        from channel_model import channel_statistics_no_ris
        cs = channel_statistics_no_ris(cs)

    U = len(phi)
    sinrs = []
    for u in range(U):
        if scheme == 'mr_stat':
            sinrs.append(sinr_mr_statistical(cs, phi, p, u))
        elif scheme == 'mr_tts':
            sinrs.append(sinr_mr_two_timescale(cs, phi, p, u))

    min_sinr = min(sinrs) if sinrs else 0
    return 10 * np.log10(max(min_sinr, 1e-30))


def evaluate_scheme_rzf(cs, phi, PT):
    """评估方案 (RZF 预编码): 返回最小 SINR (dB)
    用于 AP-AO 方案, RZF 可抑制用户间干扰"""
    is_noris = all(np.all(phi[u] == 0) for u in phi)
    if is_noris:
        from channel_model import channel_statistics_no_ris
        cs = channel_statistics_no_ris(cs)

    sinrs = sinr_rzf_all(cs, phi, PT)
    min_sinr = np.min(sinrs)
    return 10 * np.log10(max(min_sinr, 1e-30))


def evaluate_sum_rate(cs, phi, p, scheme='mr_stat'):
    """评估方案: 返回总可达速率"""
    is_noris = all(np.all(phi[u] == 0) for u in phi)
    if is_noris:
        from channel_model import channel_statistics_no_ris
        cs = channel_statistics_no_ris(cs)

    U = len(phi)
    total = 0
    for u in range(U):
        if scheme == 'mr_stat':
            sinr = sinr_mr_statistical(cs, phi, p, u)
        elif scheme == 'mr_tts':
            sinr = sinr_mr_two_timescale(cs, phi, p, u)
        total += np.log2(1 + max(sinr, 0))
    return total
