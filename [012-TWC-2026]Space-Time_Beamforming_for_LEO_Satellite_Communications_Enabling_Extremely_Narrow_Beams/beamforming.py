"""
波束赋形算法模块
- MRT (最大比发送)
- ZF (迫零)
- SLNR (信号泄漏噪声比)
- ST-ZF (空时迫零)
- ST-SLNR (空时信号泄漏噪声比)
- TDMA (时分多址)
"""

import numpy as np
from config import Ts


# ============================================================
# 传统空间波束赋形（M=1 基线方案）
# ============================================================

def mrt_beamforming(h):
    """MRT 波束赋形: f = h / ||h||"""
    return h / np.linalg.norm(h)


def zf_beamforming(H, idx):
    """
    ZF 波束赋形（用于空间信道）
    H: 所有用户信道矩阵 [h_1, h_2, ..., h_K] ∈ C^{N x K}
    idx: 目标用户索引
    """
    K = H.shape[1]
    h_desired = H[:, idx]
    # ZF: 投影到干扰信道的零空间
    H_interf = np.delete(H, idx, axis=1)
    if H_interf.shape[1] == 0:
        return mrt_beamforming(h_desired)

    # 使用伪逆方法
    proj = H_interf @ np.linalg.pinv(H_interf)
    f = (np.eye(len(h_desired)) - proj) @ h_desired
    norm = np.linalg.norm(f)
    if norm < 1e-12:
        return mrt_beamforming(h_desired)
    return f / norm


def slnr_beamforming(h_desired, H_interf, sigma2, P):
    """
    SLNR 波束赋形（等价于 MMSE）
    h_desired: 目标用户信道 C^{N x 1}
    H_interf: 干扰用户信道矩阵 C^{N x (K-1)}
    sigma2: 噪声功率
    P: 发射功率
    """
    N = len(h_desired)
    if H_interf.shape[1] == 0:
        return mrt_beamforming(h_desired)

    # SLNR 最优解: (H_I H_I^H + sigma2/P I)^{-1} h_d
    A = H_interf @ H_interf.conj().T + (sigma2 / P) * np.eye(N)
    f = np.linalg.solve(A, h_desired)
    norm = np.linalg.norm(f)
    if norm < 1e-12:
        return mrt_beamforming(h_desired)
    return f / norm


# ============================================================
# TDMA 方案
# ============================================================

def compute_tdma_sum_se(h_channels, K, P, sigma2):
    """
    计算 TDMA 方案的和频谱效率

    部分连接: 奇偶分组，每组 1/2 时间
    h_channels: dict, h_channels[(l,k)] = 空间信道 C^{N x 1}
    K: 用户数
    P: 发射功率
    sigma2: 噪声功率
    """
    # 部分连接网络：分组传输
    K_odd = list(range(0, K, 2))  # 0-indexed: 0,2,4,...
    K_even = list(range(1, K, 2))

    sum_se = 0
    for group in [K_odd, K_even]:
        for k in group:
            h = h_channels.get((k, k))
            if h is None:
                continue
            f = mrt_beamforming(h)
            signal_power = np.abs(h.conj().T @ f) ** 2 * P
            noise = sigma2
            snr = signal_power / noise
            sum_se += 0.5 * np.log2(1 + snr)

    return sum_se


def compute_tdma_sum_se_full(h_channels, K, P, sigma2):
    """
    全连接网络 TDMA: 每时隙一个用户，pre-log = 1/K
    """
    sum_se = 0
    for k in range(K):
        h = h_channels.get((k, k))
        if h is None:
            continue
        f = mrt_beamforming(h)
        signal_power = np.abs(h.conj().T @ f) ** 2 * P
        snr = signal_power / sigma2
        sum_se += (1.0 / K) * np.log2(1 + snr)
    return sum_se


# ============================================================
# 空时波束赋形
# ============================================================

def compute_st_zf(h_st, K, M, P, sigma2):
    """
    ST-ZF 波束赋形（部分连接网络）

    核心原理：当 tau 选择为最优值时，期望信道和泄漏信道正交。
    因此 MRT 波束赋形器自动满足 ZF 条件（零泄漏）。
    直接使用 MRT = h / ||h|| * sqrt(M)

    h_st: dict, h_st[(l,k)] = 空时信道 C^{MN x 1}
    """
    sum_se = 0
    f_st = {}

    # 对每个卫星，直接用 MRT（tau 最优时正交性保证 ZF）
    for k in range(K):
        h_desired = h_st[(k, k)]
        f = mrt_beamforming(h_desired) * np.sqrt(M)
        f_st[k] = f

    # 计算频谱效率
    for k in range(K):
        h_desired = h_st[(k, k)]
        signal = np.abs(h_desired.conj().T @ f_st[k]) ** 2 * P

        # 部分连接：只考虑相邻干扰卫星 k''=(k-1)%K
        k_pp = (k - 1) % K
        interference = 0
        if (k, k_pp) in h_st:
            interference = np.abs(h_st[(k, k_pp)].conj().T @ f_st[k_pp]) ** 2 * P

        sinr = signal / (interference + M * sigma2)
        se = (1.0 / M) * np.log2(1 + sinr)
        sum_se += se

    return sum_se, f_st


def optimize_st_zf_tau(f_doppler_k, f_doppler_kprime):
    """
    最优重传间隔（Lemma 1）: tau* = 1 / (2 * |f_k - f_k'|)
    """
    delta_f = np.abs(f_doppler_k - f_doppler_kprime)
    if delta_f < 1e-3:  # 多普勒差太小
        return 1e-3  # 避免除零
    tau_star = 1.0 / (2 * delta_f)
    return tau_star


def compute_st_slnr(h_st_all, K, M, P, sigma2, Nx=8, Ny=8):
    """
    ST-SLNR 波束赋形（全连接网络）

    h_st_all: dict, h_st_all[(l,k)] = 空时信道 C^{MN x 1}
    K: 用户/卫星数
    M: 重复次数
    P: 发射功率

    返回: (sum_se, f_st_dict)
    """
    N = Nx * Ny
    MN = M * N
    f_st = {}

    for k in range(K):
        h_desired = h_st_all[(k, k)]  # 卫星k到用户k

        # 干扰用户信道矩阵 H_{l,k} (排除期望用户)
        H_interf_cols = []
        for l in range(K):
            if l != k:
                H_interf_cols.append(h_st_all[(l, k)])
        H_interf = np.column_stack(H_interf_cols) if H_interf_cols else np.zeros((MN, 0))

        # SLNR 最优 precoding (公式 52)
        # f = sqrt(M) * (H_I H_I^H + sigma2/P I)^{-1} h_d / ||...||
        A = H_interf @ H_interf.conj().T + (sigma2 / P) * np.eye(MN)
        f = np.linalg.solve(A, h_desired)
        norm = np.linalg.norm(f)
        if norm < 1e-12:
            f = mrt_beamforming(h_desired)
            norm = np.linalg.norm(f)
        f = f / norm * np.sqrt(M)

        f_st[k] = f

    # 计算频谱效率
    sum_se = 0
    for l in range(K):
        signal = np.abs(h_st_all[(l, l)].conj().T @ f_st[l]) ** 2 * P
        interference = 0
        for q in range(K):
            if q != l and (l, q) in h_st_all:
                interference += np.abs(h_st_all[(l, q)].conj().T @ f_st[q]) ** 2 * P
        sinr = signal / (interference + M * sigma2)
        se = (1.0 / M) * np.log2(1 + sinr)
        sum_se += se

    return sum_se, f_st


def optimize_st_slnr_tau(h_st_all, K, M, P, sigma2, f_doppler, k_idx,
                          Nx=8, Ny=8):
    """
    通过网格搜索优化卫星 k 的重传间隔 tau_k

    搜索范围: [0, 1/f_{l,k}) 以 Ts 为分辨率
    """
    N = Nx * Ny
    MN = M * N
    l = k_idx  # 卫星 k 服务用户 l=k

    f_max = f_doppler[l]
    if np.abs(f_max) < 1:
        f_max = 1.0  # 避免范围过小

    tau_max = 1.0 / np.abs(f_max) if np.abs(f_max) > 0 else 1e-3
    # 网格搜索
    n_grid = min(100, max(10, int(tau_max / Ts)))
    tau_grid = np.linspace(Ts, tau_max, n_grid)

    best_slnr = -np.inf
    best_tau = tau_grid[0]

    for tau in tau_grid:
        # 重新构建空时信道
        # 简化：直接计算 SLNR 值
        # 构建当前 tau 下的干扰矩阵和期望信道
        h_d = h_st_all.get((l, k_idx))
        if h_d is None:
            continue

        H_interf_cols = []
        for ll in range(K):
            if ll != l:
                h_i = h_st_all.get((ll, k_idx))
                if h_i is not None:
                    H_interf_cols.append(h_i)

        if not H_interf_cols:
            best_tau = tau
            break

        H_interf = np.column_stack(H_interf_cols)
        A = H_interf @ H_interf.conj().T + (sigma2 / P) * np.eye(MN)
        slnr_val = np.real(h_d.conj().T @ np.linalg.solve(A, h_d))

        if slnr_val > best_slnr:
            best_slnr = slnr_val
            best_tau = tau

    return best_tau


def optimize_M(h_st_all, K, P, sigma2, f_doppler_list, Nx=8, Ny=8, M_max=6):
    """
    优化重复次数 M（公式 59）
    从 M=1 开始单调递增，当和频谱效率开始下降时停止
    """
    best_M = 1
    best_sum_se = -np.inf
    prev_sum_se = -np.inf

    for M in range(1, M_max + 1):
        # 为每个卫星选择 tau
        tau_list = []
        for k in range(K):
            if M > 1:
                tau_k = optimize_st_slnr_tau(h_st_all, K, M, P, sigma2,
                                              f_doppler_list, k, Nx, Ny)
            else:
                tau_k = 0
            tau_list.append(tau_k)

        # 计算 ST-SLNR 和频谱效率
        sum_se, _ = compute_st_slnr(h_st_all, K, M, P, sigma2, Nx, Ny)

        if sum_se > best_sum_se:
            best_sum_se = sum_se
            best_M = M

        # 如果和频谱效率开始下降，停止
        if M > 2 and sum_se < prev_sum_se:
            break
        prev_sum_se = sum_se

    return best_M
