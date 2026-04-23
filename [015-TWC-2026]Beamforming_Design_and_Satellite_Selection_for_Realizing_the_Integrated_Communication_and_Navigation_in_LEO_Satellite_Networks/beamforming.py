"""
波束赋形算法实现
包含 WMMSE (等效 DC 规划) 波束赋形和基准方案 (MRT, ZF, MMSE, ST-ZF)
WMMSE 是 DC 规划的经典等效实现，收敛到加权和速率最大化问题的局部最优解
"""
import numpy as np
import config as cfg


def beamforming_mrt(h, P_max):
    """
    MRT 波束赋形（公式 30）
    最大化接收 SNR，不考虑干扰
    """
    return np.sqrt(P_max) * h / np.linalg.norm(h)


def beamforming_zf(H, P_max):
    """
    ZF 波束赋形（公式 31）
    消除卫星内干扰，但会放大噪声
    H: 信道矩阵 (Ks x N)
    返回: 波束赋形矩阵 (N x Ks)
    """
    Ks, N = H.shape
    H_pinv = np.conj(H.T) @ np.linalg.inv(H @ np.conj(H.T))
    beta = np.sqrt(P_max * Ks) / np.linalg.norm(H_pinv, 'fro')
    return beta * H_pinv


def beamforming_mmse(H, P_max, noise_power):
    """
    MMSE 波束赋形（公式 32）
    平衡干扰抑制和噪声增强
    H: 信道矩阵 (Ks x N)
    返回: 波束赋形矩阵 (N x Ks)
    """
    Ks, N = H.shape
    W = np.conj(H.T) @ np.linalg.inv(H @ np.conj(H.T) + (noise_power / P_max) * np.eye(Ks))
    # 归一化每列功率不超过 P_max
    for k in range(Ks):
        norm = np.linalg.norm(W[:, k])
        if norm > 0:
            W[:, k] *= np.sqrt(P_max) / norm
    return W


def wmmse_beamforming(H, P_max, noise_power, max_iter=20):
    """
    WMMSE 迭代波束赋形（等效于论文中的 DC 规划，Algorithm 1）
    通过加权的 MMSE 问题与加权和速率最大化问题的等价性，
    利用交替优化求局部最优解

    H: 信道矩阵 (Ks x N)
    返回: 波束赋形矩阵 W (N x Ks)
    """
    Ks, N = H.shape
    if Ks == 0:
        return np.zeros((N, 0))

    # 初始化: MMSE 波束赋形（更好的初始点）
    W = beamforming_mmse(H, P_max, noise_power)

    best_rate = 0.0
    best_W = W.copy()

    for iteration in range(max_iter):
        # 计算当前和速率
        cur_rate = 0.0
        for k in range(Ks):
            h_k = H[k]
            signal = np.abs(np.conj(h_k).T @ W[:, k]) ** 2
            interference = sum(np.abs(np.conj(h_k).T @ W[:, j]) ** 2 for j in range(Ks) if j != k)
            sinr = signal / (interference + noise_power)
            cur_rate += np.log2(1 + sinr)

        if cur_rate > best_rate:
            best_rate = cur_rate
            best_W = W.copy()

        # 步骤 1: 固定 W，计算 MMSE 接收机 u_k 和 MSE 权重 ξ_k
        u = np.zeros(Ks, dtype=complex)
        xi = np.zeros(Ks)

        for k in range(Ks):
            h_k = H[k]
            sig_plus_int = sum(np.abs(np.conj(h_k).T @ W[:, j]) ** 2 for j in range(Ks)) + noise_power
            u[k] = np.conj(W[:, k]).T @ h_k / sig_plus_int
            mse = 1 - 2 * np.real(u[k] * np.conj(h_k).T @ W[:, k]) + np.abs(u[k]) ** 2 * sig_plus_int
            xi[k] = 1.0 / max(mse, 1e-10)

        # 步骤 2: 逐波束更新 W
        # 对每个 k: w_k = (A_k)^{-1} ξ_k u_k^* h_k
        # A_k = Σ_j ξ_j |u_j|² h_j h_j^H + ξ_k σ² I
        for k in range(Ks):
            h_k = H[k]
            # 构建 A_k
            A_k = np.zeros((N, N), dtype=complex)
            for j in range(Ks):
                h_j = H[j]
                A_k += xi[j] * np.abs(u[j]) ** 2 * np.outer(h_j, np.conj(h_j))
            A_k += xi[k] * noise_power * np.eye(N)

            try:
                w_k = xi[k] * np.conj(u[k]) * np.linalg.solve(A_k, h_k)
            except np.linalg.LinAlgError:
                continue

            # 功率约束: ||w_k||² ≤ P_max
            w_norm = np.linalg.norm(w_k)
            if w_norm > np.sqrt(P_max):
                w_k *= np.sqrt(P_max) / w_norm
            W[:, k] = w_k

    # 返回最优解
    return best_W


def compute_sum_rate_for_scheme(channels, alpha, scheme, P_max, noise_power,
                                 S=None, C=None):
    """
    给定波束赋形方案，计算系统和速率
    scheme: 'DC', 'MRT', 'ZF', 'MMSE', 'ST-ZF'
    """
    if S is None:
        S = cfg.S
    if C is None:
        C = cfg.C

    sum_rate = 0.0

    for s in range(S):
        served_ues = [c for c in range(C) if alpha.get((s, c), 0) == 1]
        Ks = len(served_ues)
        if Ks == 0:
            continue

        h_list = [channels[(s, c)] for c in served_ues]
        H = np.array(h_list)  # (Ks x N)

        # 计算波束赋形
        try:
            if scheme == 'DC':
                W = wmmse_beamforming(H, P_max, noise_power, max_iter=cfg.J_DC)
            elif scheme == 'MRT':
                W = np.array([beamforming_mrt(h, P_max) for h in h_list]).T
            elif scheme == 'ZF':
                W = beamforming_zf(H, P_max)
            elif scheme == 'MMSE':
                W = beamforming_mmse(H, P_max, noise_power)
            elif scheme == 'ST-ZF':
                # ST-ZF 等效于 ZF 但频谱效率降为 1/T
                W = beamforming_zf(H, P_max)
            else:
                W = np.array([beamforming_mrt(h, P_max) for h in h_list]).T
        except (np.linalg.LinAlgError, Exception):
            W = np.array([beamforming_mrt(h, P_max) for h in h_list]).T

        # 计算每个 UE 的 SINR 和速率
        st_zf_factor = 1.0 / 3.0 if scheme == 'ST-ZF' else 1.0

        for idx in range(Ks):
            h_c = h_list[idx]
            w_c = W[:, idx]

            signal = np.abs(np.conj(h_c).T @ w_c) ** 2
            interference = sum(np.abs(np.conj(h_c).T @ W[:, j]) ** 2
                              for j in range(Ks) if j != idx)

            sinr = signal / (interference + noise_power)
            rate = cfg.B * np.log2(1 + sinr) * st_zf_factor
            sum_rate += rate

    return sum_rate
