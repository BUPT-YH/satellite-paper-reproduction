"""
阶段2: 统计解到确定性解的映射
包括: 离散需求转换 (Algorithm 2) + 照明模式设计
"""
import numpy as np
from config import BANDWIDTH, NOISE_FLOOR


def rounding_algorithm(rho_star, M, K):
    """
    Algorithm 2: 将连续激活概率转换为离散时隙需求
    """
    N = len(rho_star)
    d_hat = np.floor(rho_star * M).astype(int)
    residual = rho_star * M - d_hat
    target = int(K * M)
    current = np.sum(d_hat)
    needed = target - current
    if needed > 0:
        indices = np.argsort(-residual)
        for i in range(min(needed, N)):
            d_hat[indices[i]] += 1
    d_hat = np.minimum(d_hat, M)
    d_hat = np.maximum(d_hat, 1)
    return d_hat


def compute_modified_power(d_hat, M, A, b, d, B=BANDWIDTH):
    """
    根据离散化需求重新计算功率
    论文式 (13)
    """
    from inverse_matrix_optimization import g_vector
    rho_hat = d_hat / M
    N = len(d_hat)
    G_hat = np.diag(g_vector(rho_hat, d, B))
    sr = np.max(np.abs(np.linalg.eigvals(G_hat @ A)))
    if sr >= 1:
        rho_hat = np.minimum(rho_hat * 1.05, 1.0)
        d_hat_adj = np.round(rho_hat * M).astype(int)
        d_hat_adj = np.maximum(d_hat_adj, 1)
        rho_hat = d_hat_adj / M
        G_hat = np.diag(g_vector(rho_hat, d, B))
    I_mat = np.eye(N)
    try:
        inv_matrix = np.linalg.inv(I_mat - G_hat @ A)
        p_hat = np.diag(1.0 / rho_hat) @ inv_matrix @ G_hat @ b
    except np.linalg.LinAlgError:
        p_hat = np.ones(N) * 1.0
    p_hat = np.maximum(p_hat, 0.01)
    return p_hat, rho_hat
def mpmm_scheduling(rho_hat, p_hat, d_hat, M, K, A, penalty_weight=0.0):
    """
    MPMM: 二元二次规划求解照明模式
    贪心 + 局部搜索
    """
    N = len(d_hat)
    X = np.zeros((N, M), dtype=int)
    remaining = d_hat.copy()
    for t in range(M):
        priorities = remaining.copy().astype(float)
        if t > 0:
            for n in range(N):
                for j in range(N):
                    if X[j, t-1] == 1 and j != n:
                        priorities[n] -= A[j, n] * 0.5
        active_count = min(K, int(np.sum(remaining > 0)))
        if active_count == 0:
            break
        indices = np.argsort(-priorities)[:active_count]
        X[indices, t] = 1
        remaining[indices] -= 1
    for n in range(N):
        while remaining[n] > 0:
            inactive_ts = np.where(X[n, :] == 0)[0]
            if len(inactive_ts) == 0:
                break
            active_counts = np.sum(X[:, inactive_ts], axis=0)
            best_t = inactive_ts[np.argmin(active_counts)]
            if np.sum(X[:, best_t]) >= K:
                active_beams = np.where(X[:, best_t] == 1)[0]
                surplus = remaining[active_beams]
                replace_idx = active_beams[np.argmax(surplus)]
                X[replace_idx, best_t] = 0
                remaining[replace_idx] += 1
            X[n, best_t] = 1
            remaining[n] -= 1
    return X
def compute_capacity(X, p_hat, H, M, B=BANDWIDTH, noise_power=NOISE_FLOOR):
    """根据照明模式计算实际容量"""
    N = H.shape[0]
    capacity = np.zeros(N)
    for t in range(M):
        active = np.where(X[:, t] == 1)[0]
        if len(active) == 0:
            continue
        for n in active:
            signal = p_hat[n] * H[n, n]**2
            interference = sum(p_hat[j] * H[j, n]**2 for j in active if j != n)
            sinr = signal / (interference + noise_power)
            capacity[n] += B * np.log2(1 + sinr)
    capacity /= M
    return capacity
