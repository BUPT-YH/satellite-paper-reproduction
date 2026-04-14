"""
阶段1: 逆矩阵优化算法 (Algorithm 1)
基于平均场理论解耦功率和波束激活概率
通过 SCA (Successive Convex Approximation) 迭代求解
"""
import numpy as np
from config import BANDWIDTH, CONVERGENCE_THRESHOLD, K_ACTIVE_DEFAULT


def g_function(rho_n, d_n, B=BANDWIDTH):
    """
    辅助函数 g(rho_n) = rho_n * (2^(d_n/(B*rho_n)) - 1)
    论文式 (6d) 中定义
    """
    if rho_n <= 0:
        return np.inf
    sinr_req = 2 ** (d_n / (B * rho_n)) - 1
    return rho_n * sinr_req


def g_vector(rho, d, B=BANDWIDTH):
    """计算向量 g(rho)"""
    return np.array([g_function(rho_n, d_n, B) for rho_n, d_n in zip(rho, d)])


def compute_power_from_rho(rho, A, b, d, B=BANDWIDTH):
    """
    根据激活概率 rho 计算最优功率 p
    p = Diag(rho^{-1}) (I - G A)^{-1} G b
    论文式 (7b)
    """
    N = len(rho)
    G = np.diag(g_vector(rho, d, B))
    GA = G @ A
    spectral_radius = np.max(np.abs(np.linalg.eigvals(GA)))
    if spectral_radius >= 1:
        return None, spectral_radius
    try:
        inv_matrix = np.linalg.inv(np.eye(N) - GA)
        p = np.diag(1.0 / rho) @ inv_matrix @ G @ b
    except np.linalg.LinAlgError:
        return None, spectral_radius
    return p, spectral_radius
def compute_energy(rho, p):
    """计算总能耗 rho^T p"""
    return np.dot(rho, p)
def objective_function(rho, A, b, d, B=BANDWIDTH):
    """
    目标函数 L(rho) = 1^T (I - G(rho)A)^{-1} G(rho) b
    论文 P2 中的目标函数
    """
    N = len(rho)
    G = np.diag(g_vector(rho, d, B))
    GA = G @ A
    spectral_radius = np.max(np.abs(np.linalg.eigvals(GA)))
    if spectral_radius >= 1.0:
        return np.inf
    try:
        inv_matrix = np.linalg.inv(np.eye(N) - GA)
        energy = np.sum(inv_matrix @ G @ b)
    except np.linalg.LinAlgError:
        return np.inf
    return energy


def algorithm1_solve(A, b, d, K=K_ACTIVE_DEFAULT, B=BANDWIDTH,
                     max_iter=100, verbose=False):
    """
    Algorithm 1: 逆矩阵优化
    通过 SCA (Successive Convex Approximation) 迭代求解 P3
    """
    N = len(d)
    rho = np.ones(N) * K / N
 # rho^T 1 = K
    rho = np.clip(rho, 1e-6, 1.0)

    energy_history = []
    rho_history = [rho.copy()]

    for iteration in range(max_iter):
        # 计算当前能耗
        p_curr, sr = compute_power_from_rho(rho, A, b, d, B)

        if p_curr is None:
            rho = np.minimum(rho * 1.1, 1.0)
            rho = project_to_simplex(rho, K)
            energy_history.append(np.nan)
            continue

        energy = compute_energy(rho, p_curr)
        energy_history.append(energy)
        # 梯度下降步
        grad = numerical_gradient(rho, A, b, d, B)
        step_size = 0.05
        rho_new = rho - step_size * grad

        # 投影
        rho_new = np.clip(rho_new, 1e-6, 1.0)
        rho_new = project_to_simplex(rho_new, K)

        # 磟半径条件
        for _ in range(10):
            G_new = np.diag(g_vector(rho_new, d, B))
            sr_new = np.max(np.abs(np.linalg.eigvals(G_new @ A)))
            if sr_new < 0.99:
                break
            rho_new = np.minimum(rho_new * 1.05, 1.0)
            rho_new = project_to_simplex(rho_new, K)

        diff = np.linalg.norm(rho_new - rho) / np.sqrt(N)
        rho_history.append(rho_new.copy())
        rho = rho_new
        if diff < CONVERGENCE_THRESHOLD:
            break
    # 最终结果
    p_opt, _ = compute_power_from_rho(rho, A, b, d, B)
    return rho, p_opt, energy_history


def project_to_simplex(rho, K):
    """将 rho 投影到约束 {rho: 0 < rho <= 1, rho^T 1 = K}"""
    rho = np.clip(rho, 1e-6, 1.0)
    total = np.sum(rho)
    if total > 0:
        rho = rho * K / total
    rho = np.clip(rho, 1e-6, 1.0)
    return rho
def numerical_gradient(rho, A, b, d, B, eps=1e-5):
    """数值梯度"""
    N = len(rho)
    grad = np.zeros(N)
    f0 = objective_function(rho, A, b, d, B)
    for i in range(N):
        rho_plus = rho.copy()
        rho_plus[i] += eps
        f_plus = objective_function(rho_plus, A, b, d, B)
        grad[i] = (f_plus - f0) / eps
    return grad
