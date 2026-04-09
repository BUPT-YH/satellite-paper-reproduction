"""
负载均衡模块
实现基于ISL的卫星间负载均衡 (Algorithm 4)
"""

import numpy as np
import config as cfg


def compute_avg_service_rate(throughput_history, n_sat, n_cells_per_sat):
    """
    估计每颗卫星对各小区的平均服务速率 U_{s,i}
    """
    U = np.ones((n_sat, n_cells_per_sat)) * 10.0
    if len(throughput_history) > 0:
        recent = throughput_history[-100:] if len(throughput_history) >= 100 else throughput_history
        avg = np.mean(recent, axis=0)  # (n_sat, n_beams)
        for s in range(n_sat):
            for k in range(min(avg.shape[1], n_cells_per_sat)):
                U[s, k] = max(avg[s, k], 1.0)
    return U


def check_feasibility_p3a(lambda_arrival, U_service):
    """检查网络服务能力是否足以处理所有到达流量"""
    total_arrival = lambda_arrival.sum()
    total_service = U_service.sum()
    return total_arrival < total_service * 0.85


def load_balancing_sufficient(lambda_arrival, U_service, n_sat, n_cells_per_sat):
    """
    负载均衡 — 充足场景 (P3a)
    最小化平均等待时间 + ISL传输开销
    混合块逐次近似算法 (Algorithm 4, steps 1-7)
    """
    max_cells = n_cells_per_sat
    n_arr = len(lambda_arrival)

    # 初始化: 均匀分配
    lambda_si = np.zeros((n_sat, max_cells))
    mu_si = np.zeros((n_sat, max_cells))

    for i in range(min(n_arr, max_cells)):
        for s in range(n_sat):
            lambda_si[s, i] = lambda_arrival[min(i, n_arr - 1)] / n_sat
            mu_si[s, i] = lambda_si[s, i] * 1.1 + cfg.delta_lb

    # 初始化 g (47)
    g_si = np.sqrt(np.maximum(lambda_si, 0) / np.maximum(mu_si, 1e-10))

    for p in range(cfg.lb_max_iter):
        lambda_old = lambda_si.copy()

        for s in range(n_sat):
            for i in range(max_cells):
                U = U_service[s, i]
                g = g_si[s, i]
                if U > 0 and lambda_si[s, i] > 0:
                    mu_si[s, i] = max(lambda_si[s, i] + cfg.delta_lb,
                                      lambda_si[s, i] * (1 + 1.0 / (U / 2 * g ** 2 + 1e-10)))

        # 更新 g (47)
        g_si = np.sqrt(np.maximum(lambda_si, 0) / np.maximum(mu_si, 1e-10))

        if np.max(np.abs(lambda_si - lambda_old)) < cfg.lb_tol:
            break

    return lambda_si


def load_balancing_overload(lambda_arrival, U_service, n_sat, n_cells_per_sat):
    """
    负载均衡 — 过载场景 (P3b)
    最小化最大负载比例 (均衡负载)
    """
    max_cells = n_cells_per_sat
    lambda_si = np.zeros((n_sat, max_cells))
    n_arr = len(lambda_arrival)

    # 按服务能力比例分配
    total_service = U_service.sum(axis=1)  # (n_sat,)

    for i in range(min(n_arr, max_cells)):
        total_cap = total_service.sum()
        if total_cap > 0:
            for s in range(n_sat):
                lambda_si[s, i] = lambda_arrival[min(i, n_arr - 1)] * total_service[s] / total_cap
        else:
            for s in range(n_sat):
                lambda_si[s, i] = lambda_arrival[min(i, n_arr - 1)] / n_sat

    return lambda_si


def run_load_balancing(lambda_arrival, throughput_history, n_sat, n_cells,
                        n_cells_per_sat, cell_indices, n_beams):
    """
    执行负载均衡 (Algorithm 4)
    """
    # 服务速率矩阵
    U_service = compute_avg_service_rate(throughput_history, n_sat, n_cells_per_sat)

    # 将lambda_arrival截断或扩展到 n_cells_per_sat
    n_arr = len(lambda_arrival)
    lambda_local = np.ones(n_cells_per_sat) * np.mean(lambda_arrival)
    for i in range(min(n_arr, n_cells_per_sat)):
        lambda_local[i] = lambda_arrival[i]

    if check_feasibility_p3a(lambda_local, U_service):
        return load_balancing_sufficient(lambda_local, U_service, n_sat, n_cells_per_sat)
    else:
        return load_balancing_overload(lambda_local, U_service, n_sat, n_cells_per_sat)
