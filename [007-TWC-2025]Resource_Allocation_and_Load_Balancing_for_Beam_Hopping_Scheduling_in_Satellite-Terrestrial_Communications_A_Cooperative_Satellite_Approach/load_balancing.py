"""
负载均衡模块
实现基于ISL的卫星间负载均衡 (Algorithm 4)
修正: 使用覆盖信息正确分配流量
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


def load_balancing_sufficient(lambda_si_init, U_service, n_sat, n_cells_per_sat):
    """
    负载均衡 — 充足场景 (P3a)
    最小化平均等待时间 + ISL传输开销
    混合块逐次近似算法 (Algorithm 4, steps 1-7)
    lambda_si_init: 初始流量分配 (已按覆盖卫星数均分)
    """
    max_cells = n_cells_per_sat
    lambda_si = lambda_si_init.copy()
    mu_si = np.zeros((n_sat, max_cells))

    for s in range(n_sat):
        for i in range(max_cells):
            mu_si[s, i] = lambda_si[s, i] * 1.1 + cfg.delta_lb

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

        g_si = np.sqrt(np.maximum(lambda_si, 0) / np.maximum(mu_si, 1e-10))

        if np.max(np.abs(lambda_si - lambda_old)) < cfg.lb_tol:
            break

    return lambda_si


def load_balancing_overload(lambda_si_init, U_service, n_sat, n_cells_per_sat):
    """
    负载均衡 — 过载场景 (P3b)
    最小化最大负载比例 (均衡负载)
    """
    lambda_si = lambda_si_init.copy()
    total_service = U_service.sum(axis=1)

    for s in range(n_sat):
        for i in range(n_cells_per_sat):
            total_cap = total_service.sum()
            if total_cap > 0 and lambda_si[s, i] > 0:
                lambda_si[s, i] *= total_service[s] / (total_cap / n_sat)

    return lambda_si


def run_load_balancing(lambda_arrival, throughput_history, n_sat, n_cells,
                       n_cells_per_sat, cell_indices, n_beams, cell_to_sats=None):
    """
    执行负载均衡 (Algorithm 4)
    cell_indices: list of lists, cell_indices[s] = global cell indices covered by satellite s
    cell_to_sats: dict mapping global cell index -> list of covering satellite indices
    """
    U_service = compute_avg_service_rate(throughput_history, n_sat, n_cells_per_sat)

    # 初始化: 按覆盖卫星数均分流量 (关键修正)
    lambda_si = np.zeros((n_sat, n_cells_per_sat))

    for s in range(n_sat):
        cells = cell_indices[s] if s < len(cell_indices) else list(range(n_cells_per_sat))
        for local_i in range(n_cells_per_sat):
            if local_i < len(cells):
                cell_idx = cells[local_i] % n_cells
            else:
                cell_idx = local_i % n_cells

            # 按覆盖卫星数均分该小区流量
            if cell_to_sats and cell_idx in cell_to_sats:
                n_covering = len(cell_to_sats[cell_idx])
            else:
                n_covering = max(1, sum(1 for sat_cells in cell_indices
                                        if cell_idx in [c % n_cells for c in sat_cells]))

            lambda_si[s, local_i] = lambda_arrival[cell_idx] / max(n_covering, 1)

    if check_feasibility_p3a(lambda_arrival, U_service):
        return load_balancing_sufficient(lambda_si, U_service, n_sat, n_cells_per_sat)
    else:
        return load_balancing_overload(lambda_si, U_service, n_sat, n_cells_per_sat)
