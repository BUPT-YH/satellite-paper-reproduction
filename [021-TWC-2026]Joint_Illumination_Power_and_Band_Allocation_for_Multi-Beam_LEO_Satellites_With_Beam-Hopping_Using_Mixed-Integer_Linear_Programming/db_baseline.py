"""
DB (Deterministic Baseline) 基线方法
基于需求的确定性照射调度：每个时隙选择需求最高的小区照射
"""
import numpy as np
from config import N_ILL, N_FREQ, M_NUM, PT


def run_db_baseline(system, beta=0.7):
    """
    需求驱动的确定性基线方法
    每个时隙：选择未服务需求最高的 N_ILL 个小区照射，
    分配功率和带宽以匹配需求
    """
    NC = system.NC
    NU = system.nu
    NT = system.NT
    P_req = system.P_req
    D_table = system.D
    R = system.R.copy()
    upc = system.upc

    # KPI 跟踪
    gamma = np.zeros(NU)  # 累积未服务容量
    total_uc = np.zeros(NU)
    total_ec = np.zeros(NU)
    tts_counter = np.zeros(NC)  # 每个小区的 TTS 计数器
    max_tts = np.zeros(NC)

    for t in range(NT):
        # 累加需求
        gamma += R[:, t]

        # 选择需求最高的小区
        cell_demand = np.zeros(NC)
        for c in range(NC):
            users = np.where(upc[:, c] == 1)[0]
            cell_demand[c] = sum(gamma[u] for u in users)

        # 排序选择 top N_ILL
        illuminated = np.argsort(cell_demand)[-N_ILL:]

        # 为被照射小区的用户分配资源
        remaining_power = PT

        for c in illuminated:
            users = np.where(upc[:, c] == 1)[0]
            if len(users) == 0:
                continue

            # 按需求排序用户
            user_demand = [(u, gamma[u]) for u in users if gamma[u] > 0]
            user_demand.sort(key=lambda x: -x[1])

            for u, dem in user_demand:
                if dem <= 0 or remaining_power <= 0:
                    break

                # 选择最优 (mcs, n_bins) 组合：满足需求且功率最小的
                best_m, best_n, best_d, best_p = None, None, None, None
                for m in range(M_NUM):
                    for n in range(N_FREQ):
                        d = D_table[m, n]
                        p = P_req[m, n]
                        if d >= dem * 0.8 and p <= remaining_power:
                            if best_p is None or p < best_p:
                                best_m, best_n = m, n
                                best_d, best_p = d, p

                if best_m is not None:
                    # 分配资源
                    served = min(best_d, gamma[u])
                    excess = max(0, best_d - gamma[u])
                    gamma[u] -= served
                    total_uc[u] += max(0, gamma[u])
                    total_ec[u] += excess
                    remaining_power -= best_p
                else:
                    # 尝试分配最便宜的资源
                    for m in range(M_NUM):
                        n = 0
                        p = P_req[m, n]
                        d = D_table[m, n]
                        if p <= remaining_power and d > 0:
                            served = min(d, gamma[u])
                            excess = max(0, d - gamma[u])
                            gamma[u] -= served
                            total_ec[u] += excess
                            remaining_power -= p
                            break

            # 重置 TTS
            tts_counter[c] = 0

        # 未照射小区 TTS 递增
        for c in range(NC):
            if c not in illuminated:
                users = np.where(upc[:, c] == 1)[0]
                if any(gamma[u] > 0 for u in users):
                    tts_counter[c] += 1
                    max_tts[c] = max(max_tts[c], tts_counter[c])

    # 计算最终 KPI
    total_demand = np.sum(R)
    uc_pct = 100 * np.sum(gamma) / total_demand if total_demand > 0 else 0
    ec_pct = 100 * np.sum(total_ec) / total_demand if total_demand > 0 else 0
    avg_tts = np.mean(max_tts)

    return {"UC": uc_pct, "EC": ec_pct, "TTS": avg_tts}
