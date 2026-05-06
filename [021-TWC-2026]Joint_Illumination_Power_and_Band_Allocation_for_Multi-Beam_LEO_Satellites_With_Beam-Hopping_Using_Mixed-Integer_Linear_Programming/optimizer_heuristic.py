"""
优化方法 — 改进的贪心策略，逼近 MILP 性能
核心思路：每时隙选择最优照射模式 + 最优功率/带宽分配
"""
import numpy as np
from config import N_ILL, N_FREQ, M_NUM, PT


def allocate_resources(system, users_list, t, gamma, power_budget, beta=0.7):
    """
    给定用户列表，在功率预算内分配资源
    策略: 按需求优先级，为每个用户找到满足需求的最小功率分配
    """
    P_req = system.P_req
    D_table = system.D
    R = system.R

    assignments = {}  # u -> (m, n, d, p)
    remaining = power_budget

    # 按未服务需求降序
    user_demand = [(u, gamma[u] + R[u, t]) for u in users_list
                   if gamma[u] + R[u, t] > 0]
    user_demand.sort(key=lambda x: -x[1])

    for u, demand in user_demand:
        if remaining <= 0 or demand <= 0:
            break

        # 找到满足需求的最小功率方案（节省功率给其他用户）
        best = None  # (m, n, d, p, excess)
        for m in range(M_NUM):
            for n in range(1, N_FREQ + 1):
                d = D_table[m, n - 1]
                p = P_req[m, n - 1]
                if p > remaining:
                    continue
                if d < demand * 0.5:
                    continue  # 容量太低，不值得分配

                excess = max(0, d - demand)
                # beta 高 → 接受更多 excess; beta 低 → 避免过量
                max_excess = demand * (3 - 2 * beta)  # beta=1: excess≤demand; beta=0: excess≤3*demand
                if excess > max_excess and demand > 0:
                    continue

                if best is None or p < best[3]:
                    best = (m, n, d, p, excess)

        if best is not None:
            m, n, d, p, excess = best
            assignments[u] = (m, n, d, p)
            remaining -= p

    # 如果还有剩余功率，尝试为未分配的用户找任意可行方案
    for u, demand in user_demand:
        if u in assignments or remaining <= 0 or demand <= 0:
            continue
        for m in range(M_NUM):
            for n in range(1, N_FREQ + 1):
                d = D_table[m, n - 1]
                p = P_req[m, n - 1]
                if p <= remaining and d > 0:
                    assignments[u] = (m, n, d, p)
                    remaining -= p
                    break
            if u in assignments:
                break

    return assignments


def simulate_step(system, illuminated_cells, t, gamma, beta=0.7):
    """模拟一个时隙，返回更新后的 gamma, ec"""
    upc = system.upc
    D_table = system.D
    all_users = []
    for c in illuminated_cells:
        all_users.extend(np.where(upc[:, c] == 1)[0].tolist())

    assignments = allocate_resources(system, all_users, t, gamma, PT, beta)

    ec_step = np.zeros(system.nu)
    for u, (m, n, d, p) in assignments.items():
        served = min(d, gamma[u])
        excess = max(0, d - gamma[u])
        gamma[u] -= served
        ec_step[u] = excess

    return gamma, ec_step


def run_optimized_method(system, beta=0.7):
    """
    改进的贪心优化方法
    每时隙: 尝试多种照射策略，选择最优
    """
    NC, NU, NT = system.NC, system.nu, system.NT
    R = system.R
    upc = system.upc

    gamma = np.zeros(NU)
    total_ec = np.zeros(NU)
    tts_track = np.zeros(NC)
    tts_max = np.zeros(NC)

    for t in range(NT):
        gamma += R[:, t]

        # 计算小区需求
        cell_demand = np.zeros(NC)
        for c in range(NC):
            users = np.where(upc[:, c] == 1)[0]
            cell_demand[c] = sum(gamma[u] for u in users)

        # 生成多个候选照射策略
        candidates = []

        # 策略1: 纯需求驱动
        cand1 = set(np.argsort(cell_demand)[-N_ILL:])
        candidates.append(cand1)

        # 策略2: 需求 + TTS 加权
        tts_score = cell_demand.copy()
        for c in range(NC):
            if tts_track[c] > 0 and cell_demand[c] > 0:
                tts_score[c] += tts_track[c] * (1 - beta) * 10
        cand2 = set(np.argsort(tts_score)[-N_ILL:])
        candidates.append(cand2)

        # 策略3: 替换最弱的一个小区（局部搜索）
        if len(cand1) > 0:
            weakest = min(cand1, key=lambda c: cell_demand[c])
            remaining = [c for c in range(NC) if c not in cand1 and cell_demand[c] > 0]
            if remaining:
                best_swap = max(remaining, key=lambda c: cell_demand[c] + tts_track[c] * 2)
                cand3 = (cand1 - {weakest}) | {best_swap}
                candidates.append(cand3)

        # 评估每个策略，选择使 gamma 最小的
        best_illum = cand1
        best_gamma_sum = float('inf')

        for cand in candidates:
            g_test = gamma.copy()
            g_test, _ = simulate_step(system, cand, t, g_test, beta)
            g_sum = np.sum(g_test)
            if g_sum < best_gamma_sum:
                best_gamma_sum = g_sum
                best_illum = cand

        # 执行最优策略
        gamma, ec_step = simulate_step(system, best_illum, t, gamma, beta)
        total_ec += ec_step

        # 更新 TTS
        for c in range(NC):
            users = np.where(upc[:, c] == 1)[0]
            if c in best_illum:
                tts_track[c] = 0
            elif any(gamma[u] > 0.01 for u in users):
                tts_track[c] += 1
                tts_max[c] = max(tts_max[c], tts_track[c])

    total_demand = np.sum(R)
    uc_pct = 100 * np.sum(gamma) / total_demand if total_demand > 0 else 0
    ec_pct = 100 * np.sum(total_ec) / total_demand if total_demand > 0 else 0
    avg_tts = np.mean(tts_max)

    return {"UC": uc_pct, "EC": ec_pct, "TTS": avg_tts}
