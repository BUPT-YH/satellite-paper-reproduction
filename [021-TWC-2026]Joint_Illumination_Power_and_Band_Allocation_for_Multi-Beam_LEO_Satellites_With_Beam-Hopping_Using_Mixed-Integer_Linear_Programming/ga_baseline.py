"""
GA (Genetic Algorithm) 基线方法 — 简化 NSGA-II
基于 Table V 配置参数
"""
import numpy as np
from config import N_ILL, N_FREQ, M_NUM, PT, GA_CONFIG


def evaluate_solution(system, illum_seq, mcs_assign, bin_assign, beta=0.7):
    """
    评估一个解的 KPI (UC, EC, TTS)
    illum_seq: (NT,) 每个时隙的照射小区列表
    mcs_assign: (NU, NT) MCS 分配
    bin_assign: (NU, NT) 频率分块数分配
    """
    NC, NU, NT = system.NC, system.nu, system.NT
    R = system.R.copy()
    P_req = system.P_req
    D_table = system.D
    upc = system.upc

    gamma = np.zeros(NU)
    total_ec = np.zeros(NU)
    tts_counter = np.zeros(NC)
    max_tts = np.zeros(NC)

    for t in range(NT):
        gamma += R[:, t]

        illuminated = illum_seq[t]
        remaining_power = PT

        # 检查功率约束，按需分配
        for c in illuminated:
            users = np.where(upc[:, c] == 1)[0]
            for u in users:
                if gamma[u] <= 0 or remaining_power <= 0:
                    continue
                m = mcs_assign[u, t]
                n = bin_assign[u, t]
                if m < 0 or n <= 0:
                    continue
                p = P_req[m, n - 1]
                d = D_table[m, n - 1]
                if p > remaining_power:
                    # 降低 MCS
                    for mm in range(m, -1, -1):
                        pp = P_req[mm, n - 1]
                        if pp <= remaining_power:
                            p, d = pp, D_table[mm, n - 1]
                            break
                    else:
                        continue

                served = min(d, gamma[u])
                excess = max(0, d - gamma[u])
                gamma[u] -= served
                total_ec[u] += excess
                remaining_power -= p

            tts_counter[c] = 0

        for c in range(NC):
            if c not in illuminated:
                users = np.where(upc[:, c] == 1)[0]
                if any(gamma[u] > 0 for u in users):
                    tts_counter[c] += 1
                    max_tts[c] = max(max_tts[c], tts_counter[c])

    total_demand = np.sum(R)
    uc_pct = 100 * np.sum(gamma) / total_demand if total_demand > 0 else 0
    ec_pct = 100 * np.sum(total_ec) / total_demand if total_demand > 0 else 0
    avg_tts = np.mean(max_tts)

    return uc_pct, ec_pct, avg_tts


def create_individual(system, rng):
    """创建一个随机个体"""
    NC, NU, NT = system.NC, system.nu, system.NT
    illum_seq = []
    mcs_assign = np.full((NU, NT), -1, dtype=int)
    bin_assign = np.zeros((NU, NT), dtype=int)

    for t in range(NT):
        cells = rng.choice(NC, size=min(N_ILL, NC), replace=False)
        illum_seq.append(set(cells))
        for c in cells:
            users = np.where(system.upc[:, c] == 1)[0]
            for u in users:
                mcs_assign[u, t] = rng.randint(0, M_NUM)
                bin_assign[u, t] = rng.randint(1, N_FREQ + 1)

    return illum_seq, mcs_assign, bin_assign


def demand_based_individual(system, rng):
    """基于需求的个体（更优的初始解）"""
    NC, NU, NT = system.NC, system.nu, system.NT
    R = system.R.copy()
    P_req = system.P_req
    D_table = system.D
    upc = system.upc

    illum_seq = []
    mcs_assign = np.full((NU, NT), -1, dtype=int)
    bin_assign = np.zeros((NU, NT), dtype=int)
    gamma = np.zeros(NU)

    for t in range(NT):
        gamma += R[:, t]

        cell_demand = np.zeros(NC)
        for c in range(NC):
            users = np.where(upc[:, c] == 1)[0]
            cell_demand[c] = sum(gamma[u] for u in users)

        top_cells = np.argsort(cell_demand)[-N_ILL:]
        illum_seq.append(set(top_cells))

        remaining_power = PT
        for c in top_cells:
            users = np.where(upc[:, c] == 1)[0]
            for u in users:
                if gamma[u] <= 0:
                    continue
                # 选择满足需求的最小功率组合
                best = None
                for m in range(M_NUM):
                    for n in range(1, N_FREQ + 1):
                        d = D_table[m, n - 1]
                        p = P_req[m, n - 1]
                        if d >= gamma[u] * 0.8 and p <= remaining_power:
                            if best is None or p < best[1]:
                                best = (m, n, d, p)

                if best:
                    m, n, d, p = best
                    mcs_assign[u, t] = m
                    bin_assign[u, t] = n
                    gamma[u] -= min(d, gamma[u])
                    remaining_power -= p

    return illum_seq, mcs_assign, bin_assign


def crossover(p1, p2, system, rng):
    """均匀交叉"""
    NC, NU, NT = system.NC, system.nu, system.NT
    illum_seq = []
    mcs_assign = np.full((NU, NT), -1, dtype=int)
    bin_assign = np.zeros((NU, NT), dtype=int)

    for t in range(NT):
        if rng.random() < 0.5:
            illum_seq.append(p1[0][t])
            mcs_assign[:, t] = p1[1][:, t]
            bin_assign[:, t] = p1[2][:, t]
        else:
            illum_seq.append(p2[0][t])
            mcs_assign[:, t] = p2[1][:, t]
            bin_assign[:, t] = p2[2][:, t]

    return illum_seq, mcs_assign, bin_assign


def mutate(ind, system, rng):
    """变异操作"""
    NC, NU, NT = system.NC, system.nu, system.NT
    illum_seq = [set(s) for s in ind[0]]
    mcs_assign = ind[1].copy()
    bin_assign = ind[2].copy()

    # 变异照射序列
    if rng.random() < 0.3:
        t = rng.randint(0, NT)
        new_cells = set()
        for _ in range(N_ILL):
            new_cells.add(rng.randint(0, NC))
        illum_seq[t] = new_cells

    # 变异 MCS 和带宽分配
    for u in range(NU):
        if rng.random() < 0.1:
            t = rng.randint(0, NT)
            mcs_assign[u, t] = rng.randint(0, M_NUM)
            bin_assign[u, t] = rng.randint(1, N_FREQ + 1)

    return illum_seq, mcs_assign, bin_assign


def run_ga_baseline(system, beta=0.7, verbose=False):
    """运行 GA 优化"""
    # 根据场景大小调整 GA 参数
    if system.NC > 100:
        cfg = {"n_gen": 30, "pop_size": 40, "crossover_rate": 1.0,
               "mutation_rate": 0.15, "db_local_search_rate": 0.3}
    else:
        cfg = GA_CONFIG
    rng = np.random.RandomState(system.seed + 1000)

    # 初始化种群（混合随机和需求驱动）
    pop = []
    pop.append(demand_based_individual(system, rng))
    for _ in range(cfg["pop_size"] - 1):
        pop.append(create_individual(system, rng))

    # 评估初始种群
    fitness = []
    for ind in pop:
        uc, ec, tts = evaluate_solution(system, ind[0], ind[1], ind[2], beta)
        fitness.append((uc, ec, tts))

    best_obj = float('inf')
    best_ind = pop[0]
    w_uc = beta
    w_ec = (1 - beta) / 2
    w_tts = (1 - beta) / 2

    for gen in range(cfg["n_gen"]):
        # 计算目标函数值（加权和）
        objs = []
        for f in fitness:
            obj = w_uc * f[0] + w_ec * f[1] + w_tts * f[2]
            objs.append(obj)

        # 记录最优
        best_idx = np.argmin(objs)
        if objs[best_idx] < best_obj:
            best_obj = objs[best_idx]
            best_ind = (pop[best_idx][0], pop[best_idx][1].copy(), pop[best_idx][2].copy())

        # 锦标赛选择
        new_pop = [pop[best_idx]]  # 精英保留
        while len(new_pop) < cfg["pop_size"]:
            i1, i2 = rng.choice(cfg["pop_size"], size=2, replace=False)
            p1 = pop[i1] if objs[i1] < objs[i2] else pop[i2]

            i3, i4 = rng.choice(cfg["pop_size"], size=2, replace=False)
            p2 = pop[i3] if objs[i3] < objs[i4] else pop[i4]

            # 交叉
            child = crossover(p1, p2, system, rng)

            # 变异
            if rng.random() < cfg["mutation_rate"]:
                child = mutate(child, system, rng)

            # DB 局部搜索
            if rng.random() < cfg["db_local_search_rate"]:
                child = _db_local_search(child, system, rng)

            new_pop.append(child)

        pop = new_pop[:cfg["pop_size"]]
        fitness = []
        for ind in pop:
            uc, ec, tts = evaluate_solution(system, ind[0], ind[1], ind[2], beta)
            fitness.append((uc, ec, tts))

        if verbose and gen % 20 == 0:
            best_f = fitness[np.argmin([w_uc*f[0]+w_ec*f[1]+w_tts*f[2] for f in fitness])]
            print(f"  GA Gen {gen}: UC={best_f[0]:.1f}%, EC={best_f[1]:.1f}%, TTS={best_f[2]:.1f}")

    # 返回最优解的 KPI
    uc, ec, tts = evaluate_solution(system, best_ind[0], best_ind[1], best_ind[2], beta)
    return {"UC": uc, "EC": ec, "TTS": tts}


def _db_local_search(ind, system, rng):
    """基于需求的局部搜索"""
    NC, NU, NT = system.NC, system.nu, system.NT
    illum_seq = [set(s) for s in ind[0]]
    mcs_assign = ind[1].copy()
    bin_assign = ind[2].copy()
    R = system.R
    P_req = system.P_req
    D_table = system.D

    # 对部分时隙进行需求驱动的重分配
    t = rng.randint(0, NT)
    gamma = np.zeros(NU)
    for tt in range(t):
        gamma += R[:, tt]

    cell_demand = np.zeros(NC)
    for c in range(NC):
        users = np.where(system.upc[:, c] == 1)[0]
        cell_demand[c] = sum(gamma[u] for u in users)

    top_cells = np.argsort(cell_demand)[-N_ILL:]
    illum_seq[t] = set(top_cells)

    remaining_power = PT
    for c in top_cells:
        users = np.where(system.upc[:, c] == 1)[0]
        for u in users:
            if gamma[u] <= 0:
                continue
            for m in range(M_NUM):
                for n in range(1, N_FREQ + 1):
                    d = D_table[m, n - 1]
                    p = P_req[m, n - 1]
                    if d >= gamma[u] * 0.8 and p <= remaining_power:
                        mcs_assign[u, t] = m
                        bin_assign[u, t] = n
                        remaining_power -= p
                        break
                if mcs_assign[u, t] >= 0:
                    break

    return illum_seq, mcs_assign, bin_assign
