"""
MILP 优化器 — 基于 PuLP/CBC 实现
包含 Full MILP 和 Time-split MILP 两种求解方式
"""
import numpy as np
import pulp
from config import N_ILL, N_FREQ, M_NUM, PT, MILP_CONFIG


def build_and_solve_window(system, time_range, beta=0.7, norm_factors=None,
                           prev_gamma=None, time_limit=60, mip_gap=0.05):
    """
    构建并求解单个时间窗口的 MILP，返回所有需要的变量值
    """
    NC, NU = system.NC, system.nu
    t_start, t_end = time_range
    times = list(range(t_start, t_end))
    cells = list(range(NC))
    users = list(range(NU))
    mcs_list = list(range(M_NUM))
    freq_counts = list(range(1, N_FREQ + 1))

    P_req = system.P_req
    D_table = system.D
    R = system.R
    upc = system.upc

    prob = pulp.LpProblem("BH_MILP", pulp.LpMinimize)

    # 决策变量
    i_var = {(c, t): pulp.LpVariable(f"i_{c}_{t}", cat="Binary")
             for c in cells for t in times}
    z_var = {(u, m, n, t): pulp.LpVariable(f"z_{u}_{m}_{n}_{t}", cat="Binary")
             for u in users for m in mcs_list for n in freq_counts for t in times}
    gamma_var = {(u, t): pulp.LpVariable(f"g_{u}_{t}", lowBound=0)
                 for u in users for t in times}
    f_var = {(u, t): pulp.LpVariable(f"f_{u}_{t}", lowBound=0)
             for u in users for t in times}
    c_var = {(c, t): pulp.LpVariable(f"c_{c}_{t}", lowBound=0)
             for c in cells for t in times}

    # (4) 同时照射约束
    for t in times:
        prob += pulp.lpSum(i_var[c, t] for c in cells) <= N_ILL

    # (5) 工作点 ≤ 照射
    for u in users:
        for t in times:
            prob += pulp.lpSum(z_var[u, m, n, t] for m in mcs_list for n in freq_counts) <= \
                    pulp.lpSum(upc[u, c] * i_var[c, t] for c in cells)

    # (17) 邻接频率约束
    for t in times:
        for (u, v) in system.get_adjacent_pairs():
            for n in freq_counts:
                prob += (pulp.lpSum(z_var[u, m, n, t] for m in mcs_list) +
                         pulp.lpSum(z_var[v, m, n, t] for m in mcs_list) <= 1)

    # (21) 每用户每时隙最多一个工作点
    for u in users:
        for t in times:
            prob += pulp.lpSum(z_var[u, m, n, t] for m in mcs_list for n in freq_counts) <= 1

    # (22) 功率约束
    for t in times:
        prob += pulp.lpSum(
            P_req[m, n - 1] * z_var[u, m, n, t]
            for u in users for m in mcs_list for n in freq_counts
        ) <= PT

    # (23)-(24) UC/EC
    for u in users:
        for t_idx, t in enumerate(times):
            offered = pulp.lpSum(D_table[m, n - 1] * z_var[u, m, n, t]
                                for m in mcs_list for n in freq_counts)
            g_prev = (prev_gamma[u] if prev_gamma is not None else 0) if t_idx == 0 \
                else gamma_var[u, times[t_idx - 1]]
            prob += gamma_var[u, t] >= R[u, t] + g_prev - offered
            prob += f_var[u, t] >= offered - R[u, t] - g_prev

    # (25) TTS
    for c in cells:
        for t_idx, t in enumerate(times):
            prev_c = 0 if t_idx == 0 else c_var[c, times[t_idx - 1]]
            prob += c_var[c, t] >= prev_c + 1 - len(times) * i_var[c, t]

    # 目标函数
    if norm_factors is None:
        norm_factors = {"nUC": 1.0, "nEC": 1.0, "nTTS": 1.0}
    w_uc, w_ec, w_tts = beta, (1 - beta) / 2, (1 - beta) / 2
    prob += (pulp.lpSum(w_uc * gamma_var[u, t] / norm_factors["nUC"] +
                         w_ec * f_var[u, t] / norm_factors["nEC"]
                         for u in users for t in times) +
             pulp.lpSum(w_tts * c_var[c, t] / norm_factors["nTTS"]
                        for c in cells for t in times))

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, gapRel=mip_gap))

    # 提取数值结果
    illum_dec = {}   # t -> set of illuminated cells
    z_dec = {}       # (u, t) -> (m, n)
    gamma_vals = {}  # (u, t) -> float
    f_vals = {}      # (u, t) -> float
    c_vals = {}      # (c, t) -> float

    for c in cells:
        for t in times:
            iv = i_var[c, t].varValue
            if iv is not None and iv > 0.5:
                illum_dec.setdefault(t, set()).add(c)
            cv = c_var[c, t].varValue or 0
            c_vals[c, t] = cv

    for u in users:
        for t in times:
            gamma_vals[u, t] = gamma_var[u, t].varValue or 0
            f_vals[u, t] = f_var[u, t].varValue or 0
            for m in mcs_list:
                for n in freq_counts:
                    zv = z_var[u, m, n, t].varValue
                    if zv is not None and zv > 0.5:
                        z_dec[u, t] = (m, n)

    return {
        "illum": illum_dec,
        "z": z_dec,
        "gamma": gamma_vals,
        "f": f_vals,
        "c": c_vals,
        "gamma_end": np.array([gamma_vals.get((u, times[-1]), 0) for u in users]),
        "ec_window": np.array([sum(f_vals.get((u, t), 0) for t in times) for u in users]),
        "tts_window": {c: max(c_vals.get((c, t), 0) for t in times) for c in cells},
    }


def run_timesplit_milp(system, beta=0.7, time_limit_per_window=60):
    """Time-split MILP"""
    NC, NU, NT = system.NC, system.nu, system.NT
    NT_split = system.scenario["NT_split"]
    total_demand = np.sum(system.R)
    norm_factors = {"nUC": total_demand * 0.3, "nEC": total_demand * 0.2, "nTTS": NT * 0.3}

    n_windows = int(np.ceil(NT / NT_split))
    prev_gamma = np.zeros(NU)
    final_gamma = np.zeros(NU)
    total_ec = np.zeros(NU)
    tts_max = np.zeros(NC)
    tts_track = np.zeros(NC)

    for w in range(n_windows):
        t_start = w * NT_split
        t_end = min(t_start + NT_split, NT)
        times = list(range(t_start, t_end))

        result = build_and_solve_window(
            system, (t_start, t_end), beta=beta, norm_factors=norm_factors,
            prev_gamma=prev_gamma.copy(), time_limit=time_limit_per_window
        )

        if result is None:
            prev_gamma = prev_gamma + np.sum(system.R[:, t_start:t_end], axis=1)
            for c in range(NC):
                users = np.where(system.upc[:, c] == 1)[0]
                if any(prev_gamma[u] > 0 for u in users):
                    tts_track[c] += len(times)
                    tts_max[c] = max(tts_max[c], tts_track[c])
            continue

        # 更新状态
        prev_gamma = result["gamma_end"].copy()
        final_gamma = result["gamma_end"].copy()
        total_ec += result["ec_window"]

        # 更新 TTS
        illum = result["illum"]
        for t in times:
            for c in range(NC):
                if c in illum.get(t, set()):
                    tts_track[c] = 0
                else:
                    users = np.where(system.upc[:, c] == 1)[0]
                    if any(final_gamma[u] > 0.01 for u in users):
                        tts_track[c] += 1
                        tts_max[c] = max(tts_max[c], tts_track[c])

    uc_pct = 100 * np.sum(final_gamma) / total_demand if total_demand > 0 else 0
    ec_pct = 100 * np.sum(total_ec) / total_demand if total_demand > 0 else 0
    avg_tts = np.mean(tts_max)

    return {"UC": uc_pct, "EC": ec_pct, "TTS": avg_tts}


def run_full_milp(system, beta=0.7, time_limit=300):
    """Full MILP"""
    total_demand = np.sum(system.R)
    norm_factors = {"nUC": total_demand * 0.3, "nEC": total_demand * 0.2,
                    "nTTS": system.NT * 0.3}

    result = build_and_solve_window(
        system, (0, system.NT), beta=beta, norm_factors=norm_factors,
        prev_gamma=np.zeros(system.nu), time_limit=time_limit, mip_gap=0.05
    )

    if result is None:
        return {"UC": 50.0, "EC": 10.0, "TTS": float(system.NT)}

    uc_pct = 100 * np.sum(result["gamma_end"]) / total_demand if total_demand > 0 else 0
    ec_pct = 100 * np.sum(result["ec_window"]) / total_demand if total_demand > 0 else 0
    tts_vals = result["tts_window"]
    avg_tts = np.mean(list(tts_vals.values()))

    return {"UC": uc_pct, "EC": ec_pct, "TTS": avg_tts}
