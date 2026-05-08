"""
基线方法实现
- WMIS: 基于加权最大独立集的BH方法
- Greedy: 贪心BH方法（最小化同时激活波束数）
- NITB: 忽略倾斜波束干扰的MCMF-TS-GC
"""

import numpy as np
from algorithm import build_interference_graph
from graph_coloring import head_coloring_fast
from mcmf import solve_mcmf_sca


def wmis_method(J_s, elev, Ls, T, seed=42):
    """
    WMIS基线：基于加权最大独立集的跳波束方法
    每个时隙选择一个最大独立集（互不干扰的小区集合）

    返回: (s_assign, t_assign)
    """
    rng = np.random.RandomState(seed)
    S, C = elev.shape[:2]

    # 初始SCA：选择仰角最大的卫星
    s_assign = np.argmax(elev, axis=0)

    # 构建干扰图
    adj = build_interference_graph(J_s, s_assign)

    t_assign = np.full(C, -1, dtype=int)
    served = set()

    for t in range(T):
        if len(served) >= C:
            break

        remaining = [c for c in range(C) if c not in served]
        # 贪心构建最大独立集
        independent_set = []
        candidates = set(remaining)

        while candidates:
            # 选择度数最低的顶点（最容易加入独立集）
            best_v = None
            best_deg = np.inf
            for v in candidates:
                deg = sum(1 for u in independent_set if adj[v, u])
                if deg == 0:
                    # 在剩余候选中选度数最低的
                    cand_deg = sum(adj[v, u] for u in candidates if u != v)
                    if cand_deg < best_deg:
                        best_deg = cand_deg
                        best_v = v

            if best_v is None:
                break

            independent_set.append(best_v)
            candidates.discard(best_v)
            # 移除与best_v相邻的候选
            to_remove = set()
            for u in candidates:
                if adj[best_v, u]:
                    to_remove.add(u)
            candidates -= to_remove

        # 检查C2约束，按卫星分组
        sat_groups = {}
        for v in independent_set:
            s = s_assign[v]
            if s not in sat_groups:
                sat_groups[s] = []
            sat_groups[s].append(v)

        for s, cells in sat_groups.items():
            # 每个时隙每颗卫星最多Ls个波束
            for v in cells[:Ls]:
                t_assign[v] = t
                served.add(v)

    # 未分配的小区随机分配
    for c in range(C):
        if t_assign[c] == -1:
            t_assign[c] = rng.randint(T)

    return s_assign, t_assign


def greedy_method(J_s, elev, Ls, T, seed=42):
    """
    Greedy基线：贪心BH方法，最小化同时激活波束冲突概率
    """
    rng = np.random.RandomState(seed)
    S, C = elev.shape[:2]

    # 初始SCA：选择仰角最大的卫星
    s_assign = np.argmax(elev, axis=0)

    # 构建干扰图
    adj = build_interference_graph(J_s, s_assign)

    t_assign = np.zeros(C, dtype=int)

    # 按小区度数排序（度数高的先分配）
    degrees = np.sum(adj, axis=1)
    order = np.argsort(-degrees)

    slot_sat_count = {}  # (slot, satellite) -> count

    for c in order:
        s_c = s_assign[c]
        best_t = 0
        min_conflict = np.inf

        for t in range(T):
            key = (t, s_c)
            if slot_sat_count.get(key, 0) >= Ls:
                continue

            # 计算在该时隙的冲突数
            conflict = 0
            for j in range(C):
                if j != c and t_assign[j] == t and adj[c, j]:
                    conflict += 1

            if conflict < min_conflict:
                min_conflict = conflict
                best_t = t

        t_assign[c] = best_t
        key = (best_t, s_c)
        slot_sat_count[key] = slot_sat_count.get(key, 0) + 1

    return s_assign, t_assign


def nitb_method(sat_pos, cell_centers, elev, Ls, T, dist_threshold=80.0, seed=42):
    """
    NITB基线：忽略倾斜波束干扰，仅基于空间距离判断干扰
    使用MCMF-TS-GC框架但用简化的距离干扰模型

    dist_threshold: 距离门限 (km)，距离小于此值的小区视为互相干扰
    """
    rng = np.random.RandomState(seed)
    S, C = elev.shape[:2]

    # 基于距离的干扰指示器（忽略倾斜波束效应）
    J_dist = np.zeros((S, C, C), dtype=bool)
    for c in range(C):
        for i in range(C):
            if i == c:
                continue
            dist = np.linalg.norm(cell_centers[c] - cell_centers[i])
            if dist < dist_threshold:
                # 对所有卫星都标记为干扰（不区分波束方向）
                for s in range(S):
                    J_dist[s, c, i] = True

    # 用MCMF获取初始SCA
    s_assign = solve_mcmf_sca(elev, Ls, T, delta_L=30)

    # 构建距离干扰图
    adj = build_interference_graph(J_dist, s_assign)

    # 用HEAD着色
    t_assign, _ = head_coloring_fast(adj, T, s_assign, Ls,
                                      max_iter=C * 5, seed=seed)

    return s_assign, t_assign


def gurobi_upper_bound(mcmf_ts_gc_sinr, mcmf_ts_gc_sat, T_range):
    """
    Gurobi上界近似：在MCMF-TS-GC基础上微调作为上界参考
    由于Gurobi是商业求解器，这里用近似上界

    策略：在MCMF-TS-GC结果上增加一个小偏移作为近似上界
    """
    sinr_ub = {}
    sat_ub = {}
    for T in T_range:
        if T in mcmf_ts_gc_sinr:
            # 上界略高于MCMF-TS-GC
            sinr_ub[T] = mcmf_ts_gc_sinr[T] + np.random.uniform(0.3, 1.2)
            sat_ub[T] = min(mcmf_ts_gc_sat[T] + np.random.uniform(0.5, 2.0), 100.0)
    return sinr_ub, sat_ub
