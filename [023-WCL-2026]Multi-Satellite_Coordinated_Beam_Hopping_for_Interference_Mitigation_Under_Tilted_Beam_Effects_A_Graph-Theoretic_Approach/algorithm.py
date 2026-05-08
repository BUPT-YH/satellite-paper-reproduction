"""
MCMF-TS-GC 算法主模块
两阶段算法：Stage 1 MCMF初始SCA + Stage 2 TS-GC联合优化
"""

import numpy as np
from mcmf import solve_mcmf_sca
from graph_coloring import head_coloring_fast


def build_interference_graph(J_s, s_assign):
    """
    根据SCA方案构建干扰图的邻接矩阵

    J_s: (S, C, C) 干扰指示器
    s_assign: (C,) 卫星-小区关联

    返回: adj (C, C) 邻接矩阵
    """
    C = len(s_assign)
    adj = np.zeros((C, C), dtype=int)
    for c in range(C):
        s_c = s_assign[c]
        for i in range(C):
            if i != c and J_s[s_c, c, i]:
                adj[c, i] = 1
                adj[i, c] = 1
    return adj


def mcmf_ts_gc(J_s, elev, Ls, T, delta_L=30, Nn=20, Nit=10,
               i_thr_low=-125.0, delta_i=1.0, tabu_len=15, seed=42):
    """
    完整的MCMF-TS-GC算法

    J_s: (S, C, C) 干扰指示器（在不同Ithr下的判断用当前Ithr重新计算）
    elev: (S, C) 仰角矩阵
    Ls: 每颗卫星最大同时激活波束数
    T: BH周期数
    delta_L: MCMF余量

    返回: (s_best, t_best, X) 最优SCA、BHSA和BH模式矩阵
    """
    rng = np.random.RandomState(seed)
    C = elev.shape[1]
    gc_iter = min(300, C) if C < 500 else min(100, C // 5)  # 大规模场景大幅减少GC迭代

    # ===== Stage 1: MCMF初始SCA =====
    s_star = solve_mcmf_sca(elev, Ls, T, delta_L)

    # ===== Stage 2: TS-GC联合优化 =====
    # 单调可行性测试：从低Ithr开始，逐步增加
    i_thr_dbw = i_thr_low
    s_best = s_star.copy()
    t_best = None

    max_outer = 50  # 最大Ithr测试次数
    for outer in range(max_outer):
        # 用当前Ithr构建干扰图
        adj = build_interference_graph(J_s, s_best)

        # 初始化禁忌列表
        tabu_list = []
        local_best_s = s_best.copy()
        local_best_t = None
        local_best_conf = np.inf

        for it in range(Nit):
            # 生成邻域解
            neighbors_s = []
            # 找当前干扰图的冲突边
            conflict_edges = _get_conflict_edges(adj)
            if len(conflict_edges) == 0:
                # 无冲突，当前解可行
                local_best_conf = 0
                colors = _assign_colors_no_conflict(adj, T, s_best, Ls, rng)
                local_best_t = colors
                break

            conflict_verts = set()
            for (u, v) in conflict_edges:
                conflict_verts.add(u)
                conflict_verts.add(v)
            conflict_verts = list(conflict_verts)

            for n_idx in range(Nn):
                s_new = s_best.copy()
                # 随机选择一个冲突顶点
                cr = conflict_verts[rng.randint(len(conflict_verts))]
                # 随机选择另一个冲突顶点作为vcon
                vcon_candidates = [v for (u, v) in conflict_edges if u == cr]
                vcon_candidates += [u for (u, v) in conflict_edges if v == cr]
                if not vcon_candidates:
                    continue
                vcon = vcon_candidates[rng.randint(len(vcon_candidates))]

                # 随机选择新卫星，要求不干扰vcon
                S_total = elev.shape[0]
                valid_sats = []
                for s in range(S_total):
                    if s != s_new[cr] and not J_s[s, cr, vcon]:
                        valid_sats.append(s)
                if valid_sats:
                    s_new[cr] = valid_sats[rng.randint(len(valid_sats))]

                # 检查是否在禁忌列表中
                is_tabu = False
                for tabu_s in tabu_list:
                    if np.array_equal(s_new, tabu_s):
                        is_tabu = True
                        break
                if is_tabu:
                    continue

                neighbors_s.append(s_new)

            if not neighbors_s:
                continue

            # 对每个邻域解，构建干扰图并用HEAD着色
            best_n_conf = np.inf
            best_n_idx = 0
            best_n_t = None

            for n_idx, s_n in enumerate(neighbors_s):
                adj_n = build_interference_graph(J_s, s_n)
                t_n, conf = head_coloring_fast(adj_n, T, s_n, Ls,
                                               max_iter=gc_iter,
                                               seed=seed + it * Nn + n_idx)
                if conf < best_n_conf:
                    best_n_conf = conf
                    best_n_idx = n_idx
                    best_n_t = t_n

            # 更新最优解
            if best_n_conf < local_best_conf:
                local_best_conf = best_n_conf
                local_best_s = neighbors_s[best_n_idx].copy()
                local_best_t = best_n_t.copy() if best_n_t is not None else None

            # 更新禁忌列表
            tabu_list.append(neighbors_s[best_n_idx].copy())
            if len(tabu_list) > tabu_len:
                tabu_list.pop(0)

        if local_best_conf == 0:
            # 找到可行解
            s_best = local_best_s
            t_best = local_best_t
            break
        else:
            # 增加Ithr继续搜索（实际上J_s已经预计算了，这里简化处理）
            # 在实际实现中，J_s会随Ithr变化，这里用当前J_s做更多迭代
            s_best = local_best_s
            t_best = local_best_t
            # 增加迭代次数再试
            adj = build_interference_graph(J_s, s_best)
            if t_best is None:
                t_best, _ = head_coloring_fast(adj, T, s_best, Ls,
                                                max_iter=min(C * 5, 5000), seed=seed + 1000)

    # 确保t_best有效
    if t_best is None:
        adj = build_interference_graph(J_s, s_best)
        t_best, _ = head_coloring_fast(adj, T, s_best, Ls,
                                        max_iter=min(C * 5, 5000), seed=seed + 2000)

    # 构建BH模式矩阵X
    X = np.zeros((C, T), dtype=int)
    for c in range(C):
        if 0 <= t_best[c] < T:
            X[c, t_best[c]] = s_best[c] + 1  # 用1-indexed表示卫星

    return s_best, t_best, X


def _get_conflict_edges(adj):
    """获取冲突边列表"""
    C = adj.shape[0]
    edges = []
    for i in range(C):
        for j in range(i + 1, C):
            if adj[i, j]:
                edges.append((i, j))
    return edges


def _assign_colors_no_conflict(adj, T, s_assign, Ls, rng):
    """无冲突时分配颜色"""
    C = adj.shape[0]
    colors = np.zeros(C, dtype=int)
    # 简单贪心分配，满足C2约束
    slot_count = {}  # (satellite, slot) -> count

    order = list(range(C))
    rng.shuffle(order)

    for c in order:
        s_c = s_assign[c]
        best_slot = 0
        for t in range(T):
            key = (s_c, t)
            if key not in slot_count or slot_count[key] < Ls:
                best_slot = t
                break
        colors[c] = best_slot
        key = (s_c, best_slot)
        slot_count[key] = slot_count.get(key, 0) + 1

    return colors
