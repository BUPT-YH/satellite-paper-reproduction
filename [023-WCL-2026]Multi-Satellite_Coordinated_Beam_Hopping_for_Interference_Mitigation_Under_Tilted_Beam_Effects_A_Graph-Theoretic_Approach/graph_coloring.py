"""
图着色算法 (HEAD算法简化版)
用于在给定干扰图上分配BH时隙，最小化冲突边数
同时满足卫星波束资源约束 C2
"""

import numpy as np


def head_coloring(adj_matrix, n_colors, s_assign, Ls, max_iter=5000, seed=42):
    """
    HEAD算法简化版：在干扰图上着色（分配BH时隙）

    adj_matrix: (C, C) 干扰邻接矩阵，adj[i,j]=1表示小区i和j不能同时服务
    n_colors: 可用颜色数 (=T，BH周期数)
    s_assign: (C,) 卫星-小区关联
    Ls: 每颗卫星最大同时激活波束数
    max_iter: 最大迭代次数

    返回: (colors, n_conflicts)
      colors: (C,) 颜色分配（0到n_colors-1）
      n_conflicts: 冲突边数
    """
    rng = np.random.RandomState(seed)
    C = adj_matrix.shape[0]
    S = len(np.unique(s_assign))

    # 初始化：贪心着色（DSATUR策略）
    colors = _dsatur_init(adj_matrix, n_colors, rng)

    # 计算初始冲突
    best_colors = colors.copy()
    best_conflicts = _count_conflicts(colors, adj_matrix)

    # 局部搜索迭代
    tabu = {}  # 顶点 -> (颜色, 过期迭代)
    for iteration in range(max_iter):
        if best_conflicts == 0:
            break

        # 找所有冲突顶点
        conflict_verts = _get_conflict_vertices(colors, adj_matrix)
        if len(conflict_verts) == 0:
            break

        # 随机选择一个冲突顶点
        v = rng.choice(conflict_verts)

        # 尝试每个颜色，选择使冲突减少最多的
        best_c = colors[v]
        best_c_conflicts = best_conflicts
        for c in range(n_colors):
            if c == colors[v]:
                continue
            # 检查禁忌
            if v in tabu and tabu[v] == (c, ):
                continue

            colors[v] = c
            # 检查C2约束（波束资源限制）
            if _check_c2(colors, s_assign, Ls, n_colors):
                conf = _count_conflicts(colors, adj_matrix)
                if conf < best_c_conflicts:
                    best_c_conflicts = conf
                    best_c = c
            colors[v] = colors[v] if best_c == colors[v] else colors[v]

            # 恢复原色以便继续尝试
            if best_c != c:
                colors[v] = best_c if best_c_conflicts < best_conflicts else colors[v]

        # 更新禁忌表
        tabu[v] = (colors[v], )
        if len(tabu) > C // 2:
            tabu.pop(next(iter(tabu)))

        # 更新最佳解
        current_conflicts = _count_conflicts(colors, adj_matrix)
        if current_conflicts < best_conflicts:
            best_conflicts = current_conflicts
            best_colors = colors.copy()

    return best_colors, best_conflicts


def head_coloring_fast(adj_matrix, n_colors, s_assign, Ls, max_iter=3000, seed=42):
    """
    更高效的图着色实现，使用增量冲突计算
    """
    rng = np.random.RandomState(seed)
    C = adj_matrix.shape[0]

    # DSATUR初始化
    colors = _dsatur_init(adj_matrix, n_colors, rng)

    # 预计算每个顶点的邻接列表
    neighbors = [np.where(adj_matrix[i] > 0)[0] for i in range(C)]

    # 为每个顶点维护每种颜色的冲突数
    color_conflicts = np.zeros((C, n_colors), dtype=int)
    for v in range(C):
        for nb in neighbors[v]:
            color_conflicts[v, colors[nb]] += 1

    # 计算总冲突边数
    total_conflicts = sum(color_conflicts[v, colors[v]] for v in range(C)) // 2

    best_colors = colors.copy()
    best_conflicts = total_conflicts

    tabu_recolor = {}  # (vertex, old_color) -> expiry_iteration
    tabu_len = min(C // 4, 50)

    for iteration in range(max_iter):
        if best_conflicts == 0:
            break

        # 找冲突顶点
        conflict_verts = [v for v in range(C) if color_conflicts[v, colors[v]] > 0]
        if not conflict_verts:
            break

        # 随机选冲突顶点
        v = rng.choice(conflict_verts)
        old_c = colors[v]

        # 找最佳颜色（冲突最少且满足C2）
        best_new_c = old_c
        best_new_conf = color_conflicts[v, old_c]

        candidates = list(range(n_colors))
        rng.shuffle(candidates)
        for c in candidates:
            if c == old_c:
                continue
            if (v, c) in tabu_recolor and tabu_recolor[(v, c)] > iteration:
                continue
            if color_conflicts[v, c] < best_new_conf:
                # 检查C2约束
                colors[v] = c
                if _check_c2_fast(colors, s_assign, Ls, n_colors, v, c):
                    best_new_c = c
                    best_new_conf = color_conflicts[v, c]
                colors[v] = old_c

        if best_new_c != old_c:
            # 更新颜色
            new_c = best_new_c
            colors[v] = new_c

            # 增量更新邻接顶点的color_conflicts
            for nb in neighbors[v]:
                color_conflicts[nb, old_c] -= 1
                color_conflicts[nb, new_c] += 1

            # 更新总冲突数
            total_conflicts = sum(color_conflicts[vv, colors[vv]] for vv in range(C)) // 2

            # 更新禁忌
            tabu_recolor[(v, old_c)] = iteration + tabu_len
            if len(tabu_recolor) > C * 2:
                # 清理过期条目
                expired = [k for k, exp in tabu_recolor.items() if exp <= iteration]
                for k in expired:
                    del tabu_recolor[k]

            if total_conflicts < best_conflicts:
                best_conflicts = total_conflicts
                best_colors = colors.copy()

    return best_colors, best_conflicts


def _dsatur_init(adj_matrix, n_colors, rng):
    """DSATUR贪心初始化"""
    C = adj_matrix.shape[0]
    colors = np.full(C, -1, dtype=int)
    saturation = np.zeros(C, dtype=int)  # 每个顶点的饱和度
    colored_neighbors = [set() for _ in range(C)]

    for step in range(C):
        # 选择饱和度最高的未着色顶点
        uncolored = np.where(colors == -1)[0]
        if len(uncolored) == 0:
            break

        # 优先选饱和度高的，同饱和度选度数高的
        max_sat = max(saturation[v] for v in uncolored)
        candidates = [v for v in uncolored if saturation[v] == max_sat]
        if len(candidates) > 1:
            degrees = [np.sum(adj_matrix[v]) for v in candidates]
            max_deg = max(degrees)
            candidates = [c for c, d in zip(candidates, degrees) if d == max_deg]

        v = candidates[rng.randint(len(candidates))]

        # 选择不冲突的颜色
        used = colored_neighbors[v]
        available = [c for c in range(n_colors) if c not in used]
        if available:
            colors[v] = available[rng.randint(len(available))]
        else:
            colors[v] = rng.randint(n_colors)

        # 更新邻接顶点的饱和度
        neighbors = np.where(adj_matrix[v] > 0)[0]
        for nb in neighbors:
            if colors[v] not in colored_neighbors[nb]:
                colored_neighbors[nb].add(colors[v])
                saturation[nb] += 1

    return colors


def _count_conflicts(colors, adj_matrix):
    """计算冲突边数"""
    C = len(colors)
    conflicts = 0
    for i in range(C):
        for j in range(i + 1, C):
            if adj_matrix[i, j] and colors[i] == colors[j]:
                conflicts += 1
    return conflicts


def _get_conflict_vertices(colors, adj_matrix):
    """获取所有冲突顶点"""
    C = len(colors)
    conflict = set()
    for i in range(C):
        for j in range(i + 1, C):
            if adj_matrix[i, j] and colors[i] == colors[j]:
                conflict.add(i)
                conflict.add(j)
    return list(conflict)


def _check_c2(colors, s_assign, Ls, T):
    """检查C2约束：每个时隙每颗卫星最多Ls个波束"""
    S = len(np.unique(s_assign))
    for t in range(T):
        cells_in_slot = np.where(colors == t)[0]
        for s in range(S):
            count = np.sum(s_assign[cells_in_slot] == s)
            if count > Ls:
                return False
    return True


def _check_c2_fast(colors, s_assign, Ls, T, changed_v, new_color):
    """快速检查C2约束（只检查变更涉及的卫星和时隙）"""
    s_v = s_assign[changed_v]
    # 检查新时隙
    cells_in_new_slot = np.where(colors == new_color)[0]
    count = np.sum(s_assign[cells_in_new_slot] == s_v)
    return count <= Ls
