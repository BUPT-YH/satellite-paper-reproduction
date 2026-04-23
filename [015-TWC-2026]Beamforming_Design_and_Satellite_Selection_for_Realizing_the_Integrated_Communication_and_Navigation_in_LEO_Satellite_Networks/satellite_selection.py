"""
卫星选择算法实现
包含 OCF 博弈卫星选择和基准方案（通导导向、导航导向、启发式ICAN、联盟式ICAN）
"""
import numpy as np
import config as cfg
from channel_model import (
    compute_gdop, compute_topology_contribution, compute_distance
)
from beamforming import compute_sum_rate_for_scheme


def random_satellite_selection(S, C, I, K, seed=None):
    """
    随机初始化卫星选择方案
    每个 UE 随机选择 I 颗卫星，每颗卫星最多服务 K 个 UE
    """
    if seed is not None:
        np.random.seed(seed)

    alpha = {}
    for c in range(C):
        candidates = list(range(S))
        np.random.shuffle(candidates)
        selected = candidates[:min(I, S)]
        for s in selected:
            # 检查卫星负载
            current_load = sum(1 for cc in range(C) if alpha.get((s, cc), 0) == 1)
            if current_load < K:
                alpha[(s, c)] = 1

    return alpha


def communication_oriented_selection(channels, sat_pos, ue_pos, S, C, I, K, P_max, noise_power):
    """
    通信导向卫星选择：最小化信道相似度以减少干扰
    """
    alpha = {}

    for c in range(C):
        selected = []
        # 计算所有卫星对 UE c 的信道增益
        gains = []
        for s in range(S):
            h = channels.get((s, c))
            if h is not None:
                gains.append((s, np.linalg.norm(h) ** 2))
            else:
                gains.append((s, 0))

        # 按信道增益排序
        gains.sort(key=lambda x: -x[1])

        for s, _ in gains:
            if len(selected) >= I:
                break
            # 检查信道相似度（与已选卫星的信道相关性）
            h_c = channels.get((s, c))
            if h_c is None:
                continue

            min_similarity = float('inf')
            is_diverse = True
            for s2 in selected:
                h2 = channels.get((s2, c))
                if h2 is not None:
                    sim = np.abs(np.conj(h_c).T @ h2) ** 2 / (np.linalg.norm(h_c)**2 * np.linalg.norm(h2)**2)
                    if sim > 0.8:  # 高相关性
                        is_diverse = False
                        break

            if is_diverse:
                current_load = sum(1 for cc in range(C) if alpha.get((s, cc), 0) == 1)
                if current_load < K:
                    selected.append(s)
                    alpha[(s, c)] = 1

        # 如果多样性选择不足，补充增益最大的
        if len(selected) < I:
            for s, _ in gains:
                if len(selected) >= I:
                    break
                if s not in selected:
                    current_load = sum(1 for cc in range(C) if alpha.get((s, cc), 0) == 1)
                    if current_load < K:
                        selected.append(s)
                        alpha[(s, c)] = 1

    return alpha


def navigation_oriented_selection(sat_pos, ue_pos, S, C, I, K):
    """
    导航导向卫星选择：最小化 GDOP
    每个 UE 贪心选择使 GDOP 最小的卫星组合
    """
    alpha = {}

    for c in range(C):
        # 贪心选择: 每次添加使 GDOP 下降最多的卫星
        selected = []

        # 先选距离最近的卫星
        distances = [(s, compute_distance(sat_pos[s], ue_pos[c])) for s in range(S)]
        distances.sort(key=lambda x: x[1])

        # 第一颗选最近的
        for s, _ in distances:
            selected.append(s)
            break

        # 贪心添加后续卫星
        while len(selected) < I:
            best_sat = -1
            best_gdop = float('inf')

            for s in range(S):
                if s in selected:
                    continue
                current_load = sum(1 for cc in range(C) if alpha.get((s, cc), 0) == 1)
                if current_load >= K:
                    continue

                test_selected = selected + [s]
                gdop = compute_gdop(sat_pos, ue_pos[c], test_selected)
                if gdop < best_gdop:
                    best_gdop = gdop
                    best_sat = s

            if best_sat >= 0:
                selected.append(best_sat)
            else:
                break

        for s in selected:
            alpha[(s, c)] = 1

    return alpha


def heuristic_ican_selection(channels, sat_pos, ue_pos, S, C, I, K, P_max, noise_power):
    """
    启发式 ICAN 卫星选择：同时考虑通信和导航
    先基于信道相似度筛选候选集，再从中选最优拓扑
    """
    alpha = {}

    for c in range(C):
        # 候选集：信道增益最高的前 2I 颗卫星
        gains = [(s, np.linalg.norm(channels.get((s, c), np.zeros(cfg.N)))**2) for s in range(S)]
        gains.sort(key=lambda x: -x[1])
        candidates = [s for s, _ in gains[:min(2*I, S)]]

        # 从候选集中贪心选择: 每次添加使综合指标最优的卫星
        selected = []

        while len(selected) < I and candidates:
            best_sat = -1
            best_score = -float('inf')

            for s in candidates:
                if s in selected:
                    continue
                current_load = sum(1 for cc in range(C) if alpha.get((s, cc), 0) == 1)
                if current_load >= K:
                    continue

                test_selected = selected + [s]

                # 综合指标: 信道增益 + GDOP 改善
                h_gain = np.linalg.norm(channels.get((s, c), np.zeros(cfg.N)))**2
                gdop = compute_gdop(sat_pos, ue_pos[c], test_selected)
                # 归一化综合评分
                score = h_gain * 1e10 - gdop

                if score > best_score:
                    best_score = score
                    best_sat = s

            if best_sat >= 0:
                selected.append(best_sat)
                candidates.remove(best_sat)
            else:
                break

        for s in selected:
            alpha[(s, c)] = 1

    return alpha


def coalitional_ican_selection(channels, sat_pos, ue_pos, S, C, I, K, P_max, noise_power):
    """
    联盟式 ICAN 卫星选择：枚举 C(S,I) 种组合选最优
    为降低复杂度，使用部分枚举
    """
    from itertools import combinations

    alpha = {}

    for c in range(C):
        # 候选集: 可达卫星（排除已满载的）
        candidates = []
        for s in range(S):
            current_load = sum(1 for cc in range(C) if alpha.get((s, cc), 0) == 1)
            if current_load < K:
                candidates.append(s)

        best_combo = None
        best_rate = -1

        # 限制搜索规模
        max_combos = 500
        all_combos = list(combinations(candidates, min(I, len(candidates))))
        if len(all_combos) > max_combos:
            indices = np.random.choice(len(all_combos), max_combos, replace=False)
            all_combos = [all_combos[i] for i in indices]

        for combo in all_combos:
            gdop = compute_gdop(sat_pos, ue_pos[c], list(combo))
            if gdop > cfg.gamma_nav:
                continue

            # 简化：用信道增益和估计速率
            rate = sum(np.linalg.norm(channels.get((s, c), np.zeros(cfg.N)))**2 for s in combo)
            if rate > best_rate:
                best_rate = rate
                best_combo = combo

        if best_combo is None:
            # 回退到导航导向选择
            distances = [(s, compute_distance(sat_pos[s], ue_pos[c])) for s in candidates]
            distances.sort(key=lambda x: x[1])
            best_combo = [s for s, _ in distances[:min(I, len(distances))]]

        for s in best_combo:
            alpha[(s, c)] = 1

    return alpha


def ocf_satellite_selection(channels, sat_pos, ue_pos, S, C, I, K, P_max, noise_power,
                             gamma_com, gamma_nav, rho, max_iter=20):
    """
    OCF 博弈卫星选择（Algorithm 2）
    基于 Overlapping Coalition Formation 的迭代卫星选择
    rho: 权重因子，rho=1 纯通信优化，rho=0 纯导航优化
    """
    # 阶段 1: 初始化联盟结构（均衡分配）
    np.random.seed(42)
    alpha = {}
    sat_load = np.zeros(S, dtype=int)
    for c in range(C):
        loads = [(sat_load[s] + np.random.uniform(0, 0.1), s) for s in range(S)]
        loads.sort()
        count = 0
        for _, s in loads:
            if count >= I:
                break
            if sat_load[s] < K:
                alpha[(s, c)] = 1
                sat_load[s] += 1
                count += 1

    # 迭代优化
    for iteration in range(max_iter):
        changed = False

        for c in range(C):
            current_sats = [s for s in range(S) if alpha.get((s, c), 0) == 1]
            Ic = len(current_sats)

            # 步骤 1: 退出操作 (Exiting) — 检查导航可行性
            if Ic > 0:
                gdop = compute_gdop(sat_pos, ue_pos[c], current_sats)
                if gdop > gamma_nav and Ic > 1:
                    min_mu = float('inf')
                    exit_sat = current_sats[0]
                    for s in current_sats:
                        mu = compute_topology_contribution(sat_pos, ue_pos[c], current_sats, s)
                        if mu < min_mu:
                            min_mu = mu
                            exit_sat = s
                    alpha[(exit_sat, c)] = 0
                    current_sats.remove(exit_sat)
                    sat_load[exit_sat] -= 1
                    Ic -= 1
                    changed = True

            # 步骤 2: 加入操作 (Joining) — 不足 I 颗卫星时补充
            if Ic < I:
                candidates = [s for s in range(S) if s not in current_sats and sat_load[s] < K]
                scored = []
                for s in candidates:
                    mu = compute_topology_contribution(sat_pos, ue_pos[c], current_sats, s)
                    h_gain = np.linalg.norm(channels.get((s, c), np.zeros(cfg.N))) ** 2
                    score = rho * h_gain + (1 - rho) * mu
                    scored.append((s, score))
                scored.sort(key=lambda x: -x[1])
                for s, _ in scored:
                    if Ic >= I:
                        break
                    alpha[(s, c)] = 1
                    current_sats.append(s)
                    sat_load[s] += 1
                    Ic += 1
                    changed = True

            # 步骤 3: 切换操作 (Switching) — 寻找更优卫星
            for s in list(current_sats):
                best_switch = None
                best_utility = -float('inf')

                h_old = channels.get((s, c), np.zeros(cfg.N))
                gdop_old = compute_gdop(sat_pos, ue_pos[c], current_sats)

                for s_new in range(S):
                    if s_new in current_sats or sat_load[s_new] >= K:
                        continue

                    test_sats = [ss for ss in current_sats if ss != s] + [s_new]
                    gdop_new = compute_gdop(sat_pos, ue_pos[c], test_sats)

                    if gdop_new > gamma_nav:
                        continue

                    h_new = channels.get((s_new, c), np.zeros(cfg.N))

                    # 归一化通信增益
                    rate_gain = np.linalg.norm(h_new) ** 2 - np.linalg.norm(h_old) ** 2
                    max_h = np.linalg.norm(h_new) ** 2 + np.linalg.norm(h_old) ** 2
                    norm_rate = rate_gain / max(max_h, 1e-30)

                    # 归一化导航增益
                    gdop_gain = gdop_old - gdop_new
                    norm_gdop = gdop_gain / max(gdop_old, 1e-6)

                    # 加权效用
                    utility = rho * norm_rate + (1 - rho) * norm_gdop
                    if utility > best_utility:
                        best_utility = utility
                        best_switch = s_new

                if best_switch is not None and best_utility > 0:
                    alpha[(s, c)] = 0
                    alpha[(best_switch, c)] = 1
                    sat_load[s] -= 1
                    sat_load[best_switch] += 1
                    current_sats.remove(s)
                    current_sats.append(best_switch)
                    changed = True

        if not changed:
            break

    return alpha


def evaluate_selection(channels, sat_pos, ue_pos, alpha, S, C, I, P_max, noise_power, rho):
    """
    评估卫星选择方案的性能
    返回: (sum_rate, avg_gdop)
    """
    # 使用 WMMSE (等效 DC 规划) 计算和速率
    sum_rate = compute_sum_rate_for_scheme(channels, alpha, 'DC', P_max, noise_power,
                                            S=S, C=C)

    # 计算 GDOP
    gdop_list = []
    for c in range(C):
        serving = [s for s in range(S) if alpha.get((s, c), 0) == 1]
        if len(serving) >= 4:  # GDOP 需要至少 4 颗卫星
            gdop = compute_gdop(sat_pos, ue_pos[c], serving)
            gdop_list.append(gdop)

    avg_gdop = np.mean(gdop_list) if gdop_list else 100.0

    return sum_rate, avg_gdop
