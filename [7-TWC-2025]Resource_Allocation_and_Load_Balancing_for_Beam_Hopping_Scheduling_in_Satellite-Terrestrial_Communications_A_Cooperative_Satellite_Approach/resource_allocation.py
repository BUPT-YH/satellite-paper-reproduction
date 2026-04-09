"""
资源分配模块 — MM算法 (简化高效版)
基于 Majorization-Minimization 算法求解资源分配
"""

import numpy as np
import config as cfg


def compute_channel_gain(distance_m, freq_hz=cfg.freq):
    """计算自由空间路径损耗 + 天线增益"""
    c = 3e8
    wavelength = c / freq_hz
    fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(freq_hz) - 20 * np.log10(c) + 92.45
    gain = 10 ** (-fspl_db / 10)
    antenna_gain = (np.pi * cfg.antenna_radius / wavelength) ** 2
    gain *= antenna_gain
    return gain


def generate_channel_gains(n_sat, n_cells_per_sat, n_beams, cell_distances):
    """生成信道增益矩阵 h[r,j,s,k]"""
    h = np.zeros((n_sat, n_beams, n_sat, n_beams))
    for r in range(n_sat):
        for s in range(n_sat):
            for j in range(n_beams):
                for k in range(n_beams):
                    d = cell_distances[k % len(cell_distances)]
                    if r == s and j == k:
                        h[r, j, s, k] = compute_channel_gain(d)
                    elif r == s:
                        h[r, j, s, k] = compute_channel_gain(d) * 0.08
                    else:
                        h[r, j, s, k] = compute_channel_gain(d * 1.5) * 0.03
    return h


def mm_resource_allocation(bh_patterns, h, n_sat, n_beams, n_freq, P_sat, P_tot):
    """
    MM算法资源分配 — 简化版 (解析近似)
    返回: f_alloc (离散), Pb
    """
    # 简化的功率和频率分配策略:
    # 1. 均匀功率分配
    # 2. 基于干扰感知的频率分配

    # 初始功率
    Pb = P_sat / (n_beams * n_freq)

    # 频率分配: 基于干扰最小化的贪心策略
    f_alloc = np.zeros((n_sat, n_beams, n_freq), dtype=int)

    for s in range(n_sat):
        for k in range(n_beams):
            # 计算每个频率段的干扰水平
            interferences = np.zeros(n_freq)
            for l in range(n_freq):
                for r in range(n_sat):
                    for j in range(n_beams):
                        if r == s and j == k:
                            continue
                        interferences[l] += f_alloc[r, j, l] * h[r, j, s, k] ** 2

            # 选择干扰最小的频率段
            best_l = np.argmin(interferences)
            # 分配1-2个频率段
            f_alloc[s, k, best_l] = 1
            # 如果功率允许，分配第二个频率段
            sorted_l = np.argsort(interferences)
            for l_idx in sorted_l:
                if f_alloc[s, k, l_idx] == 0:
                    total_active = f_alloc[s].sum()
                    if total_active * Pb < P_sat:
                        f_alloc[s, k, l_idx] = 1
                    break

    # MM迭代优化 (简化: 调整功率)
    for iteration in range(3):  # 仅3轮迭代
        for s in range(n_sat):
            active = f_alloc[s].sum()
            if active > 0:
                Pb_s = P_sat / active
                # 基于SINR反馈微调频率分配
                for k in range(n_beams):
                    sinrs = np.zeros(n_freq)
                    for l in range(n_freq):
                        if f_alloc[s, k, l] > 0:
                            signal = h[s, k, s, k] ** 2 * Pb_s
                            interference = 0
                            for r in range(n_sat):
                                if r != s:
                                    for j in range(n_beams):
                                        interference += f_alloc[r, j, l] * h[r, j, s, k] ** 2 * Pb_s
                                elif j != k:
                                    interference += f_alloc[s, j, l] * h[s, j, s, k] ** 2 * Pb_s
                            noise = cfg.n0 * cfg.B0
                            sinrs[l] = signal / (interference + noise)

                    # 如果SINR太低，尝试切换频率段
                    if sinrs.max() > 0 and sinrs[sinrs > 0].min() < sinrs.max() * 0.1:
                        worst_l = np.argmin(sinrs + (f_alloc[s, k] == 0) * 1e10)
                        if f_alloc[s, k, worst_l] > 0:
                            f_alloc[s, k, worst_l] = 0
                            best_l = np.argmax(sinrs == 0)
                            if sinrs[best_l] == 0:
                                f_alloc[s, k, best_l] = 1

    # 最终功率调整
    max_active = max(f_alloc[s].sum() for s in range(n_sat))
    Pb = P_sat / max(max_active, 1)

    return f_alloc, Pb


def compute_throughput_per_beam(f_alloc, Pb, h, n_sat, n_beams, n_freq):
    """计算每个波束的吞吐量 (Mbps)"""
    throughputs = np.zeros((n_sat, n_beams))
    for s in range(n_sat):
        for k in range(n_beams):
            rate = 0
            for l in range(n_freq):
                if f_alloc[s, k, l] > 0.5:
                    signal = h[s, k, s, k] ** 2 * Pb
                    interference = 0
                    for r in range(n_sat):
                        for j in range(n_beams):
                            if r == s and j == k:
                                continue
                            interference += f_alloc[r, j, l] * h[r, j, s, k] ** 2 * Pb
                    noise = cfg.n0 * cfg.B0
                    sinr = signal / (interference + noise)
                    rate += cfg.B0 * np.log2(1 + max(sinr, 0))
            throughputs[s, k] = rate / 1e6  # 转换为 Mbps
    return throughputs
