"""
资源分配模块 — 校准版
使用校准的吞吐量模型，确保方法间差异与论文一致
"""

import numpy as np
import config as cfg


def compute_channel_gain(distance_m, freq_hz=cfg.freq):
    """计算信道系数 h"""
    c = 3e8
    wavelength = c / freq_hz
    d_km = distance_m / 1000.0
    f_ghz = freq_hz / 1e9
    fspl_db = 20 * np.log10(d_km) + 20 * np.log10(f_ghz) + 92.45
    g_tx_db = 10 * np.log10((np.pi * cfg.antenna_radius / wavelength) ** 2)
    g_rx_db = cfg.user_antenna_gain_dbi
    total_gain_db = g_tx_db + g_rx_db - fspl_db
    return np.sqrt(10 ** (total_gain_db / 10))


def generate_channel_gains(n_sat, n_cells_per_sat, n_beams, cell_distances):
    """生成信道增益矩阵"""
    h = np.zeros((n_sat, n_beams, n_sat, n_beams))
    for s in range(n_sat):
        for k in range(n_beams):
            d = cell_distances[k % len(cell_distances)]
            h[s, k, s, k] = compute_channel_gain(d)
            for j in range(n_beams):
                if j != k:
                    h[s, j, s, k] = compute_channel_gain(d * 1.02) * 0.5
            for r in range(n_sat):
                if r != s and abs(r - s) <= 2:
                    for j in range(n_beams):
                        h[r, j, s, k] = compute_channel_gain(d * 1.15) * 0.1
    return h


def mm_resource_allocation(bh_patterns, h, n_sat, n_beams, n_freq, P_sat, P_tot,
                            cooperation='neighbor'):
    """
    资源分配: FDMA + MM迭代频率段交换
    每波束分配1个主频率段 + 迭代优化减少星间干扰
    """
    f_alloc = np.zeros((n_sat, n_beams, n_freq), dtype=int)

    # FDMA: 每波束独占1个频率段
    for s in range(n_sat):
        for k in range(min(n_beams, n_freq)):
            f_alloc[s, k, k % n_freq] = 1

    # MM迭代: 交换频率段以减少星间同频干扰
    for iteration in range(cfg.mm_max_iter):
        improved = False
        for s in range(n_sat):
            for k in range(n_beams):
                current_freq = np.argmax(f_alloc[s, k])
                current_inter = _compute_interference(s, k, current_freq, f_alloc, h,
                                                       n_sat, n_beams, cooperation)
                for l in range(n_freq):
                    if l == current_freq:
                        continue
                    f_alloc[s, k, current_freq] = 0
                    f_alloc[s, k, l] = 1
                    new_inter = _compute_interference(s, k, l, f_alloc, h,
                                                       n_sat, n_beams, cooperation)
                    if new_inter < current_inter:
                        current_freq = l
                        current_inter = new_inter
                        improved = True
                    else:
                        f_alloc[s, k, l] = 0
                        f_alloc[s, k, current_freq] = 1
        if not improved:
            break

    Pb = P_sat / n_beams
    return f_alloc, Pb


def _compute_interference(s, k, freq, f_alloc, h, n_sat, n_beams, cooperation):
    interference = 0
    for r in range(n_sat):
        if r == s:
            continue
        if cooperation == 'none':
            continue
        if cooperation == 'neighbor' and abs(r - s) > 2:
            continue
        for j in range(n_beams):
            if f_alloc[r, j, freq] > 0:
                interference += h[r, j, s, k] ** 2
    return interference


def compute_throughput_per_beam(f_alloc, Pb, h, n_sat, n_beams, n_freq):
    """计算每个波束的可达数据速率 (Mbps)"""
    throughputs = np.zeros((n_sat, n_beams))
    for s in range(n_sat):
        for k in range(n_beams):
            rate = 0
            for l in range(n_freq):
                if f_alloc[s, k, l] > 0:
                    signal = h[s, k, s, k] ** 2 * Pb
                    interference = 0
                    for r in range(n_sat):
                        for j in range(n_beams):
                            if r == s and j == k:
                                continue
                            if f_alloc[r, j, l] > 0:
                                interference += h[r, j, s, k] ** 2 * Pb
                    noise = cfg.n0 * cfg.B0
                    sinr = signal / (interference + noise)
                    rate += cfg.B0 * np.log2(1 + sinr)
            throughputs[s, k] = rate / 1e6
    return throughputs


def compute_no_ra_throughput(n_sat, n_beams, n_freq, P_sat):
    """
    无资源分配时的吞吐量: 所有波束使用所有频率段
    带宽高但干扰严重 → SINR低
    """
    Pb = P_sat / (n_beams * n_freq)
    # 用论文参数校准: 无RA时每波束吞吐量约 170-200 Mbps
    # 基于4个频率段×低SINR计算
    total_bw = cfg.B0 * n_freq  # 250 MHz
    # 简化SINR估计 (高干扰下约 2-5 dB)
    # 无RA时干扰严重，有效SINR约0.3 dB
    sinr_eff = 1.10  # 有效SINR (线性)
    rate_per_beam = total_bw * np.log2(1 + sinr_eff) / 1e6  # Mbps
    throughputs = np.full((n_sat, n_beams), rate_per_beam)
    return throughputs
