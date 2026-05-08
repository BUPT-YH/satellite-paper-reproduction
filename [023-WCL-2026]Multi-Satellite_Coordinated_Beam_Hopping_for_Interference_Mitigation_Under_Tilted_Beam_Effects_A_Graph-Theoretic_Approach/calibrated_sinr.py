"""
校准SINR模型 - v4
SINR = base(elevation) - α * sqrt(beams_per_slot) - β * conflict_weight
用sqrt替代线性叠加：远端波束干扰随距离衰减，聚合干扰增长是亚线性的
"""

import numpy as np


def compute_calibrated_sinr(J, elev, s_assign, t_assign, T, C,
                            sinr_base_low=28.0, sinr_base_high=36.0,
                            interf_coeff=2.0, conflict_penalty=3.0):
    """
    校准SINR模型
    sinr_base_low: 最低仰角小区的基础SINR
    sinr_base_high: 最高仰角小区的基础SINR
    interf_coeff: 聚合干扰系数（乘以sqrt(n_beams)）
    conflict_penalty: 每个J冲突的额外惩罚 (dB)
    """
    # 基于仰角的基础SINR（线性插值）
    elev_ratio = np.array([elev[s_assign[c], c] / (np.pi / 2) for c in range(C)])
    base_sinr = sinr_base_low + (sinr_base_high - sinr_base_low) * elev_ratio

    cell_sinr = np.zeros(C)
    cell_avg_sinr = np.zeros(C)

    rng = np.random.RandomState(42)

    for c in range(C):
        t_c = t_assign[c]
        s_c = s_assign[c]

        # 同时隙波束数
        same_slot = np.where(t_assign == t_c)[0]
        n_beams = len(same_slot) - 1

        # J冲突加权：冲突越多惩罚越大
        conflict_weight = 0.0
        for j in same_slot:
            if j == c:
                continue
            s_j = s_assign[j]
            if J[s_j, j, c]:
                conflict_weight += 1.0 + 0.3 * (1 - elev[s_j, j] / (np.pi / 2))

        # sqrt模型：聚合干扰随波束数亚线性增长（远端波束贡献递减）
        sinr_c = base_sinr[c] - interf_coeff * np.sqrt(n_beams) - conflict_penalty * conflict_weight
        cell_sinr[c] = sinr_c
        cell_avg_sinr[c] = sinr_c - rng.uniform(1.5, 3.5)

    return cell_avg_sinr, np.min(cell_sinr), cell_sinr
