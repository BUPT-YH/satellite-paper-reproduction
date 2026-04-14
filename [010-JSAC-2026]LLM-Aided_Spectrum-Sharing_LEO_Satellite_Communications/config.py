"""
仿真参数配置 — 论文: LLM-Aided Spectrum-Sharing LEO Satellite Communications
期刊: IEEE JSAC, Vol. 44, 2026
"""

import numpy as np
from math import factorial

# ==================== 基本物理常量 ====================
RE = 6371e3            # 地球半径 (m)

# ==================== 轨道参数 ====================
H_L = 600e3            # 最低轨道高度 (m)
H_U = 1000e3           # 最高轨道高度 (m)
RL = RE + H_L          # 最低轨道半径 (m) ≈ 6971 km
RH = RE + H_U          # 最高轨道半径 (m) ≈ 7371 km
THETA0 = np.deg2rad(25)  # 天线波束半角 (rad), 论文未明确给出，假设25°

# ==================== 路径损耗参数 ====================
ALPHA = 2.1            # 路径衰减因子

# ==================== Shadowed Rician 信道参数 ====================
# 对应 "Average Shadowing" 场景 (Abdi et al., 2003)
MN = 5                  # LoS 衰落严重程度参数 (整数)
OMEGA = 0.835           # LoS 分量平均功率
B_PARAM = 0.126         # 多径分量参数 (2b 为多径平均功率)
# 由 b 和 Ω 计算得到的参数
BETA = 1.0 / (2 * B_PARAM)                # β = 1/(2b)
DELTA = OMEGA / (2 * B_PARAM * (2 * B_PARAM * MN + OMEGA))  # δ

# αh 系数
ALPHA_H = (2 * B_PARAM * MN / (2 * B_PARAM * MN + OMEGA))**MN / (2 * B_PARAM)

# ζ(k) 系数预计算
def compute_zeta(mn, alpha_h, delta_val):
    """计算 ζ(k) 系数, k = 0, 1, ..., mn-1"""
    zeta = np.zeros(mn)
    for k in range(mn):
        # Pochhammer 符号 (1-mn)_k = (1-mn)(2-mn)...(k-mn)
        pochhammer = 1.0
        for j in range(k):
            pochhammer *= (1 - mn + j)
        zeta[k] = alpha_h * ((-1)**k) * pochhammer * (delta_val**k) / (factorial(k)**2)
    return zeta

ZETA = compute_zeta(MN, ALPHA_H, DELTA)

# ==================== 噪声参数 ====================
SIGMA_N2 = 1e-13       # 接收机噪声平均功率 (W)

# ==================== 仿真参数 ====================
MF = 80                 # Gaussian-Chebyshev 积分项数
LAMBDA_E_FS = 5.0       # λe·Δfs: 干扰信号在传输带宽内的平均数量
PE_OVER_PS = 0.1        # 干扰功率与信号功率比 (Pe·Δfe)/(Ps·Δfs) 的等效比
                         # 论文未明确给出, 假设干扰信号功率约为信号的 1/10

# ==================== 默认卫星-地面距离 ====================
DSD_DEFAULT = 800e3     # 默认 S-D 距离 (m) = 800 km

# ==================== 资源分配参数 (Fig.12, 13) ====================
TOTAL_BW = 2.16e9       # 总可用带宽 Δfs = 2.16 GHz (Hz)
DATA_VOLUMES = {         # 各业务数据量 (GB, 1 GB = 10^9 bytes = 8×10^9 bits)
    'A': 0.02,           # 指令控制数据
    'B': 12.0,           # 音视频流
    'C': 2.0,            # 实时通话
    'D': 40.0,           # 文件下载
}

# 调制方式: 每符号比特数
MODULATION_BITS = {
    'BPSK': 1,
    '4QAM': 2,
    '16QAM': 4,
    '64QAM': 6,
}

# LLM 决策方案 (DeepSeek r1 输出)
LLM_SCHEME = {
    'A': {'mod': 'BPSK',  'n_sub': 1},
    'B': {'mod': '16QAM', 'n_sub': 30},
    'C': {'mod': '4QAM',  'n_sub': 10},
    'D': {'mod': '64QAM', 'n_sub': 67},
}
LLM_TOTAL_SUB = sum(v['n_sub'] for v in LLM_SCHEME.values())  # 108

# 子载波间距
SUBCARRIER_SPACING = TOTAL_BW / LLM_TOTAL_SUB  # ≈ 20 MHz

# ==================== 绘图参数 ====================
FIG_SINGLE = (3.5, 2.8)
FIG_DOUBLE = (7.16, 3.5)
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
