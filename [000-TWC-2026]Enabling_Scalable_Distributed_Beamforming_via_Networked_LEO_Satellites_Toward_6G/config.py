"""
仿真参数配置
论文: Enabling Scalable Distributed Beamforming via Networked LEO Satellites Toward 6G
期刊: IEEE TWC, 2026
"""

import numpy as np

# ===== 物理常数 =====
R_EARTH = 6400e3          # 地球半径 (m)
C_LIGHT = 3e8             # 光速 (m/s)

# ===== 轨道参数 =====
H_ORBIT = 500e3           # 轨道高度 (m)
R_SERVICE = 200e3         # 服务区域半径 (m)

# ===== 系统参数 (Table III) =====
FC = 12.7e9               # 载波频率 (Hz), Ku波段
DELTA_F = 120e3           # 子载波间隔 (Hz)
K_SUB = 1024              # 子载波数
BANDWIDTH = DELTA_F * K_SUB  # 总带宽

# ===== 天线参数 (Table III) =====
N_H_DEFAULT = 16          # 水平方向天线数
N_V_DEFAULT = 16          # 垂直方向天线数
N_RF_DEFAULT = 8          # 射频链数

# ===== 功率与噪声 (Table III) =====
PS_DBM = 50               # 功率预算 (dBm)
PS_DEFAULT = 10 ** ((PS_DBM - 30) / 10)  # W
N0_DBM_HZ = -173.855      # 噪声PSD (dBm/Hz)
NF_DB = 10                # 噪声系数 (dB)

# ===== 默认网络配置 (Table III) =====
S_DEFAULT = 4             # 卫星数
U_DEFAULT = 16            # UT数

# ===== Rician参数 =====
KAPPA_MIN_DB = 15
KAPPA_MAX_DB = 20

# ===== WMMSE优化 =====
MAX_ITER = 50
TOL = 1e-4


def noise_variance():
    """每子载波噪声方差 σ² = N0·Δf (论文公式8下方定义)"""
    N0 = 10 ** ((N0_DBM_HZ - 30) / 10)  # W/Hz
    return N0 * DELTA_F
