"""
仿真参数配置 - Coverage Diversity in Mega Satellite Constellations
论文: IEEE TWC, Vol. 24, No. 11, November 2025
"""

import numpy as np

# ==================== 基本物理常量 ====================
R_EARTH = 6731  # 地球平均半径 (km), Table II
C_LIGHT = 3e8   # 光速 (m/s)

# ==================== 载波参数 ====================
FC = 2e9         # 载波频率 2 GHz, Table II
WAVELENGTH = C_LIGHT / FC  # 波长 (m)

# ==================== 阴影衰落参数 (参考 [54] Al-Hourani) ====================
BETA = 0.4       # LoS 概率参数, Table II

# 对数正态阴影衰落参数 (dB) — Table II 原始值 (正值)
# 分布: ζ[dB] ~ p_LoS * N(-μ_LoS, σ²_LoS) + p_nLoS * N(-μ_nLoS, σ²_nLoS)
# CDF:  erf((10*log10(x) + μ_LoS) / (√2*σ_LoS))  ← 用正值
# MC:   zeta_db ~ N(-MU_LOS, SIGMA_LOS)            ← 取负号
MU_LOS = 0.4     # LoS 均值 (dB), Table II 正值
MU_NLOS = 0.0    # nLoS 均值 (dB), Table II 正值
SIGMA_LOS = 1.0   # LoS 标准差 (dB)
SIGMA_NLOS = 5.2   # nLoS 标准差 (dB)

# ==================== 系统参数 ====================
RHO_T_DBM = 10    # 发射功率 (dBm), Table II
RHO_T = 10 ** (RHO_T_DBM / 10) * 1e-3  # 转换为瓦特 (W)

GT_DBI = 3       # 发射天线增益 (dBi), Table II
GR_DBI = 2       # 接收天线增益 (dBi), Table II
GT = 10 ** (GT_DBI / 10)  # 线性值
GR = 10 ** (GR_DBI / 10)  # 线性值

WS_DB = -160     # 噪声功率 (dB), Table II
WS = 10 ** (WS_DB / 10) * 1e-3  # 转换为瓦特 (W)

# 波束宽度
PSI_S = np.pi     # 卫星波束宽度 (全向), Table II ψ_o = 2π → 半角 π
PSI_T = np.pi     # 用户波束宽度 (全向)

# ==================== 用户参数 ====================
LAMBDA_O = 1e-10  # 用户密度: 1 user per 100 km² = 1e-10 users/m²
D_O = 0.25        # 占空比 (duty cycle), 论文中 25%
LAMBDA_U = D_O * LAMBDA_O  # 活跃用户密度

# ==================== 星座配置 ====================

# 单壳层配置
SINGLE_SHELL = {
    'Nm': 900,
    'Rm': 600,  # km
}

# 三壳层配置 (论文仿真用)
MULTI_SHELL = {
    'Nm': [900, 400, 100],
    'Rm': [600, 900, 1200],  # km
}

# Starlink Phase 2 配置 (2027年计划)
STARLINK_PHASE2 = {
    'Nm': [2493, 2478, 2547],
    'Rm': [335.9, 340.8, 345.6],  # km
}

# ==================== 仿真控制参数 ====================
N_MC_SAMPLES = 50000  # Monte Carlo 采样次数
GAMMA_RANGE_DB = np.arange(-25, 5, 0.5)  # SINR 阈值范围 (dB)
GAMMA_RANGE = 10 ** (GAMMA_RANGE_DB / 10)  # 线性值

# Fig.6 参数: 用户密度对覆盖概率的影响
GAMMA_FIG6_DB = -20  # dB
GAMMA_FIG6 = 10 ** (GAMMA_FIG6_DB / 10)
LAMBDA_SWEEP = np.logspace(-11, -7, 50)  # 用户密度扫描范围 (users/m²)

# Laplace 反演参数 (Euler's inversion formula)
L_EULER = 30  # Euler 反演的精度参数
