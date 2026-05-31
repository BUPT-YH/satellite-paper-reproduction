"""
config.py - 仿真参数配置
论文: Co-Existence Analysis of Terrestrial and Non-Terrestrial Networks
      in S-Band Using Stochastic Geometry (IEEE TCOM 2026)
"""

import numpy as np

# ============================================================
# 物理常数
# ============================================================
K_BOLTZMANN = 1.38e-23      # 玻尔兹曼常数 (J/K)
T_NOISE = 290               # 噪声温度 (K)
BANDWIDTH = 20e6            # 带宽 20 MHz
NF_DB = 7                   # 噪声系数 (dB)
R_EARTH = 6371              # 地球半径 (km)

# 噪声功率: sigma^2 = k * T * B * NF
NF_LINEAR = 10 ** (NF_DB / 10)
SIGMA2 = K_BOLTZMANN * T_NOISE * BANDWIDTH * NF_LINEAR  # 瓦特
SIGMA2_DBM = 10 * np.log10(SIGMA2 * 1000)  # dBm ≈ -94 dBm

# ============================================================
# 频段参数
# ============================================================
FREQUENCY = 2e9             # S-band 2 GHz
C = 3e8                     # 光速

# ============================================================
# TN (地面网络) 参数
# ============================================================
P_TN_DBM = 46               # TN BS发射功率 (dBm)
P_TN = 10 ** ((P_TN_DBM - 30) / 10)  # 瓦特

G_BS_DBI = 17               # TN BS天线最大增益 (dBi)
G_BS = 10 ** (G_BS_DBI / 10)

G_U_DBI = 0                 # 用户天线增益 (dBi)
G_U = 10 ** (G_U_DBI / 10)

# TN发射端等效增益 = BS增益 * 用户增益
G_TN = G_BS * G_U

ALPHA_TN = 3                # TN路径损耗指数

# Nakagami-m 参数
# K=0 → m=1 (Rayleigh衰落)
M_TN = 1

# TN簇参数
N_C = 19                    # 每簇TN BS数量 (3层六边形)

# 城市和农村的ISD
DISD_URBAN = 0.75           # km
DISD_RURAL = 7.5            # km

# TN簇半径
# 从论文图3推断: dISD=0.75km时r_TN≈1.6km, r_TN/dISD≈2.13
# 严格推导: 对于19个BS的簇(3层六边形), r_TN = dISD * sqrt(N_c/(2*pi))
# 但论文图3显示比例约为2.13, 我们使用论文的实际值
R_TN_URBAN = 1.6            # km (城市)
R_TN_RURAL = 16.0           # km (农村)

# TN UE位置偏移 x_0 (相对于簇中心)
# 图5默认 x_0=0 (UE在簇中心)
X_0_CENTER = 0.0
X_0_HALF = 0.5              # 0.5 * r_TN
X_0_EDGE = 1.0              # 1.0 * r_TN (边缘)

# ============================================================
# NTN (非地面网络) 参数 — Case I: 卫星下行干扰TN下行
# ============================================================
ALTITUDES = [200, 600, 1200]    # 卫星高度 (km)

P_NTN_DL_DBM = 46           # 卫星发射功率 (dBm) — 下行
P_NTN_DL = 10 ** ((P_NTN_DL_DBM - 30) / 10)  # 瓦特

G_SAT_DBI = 30              # 卫星天线最大增益 (dBi)
G_SAT = 10 ** (G_SAT_DBI / 10)

# NTN下行等效增益 = 卫星增益 * 用户增益
G_NTN_DL = G_SAT * G_U

ALPHA_NTN = 2               # NTN路径损耗指数 (自由空间)

# NTN信道 Nakagami-m: K=200 → m ≈ (K+1)^2/(2K+1)
K_NTN = 200
M_NTN = (K_NTN + 1) ** 2 / (2 * K_NTN + 1)  # ≈ 100.5

# ============================================================
# NTN (非地面网络) 参数 — Case II: NTN UE上行干扰TN下行
# ============================================================
P_NTN_UL_DBM = 23           # NTN UE发射功率 (dBm) — 上行
P_NTN_UL = 10 ** ((P_NTN_UL_DBM - 30) / 10)  # 瓦特

# NTN UE上行等效增益 = G_SAT * G_U
# 论文全局定义G^NTN = G_SAT * G_U，Case I和Case II使用相同的G^NTN
# Case II虽然干扰链路是NTN UE(地面)->TN UE(地面)，但论文使用统一的NTN系统增益参数
# (由T=0dB处Case II覆盖概率与论文图表对比验证确认)
G_NTN_UL = G_SAT * G_U

# 卫星波束覆盖半径
R_NTN = 25.0  # km (卫星波束覆盖半径)

# NTN UE数量 (Case II)
N_U_VALUES = [100, 1000, 2000]

# 隔离距离 (Case II)
# r_iso = 0, dISD, 2*dISD

# ============================================================
# 仿真控制参数
# ============================================================
# SINR阈值范围 (dB)
T_DB_MIN = -50
T_DB_MAX = 30
T_DB_POINTS = 161           # 0.5 dB 步长

# 数值积分参数
QUAD_EPSABS = 1e-10
QUAD_EPSREL = 1e-10
QUAD_LIMIT = 200

# 负载因子
LOAD_FULL = 1.0             # 100%
LOAD_PARTIAL = 0.25         # 25%


def get_r_tn(scenario):
    """获取TN簇半径"""
    if scenario == 'urban':
        return R_TN_URBAN
    elif scenario == 'rural':
        return R_TN_RURAL
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def get_disd(scenario):
    """获取ISD"""
    if scenario == 'urban':
        return DISD_URBAN
    elif scenario == 'rural':
        return DISD_RURAL
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def get_n_active_bs(load_factor):
    """根据负载因子获取活跃BS数量"""
    return max(1, int(round(N_C * load_factor)))


def get_max_ntn_distance(altitude_km):
    """
    计算卫星到地面UE的最大距离
    r_max = sqrt(2 * R_earth * a + a^2)
    """
    return np.sqrt(2 * R_EARTH * altitude_km + altitude_km ** 2)
