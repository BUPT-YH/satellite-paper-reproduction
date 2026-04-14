"""
仿真参数配置 - 从论文 Table I, II, III 提取
Feasibility Analysis of In-Band Coexistence in Dense LEO Satellite Communication Systems
"""

import numpy as np

# ============ 物理常数 ============
R_EARTH = 6371.0        # 地球半径 (km)
MU_EARTH = 398600.4418  # 地球引力常数 (km³/s²)
C_LIGHT = 3e8           # 光速 (m/s)
OMEGA_EARTH = 7.2921e-5 # 地球自转角速度 (rad/s)

# ============ Starlink 星座参数 (Table I) ============
# Walker-Delta 星座: [n_planes, sats_per_plane, altitude_km, inclination_deg, phasing_F]
STARLINK_SHELLS = [
    (72, 22, 550.0, 53.0, 0),   # Shell 1: 1584 sats
    (72, 22, 540.0, 53.0, 0),   # Shell 2: 1584 sats
    (36, 20, 570.0, 70.0, 0),   # Shell 3: 720 sats
    (6,  58, 560.0, 97.6, 0),   # Shell 4: 348 sats
    (4,  43, 560.0, 97.6, 0),   # Shell 5: 172 sats
]
# 总计 4408 颗卫星

# ============ Kuiper 星座参数 (Table II) ============
KUIPER_SHELLS = [
    (34, 34, 630.0, 30.0, 0),   # Shell 1: 1156 sats
    (36, 36, 610.0, 33.0, 0),   # Shell 2: 1296 sats
    (28, 28, 590.0, 51.9, 0),   # Shell 3: 784 sats
]
# 总计 3236 颗卫星

# ============ 频率和带宽 ============
CARRIER_FREQ = 20e9     # 载波频率 20 GHz
BANDWIDTH = 400e6       # 带宽 400 MHz
WAVELENGTH = C_LIGHT / CARRIER_FREQ  # 波长

# ============ 天线参数 ============
# 星载天线: 64×64 UPA
SAT_ANTENNA_NX = 64
SAT_ANTENNA_NY = 64

# 地面用户天线: 多种规格
USER_ANTENNA_CONFIGS = {
    '8x8':   (8, 8),
    '16x16': (16, 16),
    '32x32': (32, 32),
}

# 天线阵元间距 (半波长)
ANTENNA_SPACING = 0.5  # 波长归一化

# ============ 发射功率 ============
# EIRP 频谱密度 (dBW/Hz)
PRIMARY_EIRP_DENSITY = -54.3    # Starlink
SECONDARY_EIRP_DENSITY = -53.3  # Kuiper

# 转换为线性 (W/Hz)
PRIMARY_EIRP = 10 ** (PRIMARY_EIRP_DENSITY / 10)
SECONDARY_EIRP = 10 ** (SECONDARY_EIRP_DENSITY / 10)

# ============ 噪声参数 ============
NOISE_PSD_DBM = -174.0  # 噪声功率谱密度 (dBm/Hz)
NOISE_FIGURE = 1.2       # 噪声系数 (dB)
NOISE_PSD = 10 ** ((NOISE_PSD_DBM - 30 + NOISE_FIGURE) / 10)  # W/Hz

# ============ 仿真参数 ============
MIN_ELEVATION = 25.0     # 最小仰角 (度)
SIM_DURATION = 24 * 3600 # 仿真时长 24 小时 (秒)
TIME_RESOLUTION = 120    # 时间分辨率 (秒) — 使用 2 分钟以加速计算

# INR 保护阈值 (dB)
INR_THRESHOLDS = [-15.0, -12.2, -6.0, 0.0]

# ============ 地面用户位置 (全球城市) ============
# (城市名, 纬度, 经度)
CITIES = [
    ("Austin",       30.27,  -97.74),
    ("Beijing",      39.90,  116.41),
    ("Bogota",        4.71,  -74.07),
    ("Cairo",        30.04,   31.24),
    ("Cape Town",   -33.92,   18.42),
    ("Chicago",      41.88,  -87.63),
    ("London",       51.51,    0.13),
    ("Moscow",       55.76,   37.62),
    ("Mumbai",       19.07,   72.88),
    ("New York",     40.71,  -74.01),
    ("Rio",         -22.91,  -43.17),
    ("Seoul",        37.57,  126.98),
    ("Singapore",     1.35,  103.82),
    ("Sydney",      -33.87,  151.21),
    ("Tokyo",        35.68,  139.69),
]

# ============ 路径损耗参数 ============
# 大气损耗 20 GHz (近似值, dB)
ATMOS_LOSS = 1.0  # 晴天条件
# 闪烁衰落裕量 (dB)
SCINTILLATION_MARGIN = 0.5
