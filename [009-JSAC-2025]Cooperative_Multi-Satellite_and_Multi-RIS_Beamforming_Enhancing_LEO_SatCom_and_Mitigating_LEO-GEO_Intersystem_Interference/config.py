# -*- coding: utf-8 -*-
"""
仿真参数配置
论文: Cooperative Multi-Satellite and Multi-RIS Beamforming
期刊: IEEE JSAC, Vol. 43, No. 1, 2025
"""

import numpy as np

# ==================== 系统参数 ====================
# LEO 卫星参数
J = 2                  # LEO 卫星数量
N_r = 4                # 卫星天线行数 (UPA)
N_c = 4                # 卫星天线列数 (UPA)
N = N_r * N_c          # 每颗卫星天线总数 = 16

# RIS 参数
M_r = 0                # RIS 行数 (由 M 决定)
M_c = 0                # RIS 列数

# 用户参数
U = 4                  # 地面用户 (LU) 数量
K = 4                  # GEO 终端 (GT) 数量

# ==================== 轨道参数 ====================
R_earth = 6371.0       # 地球半径 (km)
h_LEO = 600.0          # LEO 轨道高度 (km)
h_GEO = 35786.0        # GEO 轨道高度 (km)
R_LEO = R_earth + h_LEO
R_GEO = R_earth + h_GEO

# ==================== 频率参数 ====================
c = 3e8                # 光速 (m/s)
f_carrier = 20e9       # 载波频率 20 GHz (Ka band)
wavelength = c / f_carrier  # 波长
d_ant = wavelength / 2  # 卫星天线间距 (半波长)
d_ris = wavelength / 2  # RIS 元素间距 (半波长)

# ==================== 发射参数 ====================
PT_default = 15.0      # 默认最大发射功率 (W)
BW = 250e6             # 带宽 250 MHz
noise_psd = -174       # 噪声功率谱密度 (dBm/Hz)
noise_figure = 7       # 接收机噪声系数 (dB)

# 信道归一化因子 (匹配论文 SINR 范围)
# NF 用于卫星-地面直连链路和卫星-RIS链路
# NF_RIS 用于 RIS-LU 近距离链路, 校准使 RIS 级联路径 ≈ 直连路径的 1-3 倍
channel_norm_factor = 5e13  # 含阵列增益补偿, 校准使 SINR 范围 -15~+10 dB
ris_norm_factor = 5e6       # RIS-LU 归一化, 使 RIS 级联路径≈直连路径的 2-3 倍
# 系统干扰裕量: 模拟未建模的干扰源 (邻星、地面干扰等)
# 使 SINR 在低 PT 时随功率增长, 匹配论文趋势
system_interference_margin = 1.0  # 等效噪声功率 (W), 加到 sigma2 上

# ==================== 干扰参数 ====================
# ITU 规定 LEO-GEO 干扰阈值 ζ_k ≤ -12.2 dB
zeta_default = -12.2   # 默认干扰阈值 (dBW)

# ==================== Rician 因子 (默认值) ====================
kappa_N_default = 10.0   # 卫星-地面链路 Rician 因子 (dB)
kappa_R_default = 10.0   # 地面 RIS-LU 链路 Rician 因子 (dB)
kappa_LR_default = 10.0  # 卫星-RIS 链路 Rician 因子 (dB)

# GEO 相关链路 Rician 因子
kappa_GL = 20.0         # GEO-地面 Rician 因子 (dB)
kappa_GR = 20.0         # GEO-RIS Rician 因子 (dB)
kappa_GG = 20.0         # GEO-GT Rician 因子 (dB)
kappa_LG = 20.0         # LEO-GT Rician 因子 (dB)

# ==================== 位置配置 ====================
# GEO 卫星位置 (0°N, 100°E)
geo_lat = 0.0          # 度
geo_lon = 100.0        # 度

# LEO 卫星: 纬度间隔 2.5°
leo_lat_offset = np.array([0.0, 2.5])  # 第一颗与 GEO 同纬度(inline)
leo_lon = 100.0        # 与 GEO 同经度

# 地面用户 (LU): 经纬度间隔 2°
lu_lat_offset = np.array([0.0, 2.0, 4.0, 6.0])
lu_lon_offset = np.array([0.0, 0.0, 0.0, 0.0])

# GT 与 LU 共址 (第一个 GT 与第一个 LU 共址)
gt_lat_offset = np.array([0.0, -2.0, -4.0, -6.0])
gt_lon_offset = np.array([0.0, 0.0, 0.0, 0.0])

# ==================== 算法参数 ====================
L1_max = 10            # AO 算法最大迭代次数
L2_max = 20            # 流形优化最大迭代次数
L3_max = 10            # 功率分配最大迭代次数
L4_max = 15            # ES-SP-RMO 最大迭代次数
epsilon_AO = 1e-3      # AO 收敛阈值
epsilon_ES = 1e-3      # 指数平滑收敛阈值
mu_init = 1.0          # 指数平滑初始参数
step_size_rg = 0.01    # RGD 步长

# ==================== 仿真扫描范围 ====================
# Fig 2, 3: PT 扫描范围
PT_range = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24.5])

# Fig 4, 5: ζ 扫描范围 (dBW)
zeta_range_dB = np.array([-18, -16, -14, -12.2, -10, -8, -6])

# Fig 6, 7: M 扫描范围
M_range = np.array([4, 8, 12, 16, 24, 32, 48, 64])

# Fig 8-10: κR 扫描范围 (dB)
kappa_R_range = np.array([-12, -6, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30])

# Fig 11: MSC vs SST 的 ζ 范围
zeta_msc_range_dB = np.array([-18, -16, -14, -12.2, -10, -8, -6])


def get_noise_power():
    """计算接收噪声功率 (W)"""
    noise_psd_W = 10 ** ((noise_psd - 30) / 10)  # dBm/Hz -> W/Hz
    noise_fig_W = 10 ** (noise_figure / 10)
    return noise_psd_W * BW * noise_fig_W


def get_kappa_linear(kappa_dB):
    """dB 转线性"""
    return 10 ** (kappa_dB / 10)


def compute_path_loss_db(distance_km, f_hz=f_carrier):
    """自由空间路径损耗 (dB)"""
    d_m = distance_km * 1e3
    fspl = 20 * np.log10(4 * np.pi * d_m * f_hz / c)
    return fspl


# 天线增益参数
sat_antenna_gain_dBi = 32.0    # LEO 卫星多波束天线增益 (dBi)
geo_antenna_gain_dBi = 40.0    # GEO 卫星天线增益 (dBi)
terminal_gain_dBi = 0.0        # 地面终端天线增益 (dBi)
ris_gain_dBi = 3.0             # RIS 反射增益 (dBi)


def compute_sat_antenna_gain(theta_deg, peak_gain_dbi=30):
    """
    多波束卫星天线增益模型 (近似)
    参考 ITU-R S.1528 建议的卫星天线辐射方向图
    theta_deg: 偏离波束中心的角距离 (度)
    """
    if np.isscalar(theta_deg):
        theta_deg = np.array([theta_deg])
    gain = np.full_like(theta_deg, peak_gain_dbi, dtype=float)
    # 主瓣外衰减
    mask = theta_deg > 1.0
    gain[mask] = peak_gain_dbi - 25 * np.log10(np.maximum(theta_deg[mask], 0.1))
    gain = np.maximum(gain, -10)  # 最低增益限制
    return gain if len(gain) > 1 else gain[0]


def deg2rad(deg):
    return deg * np.pi / 180


def geo_to_cartesian(lat_deg, lon_deg, alt_km):
    """经纬度转笛卡尔坐标"""
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    r = R_earth + alt_km
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])
