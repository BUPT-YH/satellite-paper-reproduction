"""
仿真参数配置
论文: Space-Time Beamforming for LEO Satellite Communications: Enabling Extremely Narrow Beams
期刊: IEEE TWC, 2026
"""

import numpy as np

# ===== 物理常数 =====
c = 3e5  # 光速 (km/s)
RE = 6371  # 地球半径 (km)

# ===== 卫星参数 =====
altitude = 530  # 轨道高度 (km)
fc = 1.9925e9  # 载频 (Hz)
wavelength = c * 1e3 / fc  # 波长 (m), ~0.1505 m
bandwidth = 5e6  # 信号带宽 (Hz)
Ts = 1 / bandwidth  # 采样周期 (s)

# ===== 天线阵列参数 =====
Nx_default = 8  # x轴天线数
Ny_default = 8  # y轴天线数
N_default = Nx_default * Ny_default  # 总天线数

# ===== 信道参数 =====
alpha = 2  # 路径损耗指数
noise_psd_dBm = -174  # 噪声功率谱密度 (dBm/Hz)
noise_power = 10 ** ((noise_psd_dBm - 30) / 1000) * bandwidth  # 噪声功率 (W)
# 简化：直接用 sigma^2 = N0 * B
sigma2 = 10 ** (noise_psd_dBm / 10) * 1e-3 * bandwidth  # W

# 多径参数
L_paths = 3  # 多径数
delta_tap = 0.5  # 衰减因子 (tap gain)

# 多普勒范围
doppler_range = 50e3  # 多普勒频移范围 [-50kHz, 50kHz]

# ===== 仿真参数 =====
n_channels = 1000  # Monte Carlo 信道实现数
P_range_dBm = np.arange(20, 51, 5)  # 发射功率范围 (dBm)
P_range_W = 10 ** ((P_range_dBm - 30) / 10)  # 转换为瓦特

# 部分连接网络默认参数
K_partial = 3  # 部分连接用户数

# 全连接网络默认参数
K_full = 4  # 全连接用户数
M_full = 3  # 全连接重复次数

# 延迟误差参数（不完美CSIT）
delay_error_max = 0.2e-9  # 最大延迟误差 (ns -> s)

# ===== Shadowed-Rician 衰落参数（平均阴影） =====
# 参考 Abdi et al. 2003, Table I, Average Shadowing
# 参数: (b, m, Omega) — 分别对应 non-LoS功率、Nakagami-m参数、LoS功率
sr_b = 0.126  # 平均多径功率
sr_m = 5.21  # Nakagami-m 参数
sr_Omega = 0.835  # LoS功率


def dBm_to_W(dBm):
    """dBm 转瓦特"""
    return 10 ** ((dBm - 30) / 10)


def W_to_dBm(W):
    """瓦特转 dBm"""
    return 10 * np.log10(W) + 30
