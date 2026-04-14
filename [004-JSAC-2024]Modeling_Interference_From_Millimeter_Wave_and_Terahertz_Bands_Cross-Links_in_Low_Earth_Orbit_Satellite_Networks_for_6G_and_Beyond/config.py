"""
仿真参数配置
论文: Modeling Interference From mmWave and THz Bands Cross-Links in LEO Satellite Networks
期刊: IEEE JSAC, Vol. 42, No. 5, May 2024
"""

import numpy as np

# ==================== 物理常数 ====================
R_E = 6371e3          # 地球半径 (m)
k_B = 1.38e-23        # 玻尔兹曼常数 (J/K)
c = 3e8               # 光速 (m/s)

# ==================== mmWave 参数 (Table II) ====================
mmWave = {
    'name': 'mmWave',
    'freq': 38e9,       # 频率 38 GHz [51]
    'bandwidth': 400e6, # 带宽 400 MHz [51]
    'P_Tx': 1000.0,     # 发射功率 1000 W (60 dBm) [52]
    'T_sys': 100,       # 系统温度 100 K [54]
    'beamwidths': [5, 10, 30, 40],  # 波束宽度 (度)
}

# ==================== sub-THz 参数 (Table II) ====================
subTHz = {
    'name': 'sub-THz',
    'freq': 130e9,       # 频率 130 GHz [52]
    'bandwidth': 10e9,   # 带宽 10 GHz [53]
    'P_Tx': 0.5,         # 发射功率 0.5 W (27 dBm) [53]
    'T_sys': 100,        # 系统温度 100 K [54]
    'beamwidths': [1, 3, 5],  # 波束宽度 (度)
}

# ==================== 轨道参数 ====================
h_default = 500e3      # 默认轨道高度 500 km (m)

# ==================== 单轨道场景参数 (Figure 5) ====================
single_orbit_params = {
    'h': 500e3,                      # 轨道高度 (m)
    'N_range': np.arange(5, 201),    # 卫星数量范围
    'SIR_limit_dB': 1.9,             # 理论SIR极限 (dB), Eq.(10)
}

# ==================== 偏移轨道场景参数 (Figure 9b) ====================
shifted_orbit_params = {
    'h': 500e3,                      # 轨道高度 (m)
    'inclination': 50,               # 轨道倾角 (度)
    'Delta_Omega': 90,               # RAAN偏移 (度)
    'alpha_range': np.arange(1, 41), # 波束宽度范围 (度)
    'N_values': [50, 100],           # 卫星数量
    'Delta_beta': 0,                 # 相对角偏移 (常数，同高度)
}

# ==================== 完整双星座场景参数 (Figure 10) ====================
full_constellation_params = {
    'h': 500e3,                      # 星座1轨道高度 (m)
    'h_S': 510e3,                    # 星座2轨道高度 (m)
    'inclination': 50,               # 轨道倾角 (度)
    'n_orbits': 10,                  # 轨道面数
    'N_range': np.arange(10, 501),   # 每轨道卫星数
    'Delta_Omega_spacing': 36,       # 轨道面间距 (度): 360/10
    'Delta_beta': 0,                 # 相对角偏移
}


def get_wavelength(freq):
    """根据频率计算波长"""
    return c / freq


def get_antenna_gain(alpha_deg):
    """
    锥形天线增益 Eq. G = 2/(1-cos(alpha/2))
    alpha: 波束宽度 (度)
    """
    alpha_rad = np.radians(alpha_deg)
    return 2.0 / (1.0 - np.cos(alpha_rad / 2))


def get_noise_power(band_params):
    """计算热噪声功率 PN = k * T * B"""
    return k_B * band_params['T_sys'] * band_params['bandwidth']
