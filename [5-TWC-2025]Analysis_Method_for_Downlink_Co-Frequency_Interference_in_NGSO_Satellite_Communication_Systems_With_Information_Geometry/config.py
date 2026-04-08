"""
论文复现 - 仿真参数配置
Paper: Analysis Method for Downlink Co-Frequency Interference in NGSO
       Satellite Communication Systems With Information Geometry
Journal: IEEE TWC, Vol. 24, No. 10, Oct 2025
"""

import numpy as np

# ==================== 地球参数 ====================
# WGS84 椭球参数
a_earth = 6378136.49       # 半长轴 (m)
b_earth = 6356755.00       # 半短轴 (m)
e_earth = np.sqrt((a_earth**2 - b_earth**2) / a_earth**2)  # 偏心率
c_light = 3e8              # 电磁波传播速度 (m/s)

# ==================== OneWeb 卫星系统 (被干扰系统) ====================
# OneWeb 卫星参数
Pt_OW = 8.5               # 发射功率 (dBW)
Gt_OW_peak = 25.9          # 发射天线峰值增益 (dBi)
D_OW_sat = 1.0             # 卫星天线直径 (m)
HPBW_OW_sat = 3.2          # 半功率波束宽度 (度)
lat_OW_sat = 42.635         # 星下点纬度 (度)
lon_OW_sat = 114.421        # 星下点经度 (度)
h_OW_sat = 1202.01 * 1e3   # 卫星高度 (m)

# OneWeb 地球站参数
Gr_OW_es = 32.0            # 接收天线峰值增益 (dBi)
D_OW_es = 1.2              # 接收天线直径 (m)
lat_OW_es = 43.12           # 地球站纬度 (度)
lon_OW_es = 115.73          # 地球站经度 (度)
h_OW_es = 0                 # 地球站高度 (m)

# ==================== Starlink 干扰卫星系统 ====================
# 干扰卫星 1
Pt_SL1 = 8.0               # 发射功率 (dBW)
Gt_SL1_peak = 44.0          # 发射天线峰值增益 (dBi)
HPBW_SL1 = 8.9             # 半功率波束宽度 (度)
D_SL1 = 0.6                # 天线直径 (m)
lat_SL1 = 43.509            # 星下点纬度 (度)
lon_SL1 = 116.851           # 星下点经度 (度)
h_SL1 = 479.3 * 1e3        # 卫星高度 (m)

# 干扰卫星 2
Pt_SL2 = 8.0               # 发射功率 (dBW)
Gt_SL2_peak = 34.0          # 发射天线峰值增益 (dBi)
HPBW_SL2 = 8.9             # 半功率波束宽度 (度)
D_SL2 = 0.6                # 天线直径 (m)
lat_SL2 = 41.905            # 星下点纬度 (度)
lon_SL2 = 115.589           # 星下点经度 (度)
h_SL2 = 492.3 * 1e3        # 卫星高度 (m)

# ==================== 通信参数 ====================
freq = 11.51e9              # 通信频率 (Hz) = 11.51 GHz
bandwidth = 32e6            # 通信带宽 (Hz) = 32 MHz
T_noise = 290               # 接收机噪声温度 (K)
k_boltz = 1.38e-23          # 玻尔兹曼常数 (J/K)

# ==================== 衰减模型参数 ====================
# 雨衰模型 - ITU-R P.618-12
rain_outage_rate = 0.1      # 中断率

# 云雾衰减模型 - ITU-R P.840-7
cloud_ceiling = 3e3         # 云层上限 (m)
cloud_thickness = 0.5e3     # 云层厚度 (m)
cloud_temp = 273.15         # 云层温度 (K)
liquid_water_density = 0.0001  # 液态水含量密度 (kg/m^3)

# ==================== 信息几何参数 ====================
Pf = 0.03                  # 虚警率
Q_monte_carlo = 1000       # 蒙特卡洛仿真次数
N_independent_samples = 10  # 独立采样次数
AIRM_max_iter = 30         # AIRM 迭代次数
AIRM_step = 0.1            # AIRM 迭代步长

# ==================== Fig.8 3D可视化参数 ====================
# Starlink 卫星星下点范围
vis_lat_range = (35, 45)    # 纬度范围 (度)
vis_lon_range = (110, 120)  # 经度范围 (度)
vis_grid_step = 0.01        # 经纬度网格精度 (度)

# 三个 OneWeb 地球站位置
earth_stations = [
    (40.12, 116.23),  # 站1: (纬度, 经度)
    (40.0,  112.0),   # 站2
    (43.0,  114.0),   # 站3
]
