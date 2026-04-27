"""
仿真参数配置
论文: A Dynamic Co-Frequency Interference Analysis Model Based on TEIS for NGSO Mega-Constellations
期刊: IEEE TWC, 2026
"""
import numpy as np

# ==================== 物理常数 ====================
RE = 6371.0            # 地球半径 (km)
GM = 3.986004418e5     # 地球引力常数 GM (km³/s²)
K_BOLTZMANN = 1.380649e-23  # 玻尔兹曼常数 (J/K)
C_LIGHT = 2.998e5      # 光速 (km/s)
TE = 86164.1           # 恒星日 (s) ≈ 23h56m4s

# ==================== 仿真参数 (Table I) ====================
# 载波频率
FREQ_HZ = 20e9         # 20 GHz
FREQ_MHZ = 20000.0     # MHz (用于FSPL公式)

# 轨道高度
H_ORBIT = 550.0        # 默认轨道高度 (km)
H_ORBIT_ALT = 1200.0   # 对比场景轨道高度 (km)

# 轨道倾角 (干扰星座)
INCLINATION_1 = 80.0   # 倾角1 (度)
INCLINATION_2 = 50.0   # 倾角2 (度)
INCLINATION_3 = 40.0   # 倾角3 (度)

# Walker星座参数
N_PLANES = 72          # 默认轨道面数
N_PLANES_ALT = 36      # 对比场景轨道面数
N_SATS_PER_PLANE = 22  # 每面卫星数
N_TOTAL = N_PLANES * N_SATS_PER_PLANE  # 总卫星数 1584
F_WALKER = 1           # Walker phasing factor

# 通信卫星参数 (简化模型: 单颗卫星过境)
# 通信卫星从南向北过境，仰角从0→90→0

# 地面终端
GS_LAT = 39.0          # 被干扰终端纬度 (北京, 度)
GS_LON = 116.0         # 被干扰终端经度 (度)
THETA_MIN = 25.0       # 最小通信仰角 (度)

# 干扰终端分布
INT_RADIUS = 800.0     # 干扰终端分布半径 (km) - 扩大以覆盖更多干扰终端
MAX_INT_TERMINALS = 200  # 最大干扰终端数 - 增加以产生更连续的聚合干扰
PPP_ETA = MAX_INT_TERMINALS / (np.pi * (INT_RADIUS ** 2))  # PPP强度 (终端/km²)

# 发射接收参数
PT_DBW = 38.0          # 卫星发射功率 (dBW)
PT_W = 10 ** (PT_DBW / 10)  # 瓦特
BANDWIDTH = 400e6      # 带宽 (Hz)
GT_MAX_DBI = 30.5      # 卫星发射天线峰值增益 (dBi)
GR_MAX_DBI = 32.0      # 地面终端接收天线峰值增益 (dBi)
T_NOISE = 150.0        # 接收机噪声温度 (K)
NOISE_POWER = K_BOLTZMANN * T_NOISE * BANDWIDTH  # 噪声功率 (W)

# 天线参数
# 发射天线: ITU-R S.1528 (NGSO卫星)
D_LAMBDA_TX = 10 ** (GT_MAX_DBI / 20)  # D/λ
PHI_B_TX = 3.0 / D_LAMBDA_TX * 180 / np.pi  # 3dB波束宽度 (度), 近似
# 简化: 使用论文给定的参数
LS_TX = -6.75          # 主瓣与近旁瓣交叉点
LF_TX = -10.0          # 远旁瓣电平 (dBi)

# 接收天线: ITU-R S.465-5 (地面站)
D_LAMBDA_RX = 10 ** (GR_MAX_DBI / 20)  # D/λ
PHI_M_RX = max(2.0, 144.0 * D_LAMBDA_RX ** (-1.09))  # 度

# ==================== 仿真时间设置 ====================
DT = 10.0              # 时间步长 (秒)
T_SIM = 6000.0         # 仿真总时长 (秒), 约100分钟 (一个轨道周期的量级)

# ==================== 降雨和大气衰减参数 ====================
# 20GHz频段 (Ka波段) - 晴空/轻雨条件
GAMMA_R = 0.01             # 比衰减 (dB/km), 晴空条件, 避免低仰角衰减爆炸
RAIN_HEIGHT = 5.0           # 雨层高度 (km)
PL_ZENITH = 0.5             # 天顶总衰减 (dB), Ka波段晴空典型值

# ==================== 系统可用性参数 ====================
# PT=38 dBW 按EIRP解读 (含天线增益), 通信链路预算不再乘GT_max
# 等效链路余量 = GT_MAX_DBI = 30.5 dB
LINK_MARGIN_DB = GT_MAX_DBI  # 通信链路使用EIRP, 不重复计算GT_max

C0 = 500e6              # 最小数据速率 500 Mbps (Ka波段400MHz信道典型值)
S0 = 2 ** (C0 / BANDWIDTH) - 1  # SINR阈值 ≈ 1.4 dB
