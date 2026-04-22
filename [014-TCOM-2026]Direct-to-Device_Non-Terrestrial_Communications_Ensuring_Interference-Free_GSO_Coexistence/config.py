"""
论文复现 - 仿真参数配置
Direct-to-Device Non-Terrestrial Communications Ensuring Interference-Free GSO Coexistence
IEEE TCOM, 2026
"""
import numpy as np

# ===================== 物理常数 =====================
R_EARTH = 6371.0          # 地球半径 (km)
C_LIGHT = 3e8             # 光速 (m/s)
K_BOLTZMANN = 1.38e-23    # 玻尔兹曼常数 (J/K)

# ===================== 载波参数 =====================
F_CARRIER = 11.7e9         # 载波频率 (Hz)
WAVELENGTH = C_LIGHT / F_CARRIER  # 波长 (m)
BANDWIDTH = 100e6          # 带宽 (Hz)

# ===================== LEO 星座参数 (Table II) =====================
# Shell 1
SHELL1_ALTITUDE = 550.0    # 轨道高度 (km)
SHELL1_NUM_ORBITS = 72     # 轨道面数
SHELL1_SATS_PER_ORBIT = 20 # 每轨道卫星数 (1440/72=20)
SHELL1_INCLINATION = 53.0  # 轨道倾角 (度)

# Shell 2
SHELL2_ALTITUDE = 540.0    # 轨道高度 (km)
SHELL2_NUM_ORBITS = 72     # 轨道面数
SHELL2_SATS_PER_ORBIT = 20 # 每轨道卫星数
SHELL2_INCLINATION = 53.2  # 轨道倾角 (度)

# 每壳卫星数
SATS_PER_SHELL = 1440
TOTAL_SATS = 2880

# ===================== GSO 参数 =====================
GSO_ALTITUDE = 35786.0     # GSO 轨道高度 (km)
GSO_RADIUS = R_EARTH + GSO_ALTITUDE  # GSO 轨道半径 (km)
GSO_LATITUDE = 0.0         # GSO 卫星纬度 (度) - 赤道上空
# GSO 经度间隔: 假设每隔 2° 分布一个 GSO 轨道位置
GSO_SLOT_SPACING = 2.0     # GSO 轨道位置间隔 (度)
GSO_NUM_SLOTS = 180        # GSO 轨道位置数 (-180° 到 180° 每隔 2°)

# ===================== 卫星发射天线参数 (Table III) =====================
DS = 0.8                   # 发射天线直径 (m)
LN = -30.0                 # 天线损耗 (dB)
LF = 0.0                   # 馈线损耗 (dBi)
ZS = 1.0                   # 天线模型参数
ALPHA_S = 1.5              # 天线模型参数
AS_PARAM = 2.58            # 天线模型参数
BS_PARAM = 6.32            # 天线模型参数
SAT_HPBW = 4.0             # 卫星波束半功率波束宽度 (度)
P_MAX_DBW = 2.0            # 最大波束功率 (dBW)
P_MAX_W = 10**(P_MAX_DBW/10)  # 转换为瓦特

# ===================== GSO 接收终端参数 =====================
GSO_TERMINAL_DIAMETER = 0.7  # GSO 终端天线直径 (m)

# ===================== 用户终端天线参数 =====================
NX, NY = 4, 4              # 阵列元素数
NUM_ELEMENTS = NX * NY     # 总元素数 (16)
DX = 0.45 * WAVELENGTH     # x 方向元素间距 (m)
DY = 0.45 * WAVELENGTH     # y 方向元素间距 (m)
ELEMENT_GAIN_DBI = 8.0     # 元素增益 (dBi)
ELEMENT_GAIN_DB = ELEMENT_GAIN_DBI - 2.15  # 转换为 dBd → dB (约等于 dBi - 2.15)
ELEMENT_GAIN_LINEAR = 10**(ELEMENT_GAIN_DBI/10)

# 元素方向图参数 [33]
OMEGA_E_3DB = 80.0         # 元素 x 方向 3dB 波束宽度 (度)
PSI_E_3DB = 80.0           # 元素 y 方向 3dB 波束宽度 (度)
AM = 30.0                  # 前后比 (dB)
SLA = 30.0                 # 旁瓣电平 (dB)

# CBS 码本参数
MW = 2                     # 仰角层数 (M_omega)
NPSI = 12                  # 方位数 (N_psi)
L_CODEBOOK = MW * NPSI     # 码本字数 (24) + 1 broadside = 25
B_OMEGA = 24.0             # 仰角方向码本字 HPBW (度)
B_PSI = 30.0               # 方位方向码本字 HPBW (度)

# ===================== 仿真场景参数 =====================
MIN_ELEVATION = 30.0       # 最小仰角 (度)
BETA_MIN = MIN_ELEVATION
NOISE_TEMP = 260.0         # 噪声温度 (K)
NOISE_POWER = K_BOLTZMANN * NOISE_TEMP * BANDWIDTH  # 噪声功率 (W)
NOISE_POWER_DBW = 10 * np.log10(NOISE_POWER)

DELTA_T = 4.0              # 时间步长 (秒)
USER_LAT = 1.5             # 用户纬度 (度)
USER_LON = 16.5            # 用户经度 (度)

# 轨道周期 (秒)
SHELL1_PERIOD = 2 * np.pi * np.sqrt((R_EARTH + SHELL1_ALTITUDE)**3 / (3.986e5))
SHELL2_PERIOD = 2 * np.pi * np.sqrt((R_EARTH + SHELL2_ALTITUDE)**3 / (3.986e5))
MAX_PERIOD = max(SHELL1_PERIOD, SHELL2_PERIOD)  # 较长轨道周期

# ===================== 高度对比场景 (Fig. 4b) =====================
SHELL1_ALT_HIGH = 1150.0   # 高轨道 Shell 1 (km)
SHELL2_ALT_HIGH = 1140.0   # 高轨道 Shell 2 (km)
