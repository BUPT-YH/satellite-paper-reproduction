"""
仿真参数配置
论文: Joint Resource Management and Load Balancing in Multi-Satellite
      Beam Hopping With Interference Suppression: An Energy Minimization Perspective
期刊: IEEE TWC, Vol. 25, 2026
"""

import numpy as np

# ==================== 卫星网络参数 ====================
NUM_SATELLITES = 3          # LEO 卫星数量 |S| (默认 3, Fig.2 中变化)
NUM_BEAMS = 4               # 每卫星波束数 Nb
NUM_CELLS = 28              # 地面小区数 |K|
NUM_FREQ_SEGMENTS = 4       # 频率段数 |L|

# ==================== 轨道参数 ====================
ALTITUDE = 1200e3           # 轨道高度 (m)
CARRIER_FREQ = 20e9         # 载波频率 20 GHz (Ka 频段)
EARTH_RADIUS = 6371e3       # 地球半径 (m)
SAT_DISTANCE = 207e3        # 相邻卫星间距 (m)
C_LIGHT = 3e8               # 光速 (m/s)

# ==================== 频谱参数 ====================
BANDWIDTH_PER_SEG = 250e6   # 每频率段带宽 W = 250 MHz
TOTAL_BANDWIDTH = BANDWIDTH_PER_SEG * NUM_FREQ_SEGMENTS  # 总带宽 1 GHz

# ==================== 功率参数 ====================
P_MAX = 60.0                # 每卫星最大发射功率 (W)
P_CIRCUIT = 5.0             # 电路功耗 (W) — 论文能耗主要为发射功率
P_ISL = 5.0                 # 激光星间链路终端功耗 (W)
N0 = 4e-21                  # 噪声功率谱密度 (W/Hz), -174 dBm/Hz
NOISE_POWER = N0 * BANDWIDTH_PER_SEG  # 每段噪声功率 ≈ 1e-12 W

# ==================== 时隙参数 ====================
T0 = 10e-3                  # 时隙长度 (s) = 10 ms
M0 = 0.5e6                  # 数据包大小 (bit) = 0.5 Mbit

# ==================== 天线参数 ====================
# ITU-R S.1528 卫星发射天线
SAT_ANTENNA_GAIN_DBI = 36.0       # 卫星天线峰值增益 (dBi)
SAT_BEAM_WIDTH_DEG = 2.0          # 3dB 波束宽度 (度)
SAT_SIDELOBE_DBI = -10.0          # 旁瓣电平 (dBi)

# ITU-R S.465-6 地面接收天线
RX_ANTENNA_GAIN_DBI = 0.0         # 用户终端增益 (dBi)

# GSO 地面站天线
GSO_STATION_GAIN_DBI = 40.0       # GSO 地面站增益 (dBi)

# ==================== 业务参数 ====================
ARRIVAL_RATE_MIN = 80e6     # 最小到达速率 (bps) = 80 Mbps
ARRIVAL_RATE_MAX = 320e6    # 最大到达速率 (bps) = 320 Mbps
DRIFT_INTERVAL = 50         # 业务漂移间隔 (时隙数)
A_MAX = None                # 最大到达包数/时隙, 运行时计算

# ==================== 干扰参数 ====================
Z_MAX_RANGE_DBW = np.array([-140, -135, -130, -125, -120])  # 干扰阈值范围 (dBW)
Z_MAX_DEFAULT_DBW = -130    # 默认干扰阈值 (dBW)
NUM_GSO_STATIONS = 5        # 含 GSO 地面站的小区数
FREQ_OVERLAP_PER_STATION = 2  # 每站重叠频率段数

# ==================== 负载均衡参数 ====================
C_MAX_DEFAULT = 20          # ISL 传输上限 (包/时隙)

# ==================== Lyapunov 参数 ====================
V_DEFAULT = 200             # 权衡系数 V
V_RANGE = [50, 100, 200, 400]  # Fig.3 中的 V 取值

# ==================== 算法参数 ====================
BCD_MAX_ITER = 5            # BCD 最大迭代
MPMM_ROUNDS = 3             # MPMM 轮数 Θ
MPMM_MM_ITER = 3            # MM 迭代数 Ψ
SCA_ITER = 3                # SCA 迭代数 Ξ
MPMM_RHO = 1.5              # 惩罚系数增长率
MPMM_BETA_INIT = 1.0        # 初始惩罚系数
CONV_TOL = 1e-3             # 收敛阈值

# ==================== 仿真参数 ====================
NUM_TIME_SLOTS = 300        # 仿真时隙数 (增加以获得更稳定的稳态平均)
WARMUP_SLOTS = 100          # 预热时隙数 (增加以确保进入稳态)
DEMAND_DEFAULT = 20e9       # 默认通信需求 20 Gbps
DEMAND_RANGE = np.array([10, 15, 20, 25, 30]) * 1e9  # Fig.6 需求范围

# ==================== 信道模型参数 ====================
ATMOS_LOSS_DB = 3.0         # 大气衰减 (dB)
RAIN_MARGIN_DB = 2.0        # 雨衰余量 (dB)

# ==================== 辅助函数 ====================
def db_to_linear(db):
    """dB 转线性值"""
    return 10 ** (np.array(db) / 10.0)

def linear_to_db(linear):
    """线性值转 dB"""
    return 10 * np.log10(np.maximum(np.array(linear), 1e-30))
