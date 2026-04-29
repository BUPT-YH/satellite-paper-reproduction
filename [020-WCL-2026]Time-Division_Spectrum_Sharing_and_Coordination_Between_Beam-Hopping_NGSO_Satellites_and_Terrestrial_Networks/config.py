"""
仿真参数配置 — 基于论文 Table I
论文: Time-Division Spectrum Sharing and Coordination Between
      Beam-Hopping NGSO Satellites and Terrestrial Networks
期刊: IEEE Wireless Communications Letters, Vol. 15, 2026
"""
import numpy as np

# ===== 星座参数 (Starlink 参考) =====
ALTITUDE = 550              # 轨道高度 (km)
NUM_ORBITAL_PLANES = 48     # 轨道面数
NUM_SATS_PER_PLANE = 13    # 每轨道面卫星数
INCLINATION = 53.0          # 轨道倾角 (度)
NUM_VISIBLE_SATS = 4        # 模拟区域可见卫星数
EARTH_RADIUS = 6371.0       # 地球半径 (km)
RE = EARTH_RADIUS

# ===== 波束参数 =====
BEAM_WIDTH_3DB = 1.5        # 3dB 波束宽度 (度)
NUM_BEAMS_PER_SAT = 8       # 每卫星波束数 K
SAT_GAIN_DBI = 40.0         # 卫星天线增益 (dBi)

# ===== 跳波束参数 =====
W = 32                      # BH 模式数 (每调度周期的模式数)
L = 200                     # 调度周期重复次数

# ===== 地面小区参数 =====
NUM_TERR_CELLS = 60         # 地面小区数 M
CELL_RADIUS = 3.0           # 小区半径 (km)
AREA_SIZE = 200.0            # 模拟区域边长 (km)

# ===== 卫星小区参数 =====
NUM_SAT_CELLS = 128         # 卫星小区总数 N (32 per satellite)

# ===== 时间参数 =====
GP_S2C = 5e-6               # 卫星→地面保护间隔 (s) = 5μs
GP_C2S = 15e-6              # 地面→卫星保护间隔 (s) = 15μs
TIMESLOT_LEN = 5e-3         # 默认时隙长度 (s) = 5ms
T_SYM = 71.4e-6             # 符号长度 (s), 对应 14 OFDM symbols/ms
N_SYM = 14                  # 每时隙 OFDM 符号数

# ===== 频率与带宽参数 =====
CARRIER_FREQ = 12.0         # 载波频率 (GHz), Ku-band
BANDWIDTH_TOTAL = 250.0     # 系统总带宽 (MHz)
# 有效每小区带宽 (考虑频率复用和多小区共享)
BW_PER_TERR_CELL = 40.0     # 地面小区有效带宽 (MHz)
BW_PER_SAT_CELL = 30.0      # 卫星小区有效带宽 (MHz)

# ===== 用户与业务参数 =====
USER_SAT_TERR_RATIO = 0.1   # 卫星用户:地面用户 = 1:10
DEMAND_RATE = 5.0           # 每用户泊松到达率 (Mbps)
USER_TOTAL_RANGE = np.arange(100, 501, 50)  # Fig.5 总用户数范围

# ===== 链路参数 =====
# 地面小区下行频谱效率 (bps/Hz)
SE_TERR = 4.0
# 卫星小区下行频谱效率 (bps/Hz)
SE_SAT = 1.5
# 干扰损耗因子 (无干扰避免时的 SINR 折减)
INTERFERENCE_LOSS_TERR = 0.35  # 地面小区受干扰折减
INTERFERENCE_LOSS_SAT = 0.25   # 卫星小区受干扰折减

# ===== Fig.6 时隙长度范围 =====
TIMESLOT_LEN_RANGE = np.arange(1, 10.5, 0.5) * 1e-3  # 1ms ~ 10ms

# ===== DSS 参数 =====
DSS_GUARD_BAND_RATIO = 0.1  # DSS 保护带开销比例
DSS_SHARE_RATIO = 0.7       # DSS 可共享频谱比例

# ===== 随机种子 =====
RANDOM_SEED = 42
