"""
仿真参数配置 — 精确 FCC 轨道参数
论文: Satellite Selection for In-Band Coexistence of Dense LEO Networks
期刊: IEEE TWC, 2026

轨道参数来源:
  Starlink: FCC SAT-MOD-20200417-00037 [55], SAT-LOA-20200526-00055 [56]
  Kuiper:   FCC SAT-LOA-20190704-00057 [32], [57]
"""

import numpy as np

# ====== Starlink 星座参数 (主系统, Table I) ======
# 6 个壳层, 总计 6900 颗卫星
STARLINK_SHELLS = [
    {'altitude_km': 540, 'inclination_deg': 53.2, 'num_planes': 72, 'sats_per_plane': 22},  # 1584
    {'altitude_km': 550, 'inclination_deg': 53.0, 'num_planes': 72, 'sats_per_plane': 22},  # 1584
    {'altitude_km': 560, 'inclination_deg': 97.6, 'num_planes': 4,  'sats_per_plane': 43},  # 172
    {'altitude_km': 560, 'inclination_deg': 97.6, 'num_planes': 6,  'sats_per_plane': 58},  # 348
    {'altitude_km': 570, 'inclination_deg': 70.0, 'num_planes': 36, 'sats_per_plane': 20},  # 720
    {'altitude_km': 530, 'inclination_deg': 33.0, 'num_planes': 28, 'sats_per_plane': 89},  # 2492
]
PRIMARY_TOTAL_SATS = 6900

# ====== Kuiper 星座参数 (次系统, Table II) ======
# 3 个壳层, 总计 3236 颗卫星
KUIPER_SHELLS = [
    {'altitude_km': 630, 'inclination_deg': 51.9, 'num_planes': 34, 'sats_per_plane': 34},  # 1156
    {'altitude_km': 610, 'inclination_deg': 51.9, 'num_planes': 36, 'sats_per_plane': 36},  # 1296
    {'altitude_km': 590, 'inclination_deg': 33.0, 'num_planes': 28, 'sats_per_plane': 28},  # 784
]
SECONDARY_TOTAL_SATS = 3236

# ====== 天线参数 (Table III) ======
# 卫星发射天线
TX_ANTENNA_ARRAY = 64              # 64×64 相控阵
TX_BEAMWIDTH_3DB_DEG = 1.6         # 3dB 波束宽度 (度)
TX_MAX_GAIN_DBI = 36.0             # 最大波束增益 (dBi)

# 地面用户接收天线
RX_ANTENNA_ARRAY = 32              # 32×32 相控阵
RX_BEAMWIDTH_3DB_DEG = 3.2         # 3dB 波束宽度 (度)
RX_MAX_GAIN_DBI = 30.0             # 最大波束增益 (dBi)

# ====== 功率参数 (Table III) ======
PRIMARY_MAX_EIRP_DBW_PER_HZ = -54.3    # dBW/Hz
SECONDARY_MAX_EIRP_DBW_PER_HZ = -53.3  # dBW/Hz

# ====== 噪声参数 (Table III) ======
NOISE_PSD_DBM_PER_HZ = -174.0     # dBm/Hz
NOISE_FIGURE_DB = 1.2              # dB

# ====== 频率与带宽 ======
CARRIER_FREQ_GHZ = 12.0           # GHz, Ka 波段下行
BANDWIDTH_MHZ = 250               # MHz (频率复用后每波束带宽)

# ====== 地面小区参数 ======
CELL_RADIUS_KM = 10.0             # 小区半径 ~10 km (基于 -3dB 波束轮廓)
NUM_CLUSTERS = 10                  # 簇数 N_G
NUM_CELLS_PER_CLUSTER = 127       # 每簇小区数 N_C
BEAM_CONFIGS = [8, 16, 24, 32]   # 每卫星波束数配置
FREQ_REUSE = 3                    # 频率复用因子

# ====== 仿真时间参数 ======
TIME_RESOLUTION = 0.1             # 秒
HANDOVER_PERIOD = 15.0            # 秒, T_h
PAST_WINDOW = 10.0                # 秒, T_w
MIN_ELEVATION_DEG = 25.0          # 最小仰角 (度)

# ====== 干扰阈值 ======
INR_ITU_THRESHOLD = -12.2         # dB, ITU 建议阈值

# ====== 物理常数 ======
BOLTZMANN = 1.38e-23              # J/K
C = 3e8                           # 光速 m/s
EARTH_RADIUS = 6371               # km
GM = 3.986e5                      # km^3/s^2, 地球引力常数

# ====== 仿真区域 ======
CENTER_LAT = 40.0
CENTER_LON = -100.0

# ====== 拉格朗日松弛参数 ======
MAX_SUBGRADIENT_ITERS = 50
STEP_SIZE_INIT = 0.1
