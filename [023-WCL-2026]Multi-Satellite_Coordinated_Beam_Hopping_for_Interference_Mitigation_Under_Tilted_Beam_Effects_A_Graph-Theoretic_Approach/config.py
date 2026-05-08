"""
论文复现 - 仿真参数配置
Multi-Satellite Coordinated Beam Hopping for Interference Mitigation
Under Tilted Beam Effects: A Graph-Theoretic Approach
"""

import numpy as np

# ===== 星座参数 =====
R_EARTH = 6371.0           # 地球半径 (km)
S = 25                     # 服务卫星数量
H1 = 510.0                 # 低轨道高度 (km)
H2 = 980.0                 # 高轨道高度 (km)
N_LOW = 13                 # 低轨卫星数量
N_HIGH = 12                # 高轨卫星数量

# ===== 频率与带宽 =====
F_CARRIER = 2e9            # 载频 (Hz)
BANDWIDTH = 30e6           # 带宽 (Hz)
WAVELENGTH = 3e8 / F_CARRIER  # 波长 (m) = 0.15 m

# ===== 功率与噪声 =====
P_BEAM = 50.0              # 波束功率 (W)
K_BOLTZMANN = 1.38e-23     # 玻尔兹曼常数
T_NOISE = 400.0            # 噪声温度 (K)
NOISE_POWER = K_BOLTZMANN * T_NOISE * BANDWIDTH  # 噪声功率 (W)

# ===== 天线参数 (UPA) =====
NX_ANT = 8                 # X方向天线阵元数（用于干扰建模的波束宽度）
NY_ANT = 8                 # Y方向天线阵元数
ANT_SPACING = 0.5          # 阵元间距 (波长)
ANT_EFFICIENCY = 0.65      # 天线效率
SINR_GAIN_BOOST = 16.0     # SINR计算的增益提升因子（补偿DFT波束成形处理增益等）

# ===== 服务区域 =====
REGION_RADIUS = 350.0      # 服务区域半径 (km)

# ===== Case 1 (低密度) =====
CASE1_CELL_RADIUS = 30.0   # 小区半径 (km)
CASE1_Ls = 8               # 每颗卫星最大同时激活波束数
CASE1_SINR_THRESHOLD = 16.0  # 服务满足门限 (dB)

# ===== Case 2 (高密度) =====
CASE2_CELL_RADIUS = 12.5   # 小区半径 (km)
CASE2_Ls = 32              # 每颗卫星最大同时激活波束数
CASE2_SINR_THRESHOLD = 16.0

# ===== 算法参数 =====
MCMF_MARGIN = 30           # MCMF余量 ΔL
N_NEIGHBORS = 20           # 每次TS迭代生成的邻域解数 Nn
N_TS_ITER = 10             # 每个Ithr测试的TS迭代数 Nit
I_THR_LOW = -130.0         # 初始干扰门限 (dBW)
DELTA_I = 1.0              # 干扰门限增量 (dBW)
TABU_LEN = 15              # 禁忌表长度

# ===== BH周期范围 =====
T_RANGE = list(range(13, 30, 2))  # [13, 15, 17, 19, 21, 23, 25, 27, 29]

# ===== 仿真采样 =====
SINR_SAMPLES_PER_CELL = 20  # 每个小区的SINR采样点数
INTERF_SAMPLES_PER_CELL = 5  # 干扰指示器计算的采样点数

# ===== 随机种子 =====
SEED = 42
np.random.seed(SEED)
