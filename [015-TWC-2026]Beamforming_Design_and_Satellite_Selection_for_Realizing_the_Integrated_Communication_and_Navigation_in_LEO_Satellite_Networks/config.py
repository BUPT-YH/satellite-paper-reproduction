"""
仿真参数配置
论文: Beamforming Design and Satellite Selection for Realizing the ICAN in LEO Satellite Networks
期刊: IEEE TWC, 2026
"""
import numpy as np

# ===== 星座参数 =====
S = 12                  # 可见卫星数（默认）
C = 7                   # 用户数（默认）
Nx, Ny = 4, 4           # UPA 天线阵列尺寸 (N = 16)
N = Nx * Ny             # 总天线数
K = 4                   # 每颗卫星最大波束数
I = 6                   # 每个UE的服务卫星数（默认）

# ===== 物理参数 =====
c0 = 3e8                # 光速 (m/s)
f_carrier = 2e9         # 载波频率 2 GHz
wavelength = c0 / f_carrier
B = 10e6                # 带宽 10 MHz

# ===== 功率参数 =====
P_max = 30              # 最大发射功率 30 dBm
P_max_watt = 10 ** (P_max / 10) / 1000  # 转换为瓦特
noise_power_dbm = -90   # 噪声功率 -90 dBm
noise_power_watt = 10 ** (noise_power_dbm / 10) / 1000

# ===== QoS 约束 =====
gamma_com = 12.5e6      # 最小速率阈值 12.5 Mbps (bps)
gamma_nav = 6           # 最大 GDOP 阈值

# ===== 算法参数 =====
rho = 0.5               # 权重因子 (0: 纯导航, 1: 纯通信)
J_DC = 10               # DC 规划最大迭代次数
J_OCF = 20              # OCF 博弈最大迭代次数
L_tabu = 10             # 禁忌表长度

# ===== 仿真范围 =====
C_range = np.arange(4, 10)         # UE 数范围: Fig. 2, 3, 4
S_range = np.arange(10, 17)        # 可见卫星数范围: Fig. 5, 6, 7
I_range = [5, 6, 7]                # 服务卫星数: Fig. 3, 8
rho_range = np.arange(0, 1.05, 0.1) # 权重因子范围: Fig. 8
gamma_com_range = np.array([12.5, 15, 17.5, 20]) * 1e6  # 速率阈值范围: Fig. 4

# ===== 卫星轨道参数 =====
altitude_min = 600e3    # 最低轨道高度 600 km
altitude_max = 1200e3   # 最高轨道高度 1200 km
R_earth = 6371e3        # 地球半径

# ===== 随机种子（可复现性）=====
SEED = 42
