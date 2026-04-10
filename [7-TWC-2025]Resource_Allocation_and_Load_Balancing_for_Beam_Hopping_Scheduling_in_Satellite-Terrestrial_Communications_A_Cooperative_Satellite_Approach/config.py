"""
仿真参数配置
论文: Resource Allocation and Load Balancing for Beam Hopping Scheduling
      in Satellite-Terrestrial Communications: A Cooperative Satellite Approach
期刊: IEEE TWC, Vol.24, No.2, Feb. 2025
"""

import numpy as np

# ============ 星座参数 ============
Ns = 16              # 卫星数量 (4×4星座)
Nc = 160             # 地面小区总数
Nb = 4               # 每颗卫星波束数
NL = 4               # 频率段数
omega_s = 37         # 每颗卫星覆盖小区数 |Ω(s)|

# ============ 轨道与信道参数 ============
altitude = 1000e3    # 轨道高度 (m)
freq = 20e9          # Ka频段频率 20 GHz
c = 3e8              # 光速 (m/s)
wavelength = c / freq  # 波长 0.015 m
antenna_radius = 0.1  # 卫星天线孔径半径 (m)
user_antenna_gain_dbi = 37.0  # 用户终端天线增益 (dBi)

# ============ 时间参数 ============
T0 = 10e-3           # 时隙长度 10 ms
Tth = 10             # TTL阈值 (时隙数)
Tb = 200             # 负载均衡重算间隔 (时隙数)
traffic_walk_interval = 200  # 业务随机游走间隔
traffic_walk_range = 10      # 随机游走范围 (Mbps)

# ============ 功率与噪声参数 ============
P_sat = 40.0         # 单星最大功率 (W)
P_tot = Ns * P_sat   # 星座总功率 (W)
B0 = 250e6 / NL      # 每频率段带宽 = 62.5 MHz
n0 = 4e-21           # 噪声功率谱密度 (W/Hz)

# ============ 业务参数 ============
traffic_min = 35     # 最小初始业务速率 (Mbps)
traffic_max = 105    # 最大初始业务速率 (Mbps)

# ============ DQN 参数 ============
gamma_drl = 0.99     # 折扣因子 ρ
epsilon_start = 1.0  # 初始探索率
epsilon_end = 0.05   # 最终探索率
epsilon_decay = 1500 # 探索率衰减步数 (加快收敛)
batch_size = 64      # 经验回放批量
replay_size = 5000   # 经验池大小
Cu = 100             # 目标网络更新间隔
learning_rate = 1e-3 # 学习率

# DQN网络结构
dqn_layers = [omega_s * Tth, 256, 128, 64, 32, omega_s]

# ============ 优化参数 ============
beta = 0.7           # 吞吐量-延迟折衷系数
kappa = 0.1          # 延迟度量中最大延迟权重
Y0 = 1000.0          # 吞吐量归一化常数 (Mbps量级)
Gamma0 = 1.0         # 延迟度量归一化常数

# MM算法参数
mm_max_iter = 5      # MM算法最大迭代次数 (简化)
mm_tol = 1e-3        # MM算法收敛阈值

# 负载均衡参数
lb_max_iter = 15     # 负载均衡迭代次数
lb_tol = 1e-3        # 负载均衡收敛阈值
eta = 0.1            # SCA稳定性参数
delta_lb = 1e-3      # 可行域参数 δ

# ============ 仿真参数 ============
train_slots = 1000   # DQN训练时隙数
eval_slots = 500     # 评估时隙数

# ============ 绘图参数 ============
DPI = 300
FIG_SIZE_SINGLE = (8, 6)
FIG_SIZE_DOUBLE = (14, 5)
COLORS = {
    'proposed': '#E63946',
    'without_drl': '#457B9D',
    'without_ra': '#2A9D8F',
    'without_lb': '#E9C46A',
    'original': '#264653',
    'drl_avoid': '#F4A261',
    'max_uswg': '#6A994E',
    'pre_scheduling': '#BC6C25',
}
MARKERS = {
    'proposed': 'o',
    'without_drl': 's',
    'without_ra': '^',
    'without_lb': 'D',
    'original': 'v',
    'drl_avoid': 'P',
    'max_uswg': 'X',
    'pre_scheduling': 'p',
}
