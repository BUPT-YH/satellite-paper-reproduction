"""
仿真参数配置 — 基于 Table IV, V, VI
Joint Illumination, Power, and Band Allocation for Multi-Beam LEO Satellites
With Beam-Hopping Using Mixed-Integer Linear Programming
"""
import numpy as np

# ===== 卫星与轨道参数 =====
H_SAT = 1100          # 卫星高度 (km)
FC = 20e9              # 载波频率 20 GHz (Ka-band)
EL_MIN = 25            # 最小仰角 (degrees)
C_LIGHT = 3e8          # 光速 (m/s)

# ===== 波束与带宽参数 =====
BT = 125e6             # 总带宽 125 MHz
N_FREQ = 5             # 频率分块数
BW_PER_BIN = BT / N_FREQ  # 每块带宽 25 MHz
N_ILL = 4              # 同时照射波束数
PT = 18.0              # 最大发射功率 (W)

# ===== MCS 表 (基于 5G NR, 15 种调制编码方案) =====
# (modulation, code_rate_x1024, SE_bit_per_hz, req_CN_dB)
MCS_TABLE = [
    ("QPSK",  120, 0.2344, -2.0),
    ("QPSK",  193, 0.3770,  0.0),
    ("QPSK",  308, 0.6016,  2.0),
    ("QPSK",  449, 0.8770,  4.0),
    ("QPSK",  602, 1.1758,  5.5),
    ("16QAM", 378, 1.4766,  7.0),
    ("16QAM", 490, 1.9141,  9.0),
    ("16QAM", 616, 2.4063, 11.0),
    ("16QAM", 719, 2.8086, 13.0),
    ("64QAM", 466, 2.7305, 15.0),
    ("64QAM", 567, 3.3223, 17.0),
    ("64QAM", 666, 3.9023, 19.5),
    ("64QAM", 772, 4.5234, 21.5),
    ("64QAM", 873, 5.1152, 23.5),
    ("64QAM", 948, 5.5547, 25.5),
]
M_NUM = len(MCS_TABLE)

# ===== 天线参数 =====
# 用户终端增益 (VSAT Ka-band)
GRX_DBI = 35.0
# 等效噪声温度
TEQ_NOISE = 290  # K
K_BOLTZMANN_DBW = -228.6  # dBW/K/Hz

# ===== 链路预算参数 =====
# 典型斜距 (1100 km 高度, 25° 仰角)
SLANT_RANGE = 1300e3  # m
# 自由空间损耗
FSL_DB = 20 * np.log10(4 * np.pi * SLANT_RANGE * FC / C_LIGHT)

# 卫星天线增益 (根据 beamwidth 计算)
# Reduced: 32.2° → GTX ≈ 14.6 dBi; Enlarged: 11.85° → GTX ≈ 23.3 dBi
GTX_REDUCE = 14.6
GTX_ENLARGE = 23.3

# ===== 场景定义 =====
REDUCED = {
    "name": "Reduced",
    "NC": 37,
    "NU_range": [5, 7, 10, 15, 20],
    "NT": 30,
    "TT": 3.0,        # 总仿真时间 (s)
    "GTX": GTX_REDUCE,
    "NT_split": 10,    # time-split 窗口大小
}

ENLARGED = {
    "name": "Enlarged",
    "NC": 271,
    "NU_range": [20, 40, 60, 80, 100],
    "NT": 100,
    "TT": 10.0,
    "GTX": GTX_ENLARGE,
    "NT_split": 10,
}

# ===== GA 参数 (Table V) =====
GA_CONFIG = {
    "n_gen": 100,
    "pop_size": 100,
    "crossover_rate": 1.0,
    "mutation_rate": 0.1,
    "db_local_search_rate": 0.2,
}

# ===== MILP 求解器参数 (Table VI) =====
MILP_CONFIG = {
    "mip_gap": 0.01,
    "time_limit": 120,  # 秒 (缩短用于复现)
    "threads": 4,
}

# ===== 优化目标权重 =====
BETA = 0.7  # 默认权重参数
N_SEEDS = 10  # 随机种子数 (论文: 10 次平均)
DU_MAX = 30  # 最大用户需求 (Mbps), U(0, 30)

# ===== 预计算功率和容量表 =====
def compute_power_and_capacity(gtx_db):
    """
    预计算 P_req[m, n] (W) 和 D[m, n] (Mbps)
    基于链路预算公式 (1)
    """
    P_req = np.zeros((M_NUM, N_FREQ))
    D = np.zeros((M_NUM, N_FREQ))

    for m in range(M_NUM):
        _, _, se, cn_req = MCS_TABLE[m]
        for n in range(1, N_FREQ + 1):
            # 噪声功率: K + Teq + n * BW_per_bin
            bw_db = 10 * np.log10(n * BW_PER_BIN)
            noise_dbw = K_BOLTZMANN_DBW + 10 * np.log10(TEQ_NOISE) + bw_db

            # 需要的发射功率: P = C/N_req + noise - GTX - GRX + FSL
            p_dbw = cn_req + noise_dbw - gtx_db - GRX_DBI + FSL_DB
            P_req[m, n - 1] = 10 ** (p_dbw / 10)  # 转换为瓦特

            # 容量: D = SE * n * BW_per_bin
            D[m, n - 1] = se * n * (BW_PER_BIN / 1e6)  # Mbps

    return P_req, D
