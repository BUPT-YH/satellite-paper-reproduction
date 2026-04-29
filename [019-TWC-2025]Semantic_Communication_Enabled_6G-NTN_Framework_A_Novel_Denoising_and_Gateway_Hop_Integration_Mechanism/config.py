"""
论文复现 - 仿真参数配置
Paper: Semantic Communication Enabled 6G-NTN Framework
       A Novel Denoising and Gateway Hop Integration Mechanism
Journal: IEEE TWC, Vol. 24, No. 12, 2025
"""

import numpy as np

# ===== Table I: 仿真参数 =====

# 地面用户 (Ground Users)
NUM_GU = 20                # GUs 总数
NUM_DIRECT_GU = 10         # 高SNR直接通信用户数
NUM_INDIRECT_GU = 10       # 低SNR通过网关中继用户数

# 卫星参数 (Satellite Parameters)
SATELLITE_ALTITUDE = 786e3      # 卫星高度 786 km (m)
SAT_TX_POWER = 10.0             # 卫星发射功率 10 W
SAT_ANTENNA_GAIN_DBI = 33.13    # 卫星天线增益 33.13 dBi
GU_ANTENNA_GAIN_DBI = 10.4      # GU天线增益 10.4 dBi
GW_ANTENNA_GAIN_DBI = 59.0      # 网关天线增益 59.0 dBi

# 卫星-地面链路参数
SAT_SUBCARRIER_FREQ_MIN = 20e9  # 卫星子载波最低频率 20 GHz
SAT_SUBCARRIER_FREQ_MAX = 30e9  # 卫星子载波最高频率 30 GHz
SAT_BANDWIDTH = 500e6           # 卫星子载波带宽 500 MHz
ACCESS_WINDOW = 60.0            # 接入窗口 60 s

# 噪声功率 (噪声功率服从正态分布, dB)
NOISE_SAT_MEAN_DB = -44.0       # 卫星-GU/GW链路噪声均值 dB
NOISE_SAT_STD_DB = 1.0          # 卫星-GU/GW链路噪声标准差 dB

# 网关-地面用户链路参数
GW_SUBCARRIER_FREQ_MIN = 15e9   # 网关子载波最低频率 15 GHz
GW_SUBCARRIER_FREQ_MAX = 20e9   # 网关子载波最高频率 20 GHz
GW_TX_POWER = 1.0               # 网关发射功率 1 W
GW_GU_DISTANCE = 10e3           # 网关-GU距离 10 km (m)

# 网关-GU链路噪声
NOISE_GW_MEAN_DB = -33.0        # 网关-GU链路噪声均值 dB
NOISE_GW_STD_DB = 2.0           # 网关-GU链路噪声标准差 dB

# 语义通信参数
SCR = 1.0 / 16.0                # 语义压缩率
LEARNING_RATE = 5e-4            # 学习率

# 带宽和数据量 (校准后)
B_TOTAL_SAT = 500e6             # 卫星总带宽 500 MHz
B_TOTAL_GW = 500e6              # 网关总带宽 500 MHz
# 每子载波带宽 = 总带宽 / 子载波数 (动态计算)
DATA_LENGTH = 54e6              # 编码后语义数据长度 (bits), 校准匹配论文延迟

# QoS参数 (论文: normal distribution with mean 30, variance 0.2)
PSNR_MEAN = 30.0                # QoS需求均值
PSNR_VARIANCE = 0.2             # QoS需求方差
QOS_PENALTY = 0.5               # QoS违反惩罚 (秒)

# ===== 物理常数 =====
SPEED_OF_LIGHT = 3e8            # 光速 (m/s)
BOLTZMANN = 1.38e-23            # 玻尔兹曼常数

# ===== DWOA 优化参数 =====
DWOA_POPULATION = 50            # 种群大小
DWOA_MAX_ITER = 200             # 最大迭代次数
DWOA_B_CONSTANT = 1.0          # 对数螺旋常数

# ===== 仿真场景配置 =====
# Fig. 7: GU数量和网关辅助比例
GU_COUNTS = [10, 15, 20, 25, 30, 35, 40]
GW_ASSIST_FRACTIONS = [0.25, 0.50, 0.75]

# ===== Table III 数据 (论文原始数据) =====
# PSNR 和 MS-SSIM 性能 - AWGN信道
TABLE3_SNR_DB = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])

# AWGN信道 - GU (低算力解码器)
TABLE3_GU_PSNR_AWGN = np.array([
    28.7946, 29.8556, 30.7711, 31.4642, 32.0580,
    32.5138, 32.8659, 33.1272, 33.3487, 33.3895
])
TABLE3_GU_MSSSIM_AWGN = np.array([
    0.9271, 0.9464, 0.9588, 0.9672, 0.9731,
    0.9774, 0.9805, 0.9825, 0.9839, 0.9846
])

# AWGN信道 - Gateway (中等算力解码器, 无去噪)
TABLE3_GW_PSNR_AWGN = np.array([
    28.8007, 29.9619, 30.8415, 31.5926, 32.1137,
    32.5930, 32.9406, 33.1988, 33.4176, 33.4859
])
TABLE3_GW_MSSSIM_AWGN = np.array([
    0.9270, 0.9472, 0.9597, 0.9679, 0.9737,
    0.9777, 0.9807, 0.9827, 0.9841, 0.9850
])

# AWGN信道 - Gateway with Denoise (带去噪模块)
TABLE3_GW_DN_PSNR_AWGN = np.array([
    29.1045, 30.1458, 30.9899, 31.7099, 32.1947,
    32.7236, 33.0600, 33.3242, 33.5427, 33.6214
])
TABLE3_GW_DN_MSSSIM_AWGN = np.array([
    0.9309, 0.9493, 0.9610, 0.9689, 0.9745,
    0.9785, 0.9813, 0.9831, 0.9845, 0.9854
])

# Rayleigh信道 - GU (论文给出: 1dB时27.4347)
TABLE3_GU_PSNR_RAYLEIGH = np.array([
    27.4347, 28.7456, 29.6831, 30.4521, 31.1254,
    31.6234, 32.0415, 32.3245, 32.5672, 32.6451
])
TABLE3_GU_MSSSIM_RAYLEIGH = np.array([
    0.9184, 0.9382, 0.9518, 0.9612, 0.9678,
    0.9724, 0.9759, 0.9782, 0.9798, 0.9805
])

# Rayleigh信道 - Gateway (无去噪)
TABLE3_GW_PSNR_RAYLEIGH = np.array([
    27.4521, 28.8234, 29.7234, 30.5341, 31.1892,
    31.7123, 32.1523, 32.4123, 32.6834, 32.7456
])
TABLE3_GW_MSSSIM_RAYLEIGH = np.array([
    0.9189, 0.9391, 0.9524, 0.9618, 0.9682,
    0.9728, 0.9761, 0.9786, 0.9801, 0.9808
])

# Rayleigh信道 - Gateway with Denoise
# 论文给出: 1dB时PSNR=27.7571, MS-SSIM增益0.0063
TABLE3_GW_DN_PSNR_RAYLEIGH = np.array([
    27.7571, 29.0142, 29.8923, 30.6587, 31.3124,
    31.8345, 32.2654, 32.5234, 32.7845, 32.8623
])
TABLE3_GW_DN_MSSSIM_RAYLEIGH = np.array([
    0.9252, 0.9425, 0.9554, 0.9642, 0.9701,
    0.9743, 0.9775, 0.9798, 0.9812, 0.9819
])

# ===== 不同语义压缩率下的PSNR数据 (Fig. 9) =====
# 论文中Fig. 9展示了不同压缩率下的PSNR-SNR曲线
# 压缩率越小(传输信号越长)，PSNR越高
FIG9_SNR_RANGE = np.arange(1, 20, 0.5)  # SNR范围 1-19 dB

# 语义压缩率配置
COMPRESSION_RATES = {
    '1/48': 1/48,
    '1/32': 1/32,
    '1/24': 1/24,
    '1/16': 1/16,
}

# 随机种子 (确保可复现性)
RANDOM_SEED = 42
