"""
仿真参数配置 — 提取自论文 Table/Section V
论文: Semantic Satellite Communications Based on Generative Foundation Model
期刊: IEEE JSAC, Vol.43, No.7, July 2025
"""
import numpy as np

# ==================== 项目标识 ====================
PROJECT = "[8-JSAC-2025]Semantic_Satellite_Communications_Based_on_Generative_Foundation_Model"

# ==================== 图像参数 ====================
IMG_HEIGHT = 256
IMG_WIDTH = 512
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# ==================== 语义编解码器参数 ====================
# 编码器输出维度 (C, H, W)
ENCODER_OUTPUT_DIM = (256, 16, 8)
BINARY_LENGTH = 256 * 16 * 8  # = 32768 bits = 32 Kbits

# 语义方法传输比特数
SEMANTIC_BITS = 32  # Kbits

# ==================== 传统方法参数 ====================
JPEG_COMPRESSED_SIZE = 144  # Kbits (JPEG压缩后)
LDPC_CODE_RATE_127 = 64 / 127   # 约 0.504
LDPC_CODE_RATE_255 = 64 / 255   # 约 0.251
QAM_ORDER = 4  # 4-QAM 调制

# ==================== 信道仿真参数 ====================
SNR_RANGE = np.arange(-10, 11, 2)  # -10 ~ 10 dB, 步长 2
SNR_FINE = np.linspace(-10, 10, 100)  # 细粒度 SNR 用于平滑曲线

CCI_RATIOS = [0, 0.5]  # 同频干扰比例: 无干扰 / 50%干扰

# 卫星信道
SATELLITE_ALTITUDE_KM = 550  # LEO 轨道高度
NUM_MULTIPATH = 4  # 多径分量数 L

# ==================== 错误检测器参数 ====================
ROUGH_DETECTOR_MSE_THRESHOLD = 0.015  # 粗检测器 MSE 阈值
FINE_DETECTOR_MSE_THRESHOLD = 0.01    # 精细检测器 MSE 阈值
PARITY_BITS_ROUGH = 32  # 粗检测器奇偶校验码位数
PARITY_BITS_FINE = 32   # 精细检测器奇偶校验码位数

# ==================== 自适应编解码器参数 ====================
# 场景设置 (论文 Section III-B)
ADAPTIVE_SCENARIOS = {
    'good_channel': {'snr': 10, 'cci': 0, 'target': 'full_image'},
    'medium_channel': {'snr': -10, 'cci': 0, 'target': 'important_parts'},
    'bad_channel': {'snr': 0, 'cci': 0.5, 'target': 'important_parts_halved'},
}

# ==================== 训练参数 ====================
LEARNING_RATE_SEG = 0.0005  # UNet 分割学习率
BATCH_SIZE = 16
NUM_EPOCHS = 100

# ==================== 仿真参数 ====================
NUM_MONTE_CARLO = 10000   # Fig.12 用到的图像数量
RANDOM_SEED = 42

# ==================== UNet 分割网络结构 ====================
UNET_ENCODER_CHANNELS = [64, 128, 256, 512]
UNET_DECODER_CHANNELS = [1024, 512, 256, 128]
UNET_FINAL_CHANNELS = 64

# ==================== 语义编解码器结构 ====================
ENCODER_KERNEL_SIZES = [9, 7, 5, 3]
ENCODER_CHANNELS = [64, 128, 256, 256]
ENCODER_STRIDES = [4, 2, 2, 2]

# ==================== 颜色方案 (IEEE风格) ====================
METHOD_COLORS = {
    'JPEG+LDPC(64,127)': '#0072BD',
    'JPEG+LDPC(64,255)': '#D95319',
    'JSCC': '#EDB120',
    'FMSAT(SegGPT)': '#7E2F8E',
    'FMSAT(UNet)': '#77AC30',
    'AFMSAT': '#4DBEEE',
    'AFMSAT(Correl)': '#A2142F',
    'JSCC(Adapt)': '#EDB120',
    'FMSAT': '#7E2F8E',
}

METHOD_MARKERS = {
    'JPEG+LDPC(64,127)': 'o',
    'JPEG+LDPC(64,255)': 's',
    'JSCC': '^',
    'FMSAT(SegGPT)': 'D',
    'FMSAT(UNet)': 'v',
    'AFMSAT': 'p',
    'AFMSAT(Correl)': 'h',
    'JSCC(Adapt)': '<',
    'FMSAT': 'D',
}

METHOD_LINESTYLES = {
    'JPEG+LDPC(64,127)': '-',
    'JPEG+LDPC(64,255)': '--',
    'JSCC': '-.',
    'FMSAT(SegGPT)': '-',
    'FMSAT(UNet)': '--',
    'AFMSAT': '-',
    'AFMSAT(Correl)': '--',
    'JSCC(Adapt)': ':',
    'FMSAT': '-',
}
