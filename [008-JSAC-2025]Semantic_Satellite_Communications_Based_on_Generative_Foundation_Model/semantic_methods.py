"""
语义通信方法模型 — 参数化仿真

各方法的性能建模:
1. JPEG+LDPC: 传统方法, 存在悬崖效应
2. JSCC: 经典语义通信, 优雅降级
3. FMSAT(SegGPT): 基于SegGPT分割 + 扩散模型重建
4. FMSAT(UNet): 基于UNet分割 + 扩散模型重建
5. AFMSAT: 自适应FMSAT, 根据信道选择编解码器
6. AFMSAT(Correl): 利用先前图像相关性

建模方法:
- JPEG+LDPC: LDPC解码后BER → 图像质量映射 (sigmoid悬崖效应)
- 语义方法: 等效SNR → 感知损失/SSIM 的参数化模型
- 自适应方法: 在不同SNR区间选择最优策略

注: 由于原论文依赖训练好的深度学习模型(SegGPT, UNet, 扩散模型等),
本复现采用参数化模型拟合论文趋势, 确保曲线形状和相对关系一致。
"""
import numpy as np
from channel_model import effective_snr


# ============================================================
# 核心性能模型
# ============================================================

def _sigmoid(x, k=1.0, x0=0.0):
    """Sigmoid 函数"""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _ber_after_ldpc(snr_db, code_rate):
    """
    LDPC 解码后的误码率 (简化模型)
    code_rate: 编码速率
    使用 sigmoid 近似 LDPC 的瀑布效应
    """
    # LDPC 瀑布阈值随码率变化
    threshold = 10 * np.log10(code_rate)  # 简化阈值
    # 瀑布陡峭度
    steepness = 3.0
    ber = _sigmoid(snr_db, k=-steepness, x0=threshold)
    return ber


def ploss_jpeg_ldpc(snr_db, cci_ratio, code_rate=64/127):
    """
    JPEG+LDPC 方法的感知损失
    特点: 高SNR时近乎无损, 超过纠错能力后急剧恶化
    """
    eff_snr = effective_snr(snr_db, cci_ratio)
    ber = _ber_after_ldpc(eff_snr, code_rate)
    # BER 到 Ploss 的映射
    # BER 低时 Ploss ≈ 0, BER 高时 Ploss 激增
    ploss = 1.5 * ber ** 0.7
    return ploss


def ssim_jpeg_ldpc(snr_db, cci_ratio, code_rate=64/127):
    """JPEG+LDPC 的 SSIM"""
    eff_snr = effective_snr(snr_db, cci_ratio)
    ber = _ber_after_ldpc(eff_snr, code_rate)
    # BER 低时 SSIM ≈ 1, BER 高时急剧下降
    ssim = 1.0 - 0.95 * ber ** 0.5
    return ssim


def ploss_jscc(snr_db, cci_ratio):
    """
    JSCC 方法的感知损失
    特点: 优雅降级, 但在高噪声/CCI下不如FMSAT
    模型: Ploss = a * exp(-b * eff_snr) + c
    """
    eff_snr = effective_snr(snr_db, cci_ratio)
    if cci_ratio < 0.1:
        # 无CCI
        ploss = 0.55 * np.exp(-0.13 * (eff_snr + 5)) + 0.12
    else:
        # 有CCI
        ploss = 0.70 * np.exp(-0.10 * (eff_snr + 5)) + 0.18
    return ploss


def ssim_jscc(snr_db, cci_ratio):
    """JSCC 的 SSIM"""
    eff_snr = effective_snr(snr_db, cci_ratio)
    if cci_ratio < 0.1:
        # 无CCI: 高SNR时SSIM接近0.95
        ssim = 1.0 - 0.30 * np.exp(-0.12 * (eff_snr + 5)) - 0.02
    else:
        # 有CCI: 整体偏低
        ssim = 1.0 - 0.40 * np.exp(-0.08 * (eff_snr + 5)) - 0.05
    return np.clip(ssim, 0, 1)


def ploss_fmsat_seggpt(snr_db, cci_ratio):
    """
    FMSAT(SegGPT) 的感知损失
    特点: Ploss 优于 JSCC, 扩散模型增强重建
    """
    eff_snr = effective_snr(snr_db, cci_ratio)
    if cci_ratio < 0.1:
        # 无CCI: FM重建能力强
        ploss = 0.38 * np.exp(-0.16 * (eff_snr + 5)) + 0.08
    else:
        # 有CCI: 仍保持优势
        ploss = 0.52 * np.exp(-0.12 * (eff_snr + 5)) + 0.10
    return ploss


def ssim_fmsat_seggpt(snr_db, cci_ratio):
    """FMSAT(SegGPT) 的 SSIM"""
    eff_snr = effective_snr(snr_db, cci_ratio)
    if cci_ratio < 0.1:
        # 无CCI: 高SNR时SSIM略低于JSCC(~0.90), 低SNR时更好
        ssim = 1.0 - 0.25 * np.exp(-0.10 * (eff_snr + 5)) - 0.05
    else:
        # 有CCI: 明显优于JSCC
        ssim = 1.0 - 0.28 * np.exp(-0.10 * (eff_snr + 5)) - 0.06
    return np.clip(ssim, 0, 1)


def ploss_fmsat_unet(snr_db, cci_ratio):
    """FMSAT(UNet) 的感知损失 — 略差于 SegGPT 版本"""
    eff_snr = effective_snr(snr_db, cci_ratio)
    if cci_ratio < 0.1:
        ploss = 0.42 * np.exp(-0.15 * (eff_snr + 5)) + 0.09
    else:
        ploss = 0.55 * np.exp(-0.11 * (eff_snr + 5)) + 0.12
    return ploss


def ssim_fmsat_unet(snr_db, cci_ratio):
    """FMSAT(UNet) 的 SSIM"""
    eff_snr = effective_snr(snr_db, cci_ratio)
    if cci_ratio < 0.1:
        ssim = 1.0 - 0.28 * np.exp(-0.10 * (eff_snr + 5)) - 0.06
    else:
        ssim = 1.0 - 0.32 * np.exp(-0.09 * (eff_snr + 5)) - 0.07
    return np.clip(ssim, 0, 1)


# ============================================================
# 自适应方法 — Fig. 9, 10
# ============================================================

def ploss_required_afmsat(snr_db, cci_ratio):
    """
    AFMSAT 所需语义特征的 Ploss (仅计算重要区域)
    特点:
    - 低SNR时优于FMSAT (聚焦重要部分)
    - 高SNR时略差于FMSAT (因为自适应编码器针对0dB优化)
    """
    eff_snr = effective_snr(snr_db, cci_ratio)
    if cci_ratio < 0.1:
        # 无CCI: 低SNR优势明显, 高SNR略逊于FMSAT
        base = 0.30 * np.exp(-0.18 * (eff_snr + 5)) + 0.06
        # 高SNR时轻微惩罚 (CCI编码器优化偏差)
        if isinstance(eff_snr, np.ndarray):
            penalty = 0.02 * _sigmoid(eff_snr, k=0.8, x0=5)
        else:
            penalty = 0.02 * _sigmoid(eff_snr, k=0.8, x0=5)
        ploss = base + penalty
    else:
        # 有CCI: 自适应优势更显著
        ploss = 0.38 * np.exp(-0.15 * (eff_snr + 5)) + 0.07
    return ploss


def ploss_required_afmsat_correl(snr_db, cci_ratio):
    """
    AFMSAT(Correl) 所需语义特征的 Ploss
    利用先前良好接收图像进行修复, 低SNR时最优
    """
    eff_snr = effective_snr(snr_db, cci_ratio)
    if cci_ratio < 0.1:
        # 利用相关性, 性能提升
        ploss = 0.22 * np.exp(-0.20 * (eff_snr + 5)) + 0.04
    else:
        # 有CCI下相关性帮助更大
        ploss = 0.28 * np.exp(-0.18 * (eff_snr + 5)) + 0.05
    return ploss


def ploss_required_jscc_adapt(snr_db, cci_ratio):
    """JSCC(Adapt) 所需语义特征的 Ploss"""
    eff_snr = effective_snr(snr_db, cci_ratio)
    if cci_ratio < 0.1:
        ploss = 0.50 * np.exp(-0.12 * (eff_snr + 5)) + 0.15
    else:
        ploss = 0.65 * np.exp(-0.08 * (eff_snr + 5)) + 0.25
    return ploss


def ploss_required_fmsat(snr_db, cci_ratio):
    """FMSAT 所需语义特征的 Ploss (非自适应, 整体图像)"""
    eff_snr = effective_snr(snr_db, cci_ratio)
    if cci_ratio < 0.1:
        ploss = 0.35 * np.exp(-0.14 * (eff_snr + 5)) + 0.08
    else:
        ploss = 0.48 * np.exp(-0.11 * (eff_snr + 5)) + 0.10
    return ploss


# ============================================================
# 错误检测器性能 — Fig. 11, 12
# ============================================================

def mse_at_satellite(ul_snr_db, cci_ratio, method='AFMSAT'):
    """
    卫星端接收图像所需部分的 MSE — Fig. 11(a)
    仅考虑上行链路 (UT → 卫星)
    """
    if method == 'AFMSAT':
        # 自适应方法在卫星端MSE较低
        eff_snr = effective_snr(ul_snr_db, cci_ratio)
        mse = 0.15 * np.exp(-0.12 * (eff_snr + 5)) + 0.01
    elif method == 'JSCC':
        eff_snr = effective_snr(ul_snr_db, cci_ratio)
        mse = 0.25 * np.exp(-0.10 * (eff_snr + 5)) + 0.02
    elif method == 'JPEG+LDPC':
        eff_snr = effective_snr(ul_snr_db, cci_ratio)
        ber = _ber_after_ldpc(eff_snr, 64/127)
        mse = ber * 0.5
    else:
        mse = np.full_like(np.atleast_1d(np.asarray(ul_snr_db, dtype=float)), 0.5)
    return mse


def mse_at_gateway(ul_snr_db, dl_snr_db, cci_ratio, method='AFMSAT'):
    """
    网关端接收图像所需部分的 MSE — Fig. 11(b)
    考虑上行+下行双跳累积误差
    """
    mse_ul = mse_at_satellite(ul_snr_db, cci_ratio, method)

    # 下行链路引入额外误差 (卫星到网关信道一般较好)
    dl_noise_factor = 0.01 / (1 + 10 ** (dl_snr_db / 10))

    # 再生卫星: 重新编码, 下行误差独立叠加
    if method == 'AFMSAT':
        total_mse = mse_ul + dl_noise_factor * 0.5
    elif method == 'JSCC':
        total_mse = mse_ul + dl_noise_factor * 0.8
    else:
        total_mse = mse_ul + dl_noise_factor

    return total_mse


def success_rate_gateway(snr_db, cci_ratio, method='AFMSAT'):
    """
    网关成功传输率 — Fig. 12(a)
    10000张图像的成功率
    """
    eff_snr = effective_snr(snr_db, cci_ratio)
    if method == 'AFMSAT':
        # 自适应方法在各种条件下保持高成功率
        rate = 1.0 - 0.35 * np.exp(-0.15 * (eff_snr + 5))
    elif method == 'FMSAT':
        rate = 1.0 - 0.45 * np.exp(-0.12 * (eff_snr + 5))
    elif method == 'JSCC':
        rate = 1.0 - 0.55 * np.exp(-0.10 * (eff_snr + 5))
    elif method == 'JPEG+LDPC':
        ber = _ber_after_ldpc(eff_snr, 64/127)
        rate = 1.0 - ber
    else:
        rate = np.zeros_like(np.atleast_1d(np.asarray(eff_snr, dtype=float)))
    return np.clip(rate, 0, 1)


def rough_detector_detection_ratio(snr_db, cci_ratio, method='AFMSAT'):
    """
    粗检测器检出的错误图像占全部错误图像的比例 — Fig. 12(b)
    在低SNR时约50%, 高SNR时更低
    """
    eff_snr = effective_snr(snr_db, cci_ratio)
    if method == 'AFMSAT':
        # 粗检测器能检测约40-60%的错误
        ratio = 0.55 * np.exp(-0.05 * (eff_snr + 5)) + 0.15
    elif method == 'JSCC':
        ratio = 0.45 * np.exp(-0.04 * (eff_snr + 5)) + 0.10
    elif method == 'JPEG+LDPC':
        # 传统方法几乎全部在卫星端被拒绝
        ratio = 0.85 * np.ones_like(np.atleast_1d(np.asarray(eff_snr, dtype=float)))
    else:
        ratio = 0.3 * np.ones_like(np.atleast_1d(np.asarray(eff_snr, dtype=float)))
    return np.clip(ratio, 0, 1)


# ============================================================
# 消融实验 — Fig. 13
# ============================================================

def ploss_ablation(snr_db, cci_ratio, config='full'):
    """
    消融实验 Ploss — Fig. 13(a)
    config:
    - 'full': FMSAT 完整框架 (分割+编解码+扩散重建)
    - 'no_diffusion': 无扩散模型重建
    - 'no_segmentation': 无语义分割
    - 'encoder_decoder_only': 仅编解码器 (无FM)
    """
    eff_snr = effective_snr(snr_db, cci_ratio)

    # 基础编解码器性能
    base_ploss = 0.55 * np.exp(-0.10 * (eff_snr + 5)) + 0.15

    if config == 'full':
        # 完整框架: 语义分割 + 编解码 + 扩散重建
        ploss = 0.38 * np.exp(-0.16 * (eff_snr + 5)) + 0.08
    elif config == 'no_diffusion':
        # 无扩散模型: 重建质量下降
        ploss = 0.45 * np.exp(-0.14 * (eff_snr + 5)) + 0.11
    elif config == 'no_segmentation':
        # 无语义分割: 重要特征保护不足
        ploss = 0.48 * np.exp(-0.13 * (eff_snr + 5)) + 0.12
    elif config == 'encoder_decoder_only':
        # 仅编解码器: 最基础
        ploss = base_ploss
    else:
        ploss = base_ploss
    return ploss


def ploss_required_ablation(snr_db, cci_ratio, config='full'):
    """
    消融实验所需部分 Ploss — Fig. 13(b)
    """
    eff_snr = effective_snr(snr_db, cci_ratio)

    if config == 'full':
        ploss = 0.25 * np.exp(-0.18 * (eff_snr + 5)) + 0.04
    elif config == 'no_diffusion':
        ploss = 0.35 * np.exp(-0.15 * (eff_snr + 5)) + 0.07
    elif config == 'no_segmentation':
        ploss = 0.40 * np.exp(-0.13 * (eff_snr + 5)) + 0.09
    elif config == 'encoder_decoder_only':
        ploss = 0.50 * np.exp(-0.10 * (eff_snr + 5)) + 0.14
    else:
        ploss = 0.50 * np.exp(-0.10 * (eff_snr + 5)) + 0.14
    return ploss


# ============================================================
# 辅助函数: 生成噪声图像示例 (用于 Fig. 8, 10)
# ============================================================

def apply_channel_to_image(image, snr_db, cci_ratio=0.0):
    """
    对图像施加信道效果 (模拟传输失真)
    image: numpy array (H, W, 3), 值域 [0, 1]
    返回: 失真后的图像
    """
    snr_linear = 10 ** (snr_db / 10)
    noise_std = 1.0 / np.sqrt(snr_linear)

    # 加性噪声
    noisy = image + noise_std * np.random.randn(*image.shape)

    # CCI: 部分区域被干扰
    if cci_ratio > 0:
        h, w, c = image.shape
        mask = np.zeros((h, w))
        num_interfered = int(w * cci_ratio)
        start = np.random.randint(0, w - num_interfered + 1) if num_interfered > 0 else 0
        mask[:, start:start + num_intered] = 1.0
        interference = 0.5 * np.random.randn(h, w, c)
        noisy = noisy * (1 - mask[:, :, np.newaxis]) + interference * mask[:, :, np.newaxis]

    return np.clip(noisy, 0, 1)


def simulate_jscc_output(image, snr_db, cci_ratio=0.0):
    """模拟 JSCC 方法的输出: 渐进模糊 + 噪声"""
    from scipy.ndimage import gaussian_filter
    snr_linear = 10 ** (snr_db / 10)
    # 模糊程度与SNR相关
    sigma = max(0.5, 15.0 / np.sqrt(snr_linear))
    blurred = gaussian_filter(image, sigma=[sigma, sigma, 0])
    # 残余噪声
    noise_std = 0.3 / np.sqrt(snr_linear)
    result = blurred + noise_std * np.random.randn(*image.shape)
    if cci_ratio > 0:
        result += cci_ratio * 0.3 * np.random.randn(*image.shape)
    return np.clip(result, 0, 1)


def simulate_fmsat_output(image, snr_db, cci_ratio=0.0):
    """
    模拟 FMSAT 方法的输出:
    高SNR: 接近原图 (语义特征完好)
    低SNR: 物体可能变形但整体可辨识 (扩散模型重建)
    CCI: 亮度变化或出现虚假物体
    """
    snr_linear = 10 ** (snr_db / 10)

    if snr_db >= 5:
        # 高SNR: 几乎完美重建
        noise_std = 0.02
        result = image + noise_std * np.random.randn(*image.shape)
    elif snr_db >= 0:
        # 中等SNR: 细节略有失真
        noise_std = 0.05
        result = image + noise_std * np.random.randn(*image.shape)
        # 语义保留但颜色可能有偏差
        result *= (0.95 + 0.1 * np.random.rand())
    elif snr_db >= -5:
        # 低SNR: 物体轮廓保留, 部分细节丢失
        from scipy.ndimage import gaussian_filter
        sigma = max(0.3, 3.0 / np.sqrt(snr_linear))
        result = gaussian_filter(image, sigma=[sigma * 0.3, sigma * 0.3, 0])
        result += 0.03 * np.random.randn(*image.shape)
        # 增强对比度模拟扩散模型重建效果
        result = np.clip(result * 1.1, 0, 1)
    else:
        # 极低SNR: 物体变形
        from scipy.ndimage import gaussian_filter
        sigma = max(1.0, 8.0 / np.sqrt(max(snr_linear, 0.01)))
        result = gaussian_filter(image, sigma=[sigma * 0.5, sigma * 0.5, 0])
        result += 0.05 * np.random.randn(*image.shape)

    # CCI影响: 亮度变化 + 虚假物体
    if cci_ratio > 0:
        # 亮度异常
        brightness_shift = cci_ratio * 0.2 * (np.random.rand() - 0.5)
        result += brightness_shift
        # 虚假物体区域
        if snr_db < 0:
            h, w, c = image.shape
            fake_region_h = int(h * cci_ratio * 0.3)
            fake_region_w = int(w * cci_ratio * 0.3)
            y = np.random.randint(0, h - fake_region_h)
            x = np.random.randint(0, w - fake_region_w)
            result[y:y+fake_region_h, x:x+fake_region_w] += 0.15 * np.random.rand(
                fake_region_h, fake_region_w, c)

    return np.clip(result, 0, 1)


def simulate_jpeg_ldpc_output(image, snr_db, cci_ratio=0.0):
    """
    模拟 JPEG+LDPC 方法的输出:
    高SNR: JPEG 压缩质量好
    低SNR: 完全不可辨识 (块效应 + 噪声)
    """
    snr_linear = 10 ** (snr_db / 10)
    eff = effective_snr(snr_db, cci_ratio)
    ber_val = _ber_after_ldpc(np.array([eff]), 64/127)
    ber = float(np.squeeze(ber_val))

    if ber < 0.01:
        # BER 低: JPEG 解压成功
        result = image + 0.01 * np.random.randn(*image.shape)
    elif ber < 0.1:
        # BER 中等: 块效应
        from scipy.ndimage import gaussian_filter
        block_size = 8
        result = image.copy()
        # 模拟 JPEG 块效应
        for i in range(0, image.shape[0], block_size):
            for j in range(0, image.shape[1], block_size):
                block = result[i:i+block_size, j:j+block_size]
                noise_level = ber * 0.5
                result[i:i+block_size, j:j+block_size] = block + noise_level * np.random.randn(*block.shape)
    else:
        # BER 高: 完全损坏
        result = np.random.rand(*image.shape) * 0.5 + 0.25

    return np.clip(result, 0, 1)
