"""
卫星信道模型 — 基于论文 Eq.(1)-(3)
模拟 LEO 卫星信道: 路径损耗、多径衰落、多普勒频移、同频干扰
"""
import numpy as np
from config import SATELLITE_ALTITUDE_KM, NUM_MULTIPATH


def path_loss_db(distance_km, frequency_ghz=12.0):
    """
    自由空间路径损耗 (dB)
    distance_km: 传播距离 (km)
    frequency_ghz: 载波频率 (GHz), Ku波段默认 12 GHz
    """
    wavelength_m = 3e8 / (frequency_ghz * 1e9)
    d_m = distance_km * 1e3
    return 20 * np.log10(4 * np.pi * d_m / wavelength_m)


def rain_attenuation_db(rain_rate_mmh=25.0, path_length_km=5.0, frequency_ghz=12.0):
    """
    简化的雨衰模型 (dB)
    rain_rate_mmh: 降雨率 (mm/h)
    path_length_km: 雨区路径长度 (km)
    """
    # ITU-R P.838 简化参数 (Ku波段)
    k = 0.015
    alpha = 1.12
    specific_atten = k * rain_rate_mmh ** alpha  # dB/km
    return specific_atten * path_length_km


def leo_channel_impulse_response(num_paths=NUM_MULTIPATH, max_delay_spread_us=0.5):
    """
    LEO 卫星信道冲激响应 — Eq.(2)
    h(t, τ) = Σ αl * exp(j2πvl*t) * δ(τ - τl)
    返回: 信道增益 αl, 多普勒频移 vl, 时延 τl
    """
    # 路径增益: 主径 (LoS) + 反射径
    alpha = np.zeros(num_paths, dtype=complex)
    alpha[0] = 1.0  # LoS 主径
    for l in range(1, num_paths):
        alpha[l] = 0.1 * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

    # 多普勒频移: LEO 卫星高速运动
    # 最大多普勒 ≈ v * f / c, v≈7.5km/s, f≈12GHz → ~300 kHz
    max_doppler_hz = 300e3
    v = max_doppler_hz * (2 * np.random.rand(num_paths) - 1)

    # 多径时延
    tau = np.sort(np.random.uniform(0, max_delay_spread_us * 1e-6, num_paths))

    return alpha, v, tau


def simulate_ofdm_channel(symbols, snr_db, cci_ratio=0.0, num_subcarriers=256):
    """
    模拟 OFDM 传输信道
    symbols: 发送符号 (频域)
    snr_db: 信噪比 (dB)
    cci_ratio: 同频干扰比例 (0~1)
    num_subcarriers: 子载波数

    返回: 接收符号
    """
    snr_linear = 10 ** (snr_db / 10)

    # 信号功率归一化
    signal_power = np.mean(np.abs(symbols) ** 2)

    # 加性高斯白噪声
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*symbols.shape)
                                          + 1j * np.random.randn(*symbols.shape))

    # 同频干扰 (CCI): 在部分子载波上叠加干扰信号
    interference = np.zeros_like(symbols)
    if cci_ratio > 0:
        num_interfered = int(num_subcarriers * cci_ratio)
        interfered_indices = np.random.choice(num_subcarriers, num_interfered, replace=False)
        # 干扰功率与信号功率相当
        int_power = signal_power * 0.8  # 干扰信号略弱于有用信号
        interference_signal = np.sqrt(int_power / 2) * (np.random.randn(*symbols.shape)
                                                         + 1j * np.random.randn(*symbols.shape))
        mask = np.zeros(num_subcarriers, dtype=bool)
        mask[interfered_indices] = True
        interference[mask] = interference_signal[mask]

    # 接收信号: y = h*x + interference + noise
    # 简化: h=1 (假设完美信道估计和均衡)
    received = symbols + interference + noise

    return received


def effective_snr(snr_db, cci_ratio):
    """
    计算有效信噪比 (考虑CCI)
    当存在CCI时，有效SNR降低
    """
    snr_linear = 10 ** (snr_db / 10)
    if cci_ratio > 0:
        # CCI 降低有效SNR
        sir_linear = (1 - cci_ratio) / cci_ratio  # 信号干扰比
        effective = snr_linear * sir_linear / (snr_linear + sir_linear)
        return 10 * np.log10(np.maximum(effective, 1e-10))
    return snr_db


def simulate_end_to_end_snr(ul_snr_db, dl_snr_db, cci_ratio=0.0):
    """
    再生卫星端到端等效 SNR
    UT → 卫星 (上行) → 卫星 → 网关 (下行)
    再生卫星可以重新编码, 但传输误差会累积
    """
    ul_linear = 10 ** (ul_snr_db / 10)
    dl_linear = 10 ** (dl_snr_db / 10)

    # CCI 主要影响上行
    if cci_ratio > 0:
        ul_linear *= (1 - cci_ratio)

    # 等效 SNR: 两个链路串联
    total_mse = 1.0 / ul_linear + 1.0 / dl_linear
    total_snr_linear = 1.0 / total_mse

    return 10 * np.log10(np.maximum(total_snr_linear, 1e-10))
