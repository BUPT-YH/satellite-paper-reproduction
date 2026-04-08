"""
信道模型: 路径损耗、SNR、INR 计算
"""

import numpy as np
from config import (CARRIER_FREQ, WAVELENGTH, C_LIGHT, R_EARTH,
                    NOISE_PSD, ATMOS_LOSS, SCINTILLATION_MARGIN)


def free_space_path_loss_db(distance_km, freq_hz=CARRIER_FREQ):
    """
    自由空间路径损耗 (dB)
    FSPL = 20·log10(4π·d·f/c)
    """
    distance_m = distance_km * 1000.0
    fspl = 20 * np.log10(4 * np.pi * distance_m * freq_hz / C_LIGHT)
    return fspl


def total_path_loss_db(distance_km, elevation_deg):
    """
    总路径损耗 (dB): 自由空间 + 大气 + 闪烁

    参数:
        distance_km: 距离 (km)
        elevation_deg: 仰角 (度)
    """
    fspl = free_space_path_loss_db(distance_km)

    # 大气损耗随仰角变化 (低仰角穿越更多大气)
    # 简化模型: atm_loss = ATMOS_LOSS / sin(el)
    sin_el = np.maximum(np.sin(np.deg2rad(elevation_deg)), 0.1)
    atm_loss = ATMOS_LOSS / sin_el

    # 闪烁裕量 (随仰角减小而增大)
    scint = SCINTILLATION_MARGIN / np.sqrt(sin_el)

    return fspl + atm_loss + scint


def compute_snr_db(eirp_density, tx_gain_dbi, rx_gain_dbi, path_loss_db, noise_psd_w_hz=NOISE_PSD):
    """
    计算 SNR (dB)

    SNR = EIRP_density + Gtx + Grx - L - Pn

    注意: EIRP 已经包含了发射天线增益, 所以这里需要用发射功率而非 EIRP
    但论文中 EIRP 是频谱密度, 且 Gtx 是指向地面用户的发射增益

    实际模型: SNR = (Ptx_density + Gtx_toward_user) + Grx - L - N0
    其中 Ptx_density 是每 Hz 发射功率, Gtx 是发射天线在用户方向的增益
    EIRP_density = Ptx_density + Gtx_max

    所以: SNR = EIRP_density + (Gtx_toward_user - Gtx_max) + Grx - L - N0
    简化为: SNR = Ptx_density + Gtx + Grx - L - N0

    参数:
        eirp_density: EIRP 频谱密度 (W/Hz, 线性)
        tx_gain_dbi: 发射天线增益 (dBi)
        rx_gain_dbi: 接收天线增益 (dBi)
        path_loss_db: 路径损耗 (dB)
        noise_psd_w_hz: 噪声功率谱密度 (W/Hz)
    """
    eirp_db = 10 * np.log10(eirp_density)
    noise_db = 10 * np.log10(noise_psd_w_hz)

    snr = eirp_db + tx_gain_dbi + rx_gain_dbi - path_loss_db - noise_db
    return snr


def compute_inr_db(interferer_eirp, tx_gain_dbi, rx_gain_dbi, path_loss_db, noise_psd_w_hz=NOISE_PSD):
    """
    计算 INR (dB) — 干噪比

    INR = I / N = (Ptx·Gtx·Grx·L⁻¹) / Pn
    """
    eirp_db = 10 * np.log10(interferer_eirp)
    noise_db = 10 * np.log10(noise_psd_w_hz)

    inr = eirp_db + tx_gain_dbi + rx_gain_dbi - path_loss_db - noise_db
    return inr


def compute_sinr_db(snr_db, inr_db):
    """SINR = SNR / (1 + INR), 转换为 dB"""
    snr_lin = 10 ** (snr_db / 10)
    inr_lin = 10 ** (inr_db / 10)
    sinr_lin = snr_lin / (1 + inr_lin)
    return 10 * np.log10(np.maximum(sinr_lin, 1e-30))
