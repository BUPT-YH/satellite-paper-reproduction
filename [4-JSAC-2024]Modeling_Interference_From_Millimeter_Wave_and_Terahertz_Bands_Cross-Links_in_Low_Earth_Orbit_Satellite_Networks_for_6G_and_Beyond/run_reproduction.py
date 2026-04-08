"""
一键复现脚本
论文: Modeling Interference From mmWave and THz Bands Cross-Links in LEO Satellite Networks
期刊: IEEE JSAC, 2024

运行: python run_reproduction.py
"""

import sys
import os
import numpy as np
import time

# 确保在正确目录运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import *
from interference_model import (
    single_orbit_SIR, single_orbit_SINR, single_orbit_capacity,
    shifted_orbit_SINR, full_constellation_capacity,
    full_constellation_SINR, SNR_only
)
from plotting import (
    plot_figure5a, plot_figure5b, plot_figure5c,
    plot_figure9b, plot_figure10
)


def run_figure5():
    """
    Figure 5: 单轨道场景 (SIR, SINR, 容量 vs N)
    """
    print("\n" + "="*60)
    print("  Figure 5: 单轨道干扰分析")
    print("="*60)

    h = 500e3
    N_range = np.arange(5, 201)
    SIR_limit_dB = 1.9

    # 配置要计算的曲线
    configs = [
        ('mmWave_5',  mmWave, 5),
        ('mmWave_10', mmWave, 10),
        ('mmWave_30', mmWave, 30),
        ('subTHz_1',  subTHz, 1),
        ('subTHz_3',  subTHz, 3),
        ('subTHz_5',  subTHz, 5),
    ]

    sir_data = {}
    sinr_data = {}
    capacity_data = {}

    for key, band, alpha in configs:
        print(f"  计算 {key} (alpha={alpha}°)...")
        sirs = []
        sinrs = []
        caps = []
        for N in N_range:
            sirs.append(single_orbit_SIR(N, h, alpha))
            sinrs.append(single_orbit_SINR(N, h, band, alpha))
            caps.append(single_orbit_capacity(N, h, band, alpha))
        sir_data[key] = np.array(sirs)
        sinr_data[key] = np.array(sinrs)
        capacity_data[key] = np.array(caps)

    # 容量极限 (SIR limit -> capacity)
    # mmWave极限: B * log2(1 + 1.55) = 2GHz * log2(2.55) ≈ 2.7 Gbps (论文说约600 Mbps)
    # sub-THz极限: 50GHz * log2(1 + 1.55) ≈ 67.5 Gbps (论文说约15 Gbps)
    cap_limit_mmWave = mmWave['bandwidth'] * np.log2(1 + 10**(SIR_limit_dB/10))
    cap_limit_subTHz = subTHz['bandwidth'] * np.log2(1 + 10**(SIR_limit_dB/10))

    print(f"\n  理论极限验证:")
    print(f"    SIR limit = {SIR_limit_dB} dB (1.55 linear)")
    print(f"    mmWave capacity limit = {cap_limit_mmWave/1e9:.2f} Gbps")
    print(f"    sub-THz capacity limit = {cap_limit_subTHz/1e9:.2f} Gbps")

    # 绘图
    plot_figure5a(N_range, sir_data, SIR_limit_dB)
    plot_figure5b(N_range, sinr_data, SIR_limit_dB)
    plot_figure5c(N_range, capacity_data, cap_limit_mmWave, cap_limit_subTHz)

    # 打印关键数据点
    print(f"\n  关键数据点 (N=72, h=500km, mmWave alpha=5°):")
    N_test = 72
    sir_72 = single_orbit_SIR(N_test, h, 5)
    sinr_72 = single_orbit_SINR(N_test, h, mmWave, 5)
    cap_72 = single_orbit_capacity(N_test, h, mmWave, 5)
    print(f"    SIR = {10*np.log10(sir_72):.1f} dB")
    print(f"    SINR = {10*np.log10(max(sinr_72, 1e-10)):.1f} dB")
    print(f"    Capacity = {cap_72/1e9:.3f} Gbps")

    print("  Figure 5 完成!")


def run_figure9b():
    """
    Figure 9(b): 偏移轨道 SINR vs 波束宽度
    """
    print("\n" + "="*60)
    print("  Figure 9(b): 偏移轨道 SINR vs 波束宽度")
    print("="*60)

    h = 500e3
    alpha_range = np.arange(1, 41)
    Delta_Omega = 90  # RAAN偏移 90°
    Delta_beta = 0    # 同高度, 常数偏移

    sinr_data_mmWave = {}
    sinr_data_subTHz = {}

    for N in [50, 100]:
        print(f"  计算 N={N}...")
        sinr_mm = []
        sinr_thz = []
        for alpha in alpha_range:
            s_mm = shifted_orbit_SINR(N, N, h, 50, Delta_Omega,
                                       alpha, Delta_beta, mmWave)
            s_thz = shifted_orbit_SINR(N, N, h, 50, Delta_Omega,
                                        alpha, Delta_beta, subTHz)
            sinr_mm.append(s_mm)
            sinr_thz.append(s_thz)
        sinr_data_mmWave[f'N={N}'] = np.array(sinr_mm)
        sinr_data_subTHz[f'N={N}'] = np.array(sinr_thz)

    # SNR参考线 (N=50)
    print(f"  计算 SNR参考线 (N=50)...")
    snr_ref = []
    for alpha in alpha_range:
        snr_ref.append(SNR_only(50, h, mmWave, alpha))
    snr_ref = np.array(snr_ref)

    plot_figure9b(alpha_range, sinr_data_mmWave, sinr_data_subTHz, snr_ref)

    print("  Figure 9(b) 完成!")


def run_figure10():
    """
    Figure 10: 完整双星座部署
    10个轨道面, 倾角50°, 两个星座高度500km和510km
    sub-THz alpha=1°, mmWave alpha=5°
    """
    print("\n" + "="*60)
    print("  Figure 10: 完整双星座部署")
    print("="*60)

    h = 500e3
    h_S = 510e3
    n_orbits = 10
    gamma = 50  # 倾角
    N_range = np.arange(10, 501, 2)  # 步长2加速

    capacity_data = {}

    # sub-THz alpha=1°
    print("  计算 sub-THz (alpha=1°) - 无干扰SNR...")
    cap_snr_thz = []
    for N in N_range:
        snr = SNR_only(N, h, subTHz, 1)
        cap_snr_thz.append(subTHz['bandwidth'] * np.log2(1 + snr))
    capacity_data['sub-THz (No interference)'] = np.array(cap_snr_thz)

    print("  计算 sub-THz (alpha=1°) - 仅同轨道干扰...")
    cap_single_thz = []
    for N in N_range:
        cap_single_thz.append(single_orbit_capacity(N, h, subTHz, 1))
    capacity_data['sub-THz (Single orbit only)'] = np.array(cap_single_thz)

    print("  计算 sub-THz (alpha=1°) - 完整干扰...")
    cap_full_thz = []
    for idx, N in enumerate(N_range):
        cap_full_thz.append(full_constellation_capacity(
            int(N), n_orbits, h, h_S, gamma, 1, subTHz))
        if (idx + 1) % 50 == 0:
            print(f"    sub-THz full: {idx+1}/{len(N_range)}")
    capacity_data['sub-THz (Full interference)'] = np.array(cap_full_thz)

    # mmWave alpha=5°
    print("  计算 mmWave (alpha=5°) - 无干扰SNR...")
    cap_snr_mm = []
    for N in N_range:
        snr = SNR_only(N, h, mmWave, 5)
        cap_snr_mm.append(mmWave['bandwidth'] * np.log2(1 + snr))
    capacity_data['mmWave (No interference)'] = np.array(cap_snr_mm)

    print("  计算 mmWave (alpha=5°) - 仅同轨道干扰...")
    cap_single_mm = []
    for N in N_range:
        cap_single_mm.append(single_orbit_capacity(N, h, mmWave, 5))
    capacity_data['mmWave (Single orbit only)'] = np.array(cap_single_mm)

    print("  计算 mmWave (alpha=5°) - 完整干扰...")
    cap_full_mm = []
    for idx, N in enumerate(N_range):
        cap_full_mm.append(full_constellation_capacity(
            int(N), n_orbits, h, h_S, gamma, 5, mmWave))
        if (idx + 1) % 50 == 0:
            print(f"    mmWave full: {idx+1}/{len(N_range)}")
    capacity_data['mmWave (Full interference)'] = np.array(cap_full_mm)

    plot_figure10(N_range, capacity_data)

    # 打印关键数据
    print("\n  关键数据点:")
    for N_test in [50, 100, 200, 350, 500]:
        if N_test in N_range:
            idx = np.where(N_range == N_test)[0][0]
            print(f"    N={N_test}: sub-THz full={cap_full_thz[idx]/1e9:.2f} Gbps, "
                  f"mmWave full={cap_full_mm[idx]/1e9:.2f} Gbps")

    print("  Figure 10 完成!")


if __name__ == '__main__':
    print("="*60)
    print("  论文复现: LEO卫星mmWave/THz星间链路干扰建模")
    print("  IEEE JSAC, Vol. 42, No. 5, May 2024")
    print("="*60)

    start_time = time.time()

    # Figure 5: 单轨道 (快)
    run_figure5()

    # Figure 9b: 偏移轨道 (中等)
    run_figure9b()

    # Figure 10: 完整星座 (慢)
    run_figure10()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  全部复现完成! 总用时: {elapsed:.1f} 秒")
    print(f"  输出目录: {os.path.abspath('output')}")
    print(f"{'='*60}")
