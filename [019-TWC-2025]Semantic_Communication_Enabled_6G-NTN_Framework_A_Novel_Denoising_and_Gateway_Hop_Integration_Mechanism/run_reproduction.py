"""
一键复现脚本
运行所有仿真并生成图表
"""

import os
import sys
import numpy as np

# 确保在项目目录下运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from simulation import (
    compute_communication_times,
    compute_snr_per_gu,
    compute_latency_vs_gu_count,
    compute_psnr_vs_snr_compression,
)
from channel_model import SystemSimulator
from plotting import plot_fig5, plot_fig6, plot_fig7, plot_fig9

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("论文复现: Semantic Communication Enabled 6G-NTN Framework")
    print("IEEE TWC, Vol. 24, No. 12, 2025")
    print("=" * 60)

    # ===== 设置20 GU基准场景 =====
    print("\n[1/4] Setting up 20-GU baseline scenario...")
    sim = SystemSimulator(seed=RANDOM_SEED)
    scenario = sim.setup_scenario(NUM_GU, gw_assist_fraction=0.5)
    print(f"  GUs: {NUM_GU}, Direct: {scenario['num_direct']}, Assisted: {scenario['num_assisted']}")
    print(f"  SNR range: {10*np.log10(np.min(scenario['snr_direct'])):.2f} ~ "
          f"{10*np.log10(np.max(scenario['snr_direct'])):.2f} dB")

    # ===== Fig. 5: 通信时间对比 =====
    print("\n[2/4] Computing Fig. 5: Communication times...")
    fig5_data = compute_communication_times(scenario)
    plot_fig5(fig5_data, OUTPUT_DIR)

    # ===== Fig. 6: 各GU信道质量 =====
    print("\n[3/4] Computing Fig. 6: Channel quality per GU...")
    snr_direct, snr_gateway, assisted_idx = compute_snr_per_gu(scenario)
    plot_fig6(snr_direct, snr_gateway, assisted_idx, OUTPUT_DIR)

    # ===== Fig. 7: 延迟 vs GU数量 =====
    print("\n[4/4] Computing Fig. 7: Latency vs GU count...")
    latency_results = compute_latency_vs_gu_count()
    plot_fig7(latency_results, OUTPUT_DIR)

    # ===== Fig. 9: PSNR vs SNR (不同压缩率) =====
    print("\n[Bonus] Computing Fig. 9: PSNR vs SNR...")
    psnr_results = compute_psnr_vs_snr_compression()
    plot_fig9(psnr_results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == '__main__':
    # 从config导入常量
    from config import *
    main()
