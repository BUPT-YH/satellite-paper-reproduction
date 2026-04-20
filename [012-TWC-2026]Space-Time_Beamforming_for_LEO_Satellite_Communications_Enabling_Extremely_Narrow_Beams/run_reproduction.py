"""
一键复现脚本
运行: python run_reproduction.py
"""

import numpy as np
import os
import sys

from config import P_range_dBm, K_partial, K_full, M_full, dBm_to_W
from simulation import simulate_partial_network, simulate_full_network, simulate_fig8
from plotting import plot_fig5, plot_fig7, plot_fig8

# 输出目录
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Monte Carlo 次数
N_CH = 200


def run_fig5():
    """Fig. 5: 部分连接网络 SE vs P (K=3)"""
    print("=" * 60)
    print("Fig. 5: Partially Connected Network (K=3)")
    print("=" * 60)

    results = {'P_dBm': P_range_dBm, 'perfect': {}, 'imperfect': {}}

    for csit_type in ['perfect', 'imperfect']:
        add_error = (csit_type == 'imperfect')
        print(f"\n  CSIT: {csit_type}")

        for method in ['MRT', 'ZF', 'SLNR', 'TDMA', 'ST-ZF']:
            se_list = []
            for P_dBm in P_range_dBm:
                res = simulate_partial_network(
                    K_partial, P_dBm, M=2, n_ch=N_CH, add_delay_error=add_error
                )
                se_list.append(res[method])
                print(f"    P={P_dBm} dBm, {method}: {res[method]:.3f} bps/Hz")
            results[csit_type][method] = np.array(se_list)

    plot_fig5(results, OUTPUT_DIR)
    return results


def run_fig7():
    """Fig. 7: 全连接网络 SE vs P (M=3, K=4)"""
    print("\n" + "=" * 60)
    print("Fig. 7: Fully Connected Network (M=3, K=4)")
    print("=" * 60)

    results = {'P_dBm': P_range_dBm, 'perfect': {}, 'imperfect': {}}

    for csit_type in ['perfect', 'imperfect']:
        add_error = (csit_type == 'imperfect')
        print(f"\n  CSIT: {csit_type}")

        for method in ['MRT', 'SLNR', 'TDMA', 'ST-SLNR']:
            se_list = []
            for P_dBm in P_range_dBm:
                res = simulate_full_network(
                    K_full, P_dBm, M=M_full, n_ch=N_CH, add_delay_error=add_error
                )
                se_list.append(res[method])
                print(f"    P={P_dBm} dBm, {method}: {res[method]:.3f} bps/Hz")
            results[csit_type][method] = np.array(se_list)

    plot_fig7(results, OUTPUT_DIR)
    return results


def run_fig8():
    """Fig. 8: 全连接网络 ST-SLNR SE vs K 和 M (P=40 dBm)"""
    print("\n" + "=" * 60)
    print("Fig. 8: ST-SLNR vs K and M (P=40 dBm)")
    print("=" * 60)

    K_range = [2, 3, 4, 5]
    M_range = [1, 2, 3, 4, 5]

    results = simulate_fig8(K_range, M_range, P_dBm=40, n_ch=N_CH)

    plot_fig8(results, K_range, M_range, OUTPUT_DIR)
    return results


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print("Space-Time Beamforming for LEO Satellite Communications")
    print("Paper Reproduction - [012-TWC-2026]")
    print(f"Monte Carlo iterations: {N_CH}")
    print(f"Output directory: {OUTPUT_DIR}/")

    results_fig5 = run_fig5()
    results_fig7 = run_fig7()
    results_fig8 = run_fig8()

    print("\n" + "=" * 60)
    print("All simulations completed!")
    print(f"Results saved to {OUTPUT_DIR}/")
    print("=" * 60)
