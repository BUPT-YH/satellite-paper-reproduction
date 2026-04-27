"""
一键复现脚本
复现论文 Fig. 2, 3, 4, 6
"""

import numpy as np
import sys
import os
import time

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from channel_model import SatelliteNetwork
from simulation import run_simulation, run_bcd_convergence
from plotting import (plot_bcd_convergence, plot_v_tradeoff,
                      plot_interference_threshold, plot_method_comparison)
from optimizer import BCDOptimizer


def reproduce_fig2():
    """
    Fig. 2: BCD 算法收敛曲线 (不同卫星数量)
    """
    print("\n" + "=" * 60)
    print("Fig. 2: BCD 算法收敛性验证")
    print("=" * 60)

    results = run_bcd_convergence(num_sats_list=[2, 3, 4], V=200)
    filepath = plot_bcd_convergence(results)

    print(f"\nFig. 2 结果:")
    for ns, obj in results.items():
        print(f"  {ns} 卫星: {len(obj)} 次迭代收敛, 最终 Λ = {obj[-1]:.2f}")

    return results


def reproduce_fig3():
    """
    Fig. 3: 不同 V 值下的平均队列长度和平均功率
    通信需求 20 Gbps
    """
    print("\n" + "=" * 60)
    print("Fig. 3: 权衡系数 V 对性能的影响")
    print("=" * 60)

    net = SatelliteNetwork(num_sats=NUM_SATELLITES)
    results = {}

    for V in V_RANGE:
        print(f"\n--- V = {V} ---")
        res = run_simulation(net, method_name='proposed', V=V,
                             demand=DEMAND_DEFAULT, num_slots=300, verbose=True)
        results[V] = res
        print(f"  平均功率: {res['avg_power']:.1f} W, "
              f"平均队列: {res['avg_queue']:.1f}")

    filepath = plot_v_tradeoff(results)
    print(f"\nFig. 3 结果总结:")
    for V, res in results.items():
        print(f"  V={V}: 功率={res['avg_power']:.1f}W, 队列={res['avg_queue']:.1f}")

    return results


def reproduce_fig4():
    """
    Fig. 4: 不同干扰阈值和 ISL 传输限制下的平均功率
    """
    print("\n" + "=" * 60)
    print("Fig. 4: 干扰阈值与 ISL 传输限制的影响")
    print("=" * 60)

    net = SatelliteNetwork(num_sats=NUM_SATELLITES)

    # ISL 传输限制取值
    c_max_values = [5, 10, 20]
    # 干扰阈值范围 (dBW)
    z_max_dbw_range = Z_MAX_RANGE_DBW

    results = {}

    for c_max in c_max_values:
        for z_dbw in z_max_dbw_range:
            z_lin = 10 ** (z_dbw / 10.0)  # dBW → W
            print(f"\n--- c_max={c_max}, Z_max={z_dbw} dBW ---")
            res = run_simulation(net, method_name='proposed', V=V_DEFAULT,
                                 demand=DEMAND_DEFAULT, z_max_lin=z_lin,
                                 c_max=c_max, num_slots=150, verbose=True)
            results[(c_max, z_dbw)] = res
            print(f"  平均功率: {res['avg_power']:.1f} W")

    filepath = plot_interference_threshold(results, c_max_values, z_max_dbw_range)

    print(f"\nFig. 4 结果总结:")
    for c_max in c_max_values:
        powers = [results[(c_max, z)]['avg_power'] for z in z_max_dbw_range]
        print(f"  c_max={c_max}: " + ", ".join(
            [f"{z}dBW→{p:.1f}W" for z, p in zip(z_max_dbw_range, powers)]))

    return results


def reproduce_fig6():
    """
    Fig. 6: 不同方法在不同通信需求下的平均功率 (无限存储)
    """
    print("\n" + "=" * 60)
    print("Fig. 6: 方法对比 (不同通信需求)")
    print("=" * 60)

    method_list = ['proposed', 'drl', 'pre_scheduling',
                   'no_freq_div', 'no_lb', 'max_uswg']
    demand_gbps = [d / 1e9 for d in DEMAND_RANGE]
    results = {}

    for method in method_list:
        for demand in DEMAND_RANGE:
            d_gbps = demand / 1e9
            print(f"\n--- {method}, demand={d_gbps:.0f} Gbps ---")
            net = SatelliteNetwork(num_sats=NUM_SATELLITES)
            res = run_simulation(net, method_name=method, V=V_DEFAULT,
                                 demand=demand, num_slots=150, verbose=True)
            results[(method, d_gbps)] = res
            print(f"  平均功率: {res['avg_power']:.1f} W")

    filepath = plot_method_comparison(results, demand_gbps)

    print(f"\nFig. 6 结果总结 (20 Gbps 时各方法功率):")
    method_labels = {
        'proposed': 'Proposed', 'drl': 'DRL', 'pre_scheduling': 'Pre-sched',
        'no_freq_div': 'No Freq Div', 'no_lb': 'No LB', 'max_uswg': 'Max USWG'
    }
    for m, label in method_labels.items():
        key = (m, 20.0)
        if key in results:
            print(f"  {label}: {results[key]['avg_power']:.1f} W")

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("论文复现: Joint Resource Management and Load Balancing")
    print("         in Multi-Satellite Beam Hopping")
    print("=" * 60)

    start_time = time.time()

    # Fig. 2: BCD 收敛
    fig2_results = reproduce_fig2()

    # Fig. 3: V 权衡
    fig3_results = reproduce_fig3()

    # Fig. 4: 干扰阈值
    fig4_results = reproduce_fig4()

    # Fig. 6: 方法对比
    fig6_results = reproduce_fig6()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"复现完成! 总耗时: {elapsed/60:.1f} 分钟")
    print(f"输出目录: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')}")
    print(f"{'=' * 60}")
