"""
一键复现脚本 — 续运行版
跳过已完成的图表 (fig3, fig4, fig5)，继续运行剩余
"""

import numpy as np
import sys
import os
import time

project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)
sys.path.insert(0, project_dir)

import config as cfg
from simulation import run_simulation
from plotting import (plot_fig6, plot_fig7, plot_fig8)

os.makedirs('output', exist_ok=True)

print("=" * 70)
print("论文复现 (续): Fig 6, 7, 8")
print("=" * 70)


def run_fig6():
    """Fig.6: 折衷系数 β 分析 — 简化版"""
    print("\n[Fig.6] Trade-off Coefficient β Analysis...")
    t0 = time.time()

    # 减少 β 采样点: 5个点
    beta_range = [0.0, 0.3, 0.5, 0.7, 1.0]
    methods = ['proposed', 'without_drl', 'without_ra', 'without_lb', 'original']
    n_slots = 150  # 进一步减少

    throughput_data = {m: [] for m in methods}
    latency_data = {m: [] for m in methods}

    for beta in beta_range:
        print(f"  β = {beta:.1f}")
        for method in methods:
            use_drl = method != 'without_drl' and method != 'original'
            use_ra = method != 'without_ra' and method != 'original'
            use_lb = method != 'without_lb' and method != 'original'
            actual_method = 'proposed' if use_drl else method
            result = run_simulation(method=actual_method, beta=beta,
                                     use_drl=use_drl, use_ra=use_ra, use_lb=use_lb,
                                     total_traffic_gbps=35.0,
                                     n_slots=n_slots, seed=42)
            throughput_data[method].append(result['throughput_per_cell'])
            latency_data[method].append(result['latency_metric'])

    plot_fig6(throughput_data, latency_data, beta_range, 'output/fig6_beta_tradeoff.png')
    print(f"  Time: {time.time()-t0:.1f}s")


def run_fig7():
    """Fig.7: 不同输入流量 (消融实验)"""
    print("\n[Fig.7] Ablation Study vs Input Traffic...")
    t0 = time.time()

    traffic_rates = list(np.arange(14, 41, 6))
    methods = ['proposed', 'without_drl', 'without_ra', 'without_lb', 'original']
    n_slots = 150

    throughput_data = {r: {} for r in traffic_rates}
    latency_data = {r: {} for r in traffic_rates}

    for r in traffic_rates:
        print(f"  Traffic rate: {r} Gbps")
        for method in methods:
            use_drl = method != 'without_drl' and method != 'original'
            use_ra = method != 'without_ra' and method != 'original'
            use_lb = method != 'without_lb' and method != 'original'
            actual_method = 'proposed' if use_drl else method
            result = run_simulation(method=actual_method,
                                     use_drl=use_drl, use_ra=use_ra, use_lb=use_lb,
                                     total_traffic_gbps=r,
                                     n_slots=n_slots, seed=42)
            throughput_data[r][method] = result['throughput_per_cell']
            latency_data[r][method] = result['latency_metric']

    plot_fig7(throughput_data, latency_data, 'output/fig7_traffic_load.png')
    print(f"  Time: {time.time()-t0:.1f}s")


def run_fig8():
    """Fig.8: 不同卫星数量"""
    print("\n[Fig.8] Different Numbers of Satellites...")
    t0 = time.time()

    ns_range = [4, 8, 12, 16, 20]
    methods = ['proposed', 'without_drl', 'without_ra', 'without_lb', 'original']
    n_slots = 150

    throughput_data = {m: [] for m in methods}
    latency_data = {m: [] for m in methods}

    for n_sat in ns_range:
        n_cells = n_sat * 10
        n_cells_per_sat = max(10, 37 * n_sat // 16)
        print(f"  Ns = {n_sat}")
        for method in methods:
            use_drl = method != 'without_drl' and method != 'original'
            use_ra = method != 'without_ra' and method != 'original'
            use_lb = method != 'without_lb' and method != 'original'
            actual_method = 'proposed' if use_drl else method
            result = run_simulation(method=actual_method, n_sat=n_sat,
                                     n_cells=n_cells,
                                     n_cells_per_sat=n_cells_per_sat,
                                     use_drl=use_drl, use_ra=use_ra, use_lb=use_lb,
                                     total_traffic_gbps=35.0,
                                     n_slots=n_slots, seed=42)
            throughput_data[method].append(result['throughput_per_cell'])
            latency_data[method].append(result['latency_metric'])

    plot_fig8(throughput_data, latency_data, ns_range, 'output/fig8_satellite_number.png')
    print(f"  Time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    total_t0 = time.time()
    run_fig6()
    run_fig7()
    run_fig8()
    print(f"\nDone! Total: {time.time()-total_t0:.1f}s")
