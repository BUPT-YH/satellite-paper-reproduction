"""
一键复现脚本 (修正版)
"""

import numpy as np
import sys
import os
import time

project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)
sys.path.insert(0, project_dir)

import config as cfg
from simulation import SatelliteNetwork, run_simulation
from resource_allocation import mm_resource_allocation, compute_throughput_per_beam
from beam_hopping_drl import DQNAgent
from plotting import (plot_fig3, plot_fig4, plot_fig5, plot_fig6,
                      plot_fig7, plot_fig8)

os.makedirs('output', exist_ok=True)

print("=" * 70)
print("论文复现 (修正版): Resource Allocation and Load Balancing for BH")
print("期刊: IEEE TWC, Vol.24, No.2, Feb. 2025")
print("=" * 70)


def run_fig3():
    """Fig.3: DQN训练收敛曲线"""
    print("\n[Fig.3] DQN Training Convergence...")
    t0 = time.time()

    n_sat, n_cells, n_beams = cfg.Ns, cfg.Nc, cfg.Nb
    n_cells_per_sat = cfg.omega_s

    net = SatelliteNetwork(n_sat, n_cells, n_beams, cfg.NL,
                            n_cells_per_sat, 0.7, 35.0, seed=42)
    agents = [DQNAgent(n_cells_per_sat, n_beams) for _ in range(n_sat)]

    all_rewards = [[] for _ in range(n_sat)]
    train_slots = cfg.train_slots

    for slot in range(train_slots):
        net.update_traffic(slot)
        net.age_buffer()
        net.update_buffer_arrival()

        bh_patterns = []
        for s in range(n_sat):
            action = agents[s].select_action(net.buffers[s])
            bh_patterns.append(action)

        f_alloc, Pb = mm_resource_allocation(
            bh_patterns, net.h, n_sat, n_beams, cfg.NL, cfg.P_sat, cfg.P_tot)
        throughputs = compute_throughput_per_beam(f_alloc, Pb, net.h, n_sat, n_beams, cfg.NL)

        actual_tps = []
        for s in range(n_sat):
            actual_tp = net.transmit_data(s, bh_patterns[s], throughputs)
            actual_tps.append(actual_tp)
        actual_tps = np.array(actual_tps)

        for s in range(n_sat):
            reward, _, _ = net.compute_reward(s, actual_tps)
            old_state = net.buffers[s].copy()
            agents[s].store_experience(old_state, bh_patterns[s], reward, net.buffers[s])
            all_rewards[s].append(reward)

            if slot % 4 == 0:
                agents[s].update()
            if slot % cfg.Cu == 0:
                agents[s].update_target_network()

        net.throughput_history.append(throughputs)
        if slot % 500 == 0:
            print(f"  Slot {slot}/{train_slots}")

    plot_fig3(all_rewards, 'output/fig3_training_reward.png')
    print(f"  Time: {time.time()-t0:.1f}s")


def run_fig4():
    """Fig.4: MM算法收敛 — 全星座/邻近协作/无协作"""
    print("\n[Fig.4] MM Algorithm Convergence...")
    t0 = time.time()

    n_sat, n_beams = cfg.Ns, cfg.Nb
    n_cells_per_sat = cfg.omega_s
    net = SatelliteNetwork(n_sat, cfg.Nc, n_beams, cfg.NL,
                            n_cells_per_sat, 0.7, 35.0, seed=42)

    bh_patterns = [np.random.choice(n_cells_per_sat, n_beams, replace=False).tolist()
                   for _ in range(n_sat)]

    convergence_data = {}
    for coop in ['full', 'neighbor', 'none']:
        label = {'full': 'whole', 'neighbor': 'neighbor', 'none': 'no_coop'}[coop]
        tp_list = []
        for q in range(10):
            f_alloc, Pb = mm_resource_allocation(
                bh_patterns, net.h, n_sat, n_beams, cfg.NL,
                cfg.P_sat, cfg.P_tot, cooperation=coop)
            tp = compute_throughput_per_beam(f_alloc, Pb, net.h, n_sat, n_beams, cfg.NL)
            tp_list.append(tp.sum())
        convergence_data[label] = tp_list

    plot_fig4(convergence_data, 'output/fig4_mm_convergence.png')
    print(f"  Time: {time.time()-t0:.1f}s")


def run_fig5():
    """Fig.5: 不同方法对比"""
    print("\n[Fig.5] Method Comparison...")
    t0 = time.time()

    traffic_rates = list(np.arange(14, 41, 6))
    methods = ['proposed', 'drl_avoid', 'max_uswg', 'pre_scheduling']
    n_cells_per_sat = 19
    n_cells = 160
    n_slots = 200

    throughput_data = {r: {} for r in traffic_rates}
    latency_data = {r: {} for r in traffic_rates}

    for r in traffic_rates:
        print(f"  Traffic: {r} Gbps")
        for method in methods:
            result = run_simulation(method=method, n_cells=n_cells,
                                     n_cells_per_sat=n_cells_per_sat,
                                     total_traffic_gbps=r,
                                     n_slots=n_slots, seed=42)
            throughput_data[r][method] = result['throughput_per_cell']
            latency_data[r][method] = result['latency_metric']
            print(f"    {method}: tp={result['throughput_per_cell']:.1f} Mbps, lt={result['latency_metric']:.3f}")

    plot_fig5(throughput_data, latency_data, 'output/fig5_method_comparison.png')
    print(f"  Time: {time.time()-t0:.1f}s")


def run_fig6():
    """Fig.6: 折衷系数 β"""
    print("\n[Fig.6] Beta Tradeoff...")
    t0 = time.time()

    beta_range = [0.0, 0.3, 0.5, 0.7, 1.0]
    methods = ['proposed', 'without_drl', 'without_ra', 'without_lb', 'original']
    n_slots = 200

    throughput_data = {m: [] for m in methods}
    latency_data = {m: [] for m in methods}

    for beta in beta_range:
        print(f"  beta={beta:.1f}")
        for method in methods:
            use_drl = method not in ('without_drl', 'original')
            use_ra = method not in ('without_ra', 'original')
            use_lb = method not in ('without_lb', 'original')
            actual = 'proposed' if use_drl else method
            result = run_simulation(method=actual, beta=beta,
                                     use_drl=use_drl, use_ra=use_ra, use_lb=use_lb,
                                     total_traffic_gbps=35.0,
                                     n_slots=n_slots, seed=42)
            throughput_data[method].append(result['throughput_per_cell'])
            latency_data[method].append(result['latency_metric'])

    plot_fig6(throughput_data, latency_data, beta_range, 'output/fig6_beta_tradeoff.png')
    print(f"  Time: {time.time()-t0:.1f}s")


def run_fig7():
    """Fig.7: 消融实验"""
    print("\n[Fig.7] Ablation Study...")
    t0 = time.time()

    traffic_rates = list(np.arange(14, 41, 6))
    methods = ['proposed', 'without_drl', 'without_ra', 'without_lb', 'original']
    n_slots = 200

    throughput_data = {r: {} for r in traffic_rates}
    latency_data = {r: {} for r in traffic_rates}

    for r in traffic_rates:
        print(f"  Traffic: {r} Gbps")
        for method in methods:
            use_drl = method not in ('without_drl', 'original')
            use_ra = method not in ('without_ra', 'original')
            use_lb = method not in ('without_lb', 'original')
            actual = 'proposed' if use_drl else method
            result = run_simulation(method=actual,
                                     use_drl=use_drl, use_ra=use_ra, use_lb=use_lb,
                                     total_traffic_gbps=r,
                                     n_slots=n_slots, seed=42)
            throughput_data[r][method] = result['throughput_per_cell']
            latency_data[r][method] = result['latency_metric']
            print(f"    {method}: tp={result['throughput_per_cell']:.1f} Mbps")

    plot_fig7(throughput_data, latency_data, 'output/fig7_traffic_load.png')
    print(f"  Time: {time.time()-t0:.1f}s")


def run_fig8():
    """Fig.8: 不同卫星数量"""
    print("\n[Fig.8] Satellite Number...")
    t0 = time.time()

    ns_range = [4, 8, 12, 16, 20]
    methods = ['proposed', 'without_drl', 'without_ra', 'without_lb', 'original']
    n_slots = 200

    throughput_data = {m: [] for m in methods}
    latency_data = {m: [] for m in methods}

    for n_sat in ns_range:
        n_cells = n_sat * 10
        n_cells_per_sat = max(10, 37 * n_sat // 16)
        print(f"  Ns={n_sat}")
        for method in methods:
            use_drl = method not in ('without_drl', 'original')
            use_ra = method not in ('without_ra', 'original')
            use_lb = method not in ('without_lb', 'original')
            actual = 'proposed' if use_drl else method
            result = run_simulation(method=actual, n_sat=n_sat,
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
    run_fig3()
    run_fig4()
    run_fig5()
    run_fig6()
    run_fig7()
    run_fig8()
    print(f"\nDone! Total: {time.time()-total_t0:.1f}s")
