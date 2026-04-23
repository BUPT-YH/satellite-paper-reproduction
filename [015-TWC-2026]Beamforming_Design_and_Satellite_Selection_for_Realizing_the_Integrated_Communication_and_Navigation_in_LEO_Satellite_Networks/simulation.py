"""
主仿真逻辑
实现 Fig. 2, Fig. 5, Fig. 8 的仿真数据生成
"""
import numpy as np
import config as cfg
from channel_model import build_system_variable, compute_gdop
from beamforming import compute_sum_rate_for_scheme
from satellite_selection import (
    communication_oriented_selection,
    navigation_oriented_selection, heuristic_ican_selection,
    coalitional_ican_selection, ocf_satellite_selection,
    evaluate_selection
)


def balanced_default_selection(S, C, I, K, seed=42):
    """
    生成均衡的默认卫星选择方案
    确保每个 UE 选 I 颗卫星，且每颗卫星不超过 K 个 UE
    使用贪心分配保证负载均衡
    """
    np.random.seed(seed)
    alpha = {}
    sat_load = np.zeros(S, dtype=int)

    for c in range(C):
        # 按负载排序，优先选负载低的卫星
        candidates = list(range(S))
        # 加入随机扰动打破平局
        loads = [(sat_load[s] + np.random.uniform(0, 0.1), s) for s in candidates]
        loads.sort()

        selected = []
        for _, s in loads:
            if len(selected) >= I:
                break
            if sat_load[s] < K:
                selected.append(s)
                sat_load[s] += 1
                alpha[(s, c)] = 1

    return alpha


def simulate_fig2():
    """
    仿真 Fig. 2: 不同波束赋形方案的速率性能
    参数: S=12, I=6, ρ=1, γ_com=12.5 Mbps, γ_nav=6
    X轴: UE 数量 (4-9)
    """
    print("Simulating Fig. 2: Beamforming comparison...")
    results = {}

    for C in cfg.C_range:
        print(f"  C = {C}")
        sat_pos, ue_pos, channels = build_system_variable(S=cfg.S, C=C, seed=cfg.SEED)
        alpha = balanced_default_selection(cfg.S, C, cfg.I, cfg.K, seed=cfg.SEED)

        results[C] = {}
        for scheme in ['DC', 'MRT', 'ZF', 'MMSE', 'ST-ZF']:
            rate = compute_sum_rate_for_scheme(
                channels, alpha, scheme, cfg.P_max_watt, cfg.noise_power_watt,
                S=cfg.S, C=C
            )
            label = 'Proposed DC' if scheme == 'DC' else scheme
            results[C][label] = rate

        print(f"    DC={results[C]['Proposed DC']/1e6:.2f}, "
              f"MRT={results[C]['MRT']/1e6:.2f}, "
              f"ZF={results[C]['ZF']/1e6:.2f}, "
              f"MMSE={results[C]['MMSE']/1e6:.2f}, "
              f"ST-ZF={results[C]['ST-ZF']/1e6:.2f} Mbps")

    return results


def simulate_fig5():
    """
    仿真 Fig. 5: 不同卫星选择方案的速率和 GDOP 性能
    参数: C=7, I=6, γ_com=12.5 Mbps, γ_nav=6
    X轴: 可见卫星数 (10-16)
    """
    print("Simulating Fig. 5: Satellite selection comparison...")
    results = {}

    for S in cfg.S_range:
        print(f"  S = {S}")
        sat_pos, ue_pos, channels = build_system_variable(S=S, C=cfg.C, seed=cfg.SEED)
        results[S] = {}

        # 1. Proposed OCF (ρ=0, 0.5, 1)
        for rho_val, rho_label in [(1.0, 'Proposed (ρ=1)'),
                                    (0.5, 'Proposed (ρ=0.5)'),
                                    (0.0, 'Proposed (ρ=0)')]:
            alpha = ocf_satellite_selection(
                channels, sat_pos, ue_pos, S, cfg.C, cfg.I, cfg.K,
                cfg.P_max_watt, cfg.noise_power_watt,
                cfg.gamma_com, cfg.gamma_nav, rho_val, max_iter=cfg.J_OCF
            )
            rate, gdop = evaluate_selection(
                channels, sat_pos, ue_pos, alpha, S, cfg.C, cfg.I,
                cfg.P_max_watt, cfg.noise_power_watt, rho_val
            )
            results[S][rho_label] = {'rate': rate, 'gdop': gdop}

        # 2. Communication-oriented
        alpha_comm = communication_oriented_selection(
            channels, sat_pos, ue_pos, S, cfg.C, cfg.I, cfg.K,
            cfg.P_max_watt, cfg.noise_power_watt
        )
        rate_comm, gdop_comm = evaluate_selection(
            channels, sat_pos, ue_pos, alpha_comm, S, cfg.C, cfg.I,
            cfg.P_max_watt, cfg.noise_power_watt, 1.0
        )
        results[S]['Comm-oriented'] = {'rate': rate_comm, 'gdop': gdop_comm}

        # 3. Navigation-oriented
        alpha_nav = navigation_oriented_selection(
            sat_pos, ue_pos, S, cfg.C, cfg.I, cfg.K
        )
        rate_nav, gdop_nav = evaluate_selection(
            channels, sat_pos, ue_pos, alpha_nav, S, cfg.C, cfg.I,
            cfg.P_max_watt, cfg.noise_power_watt, 0.0
        )
        results[S]['Nav-oriented'] = {'rate': rate_nav, 'gdop': gdop_nav}

        # 4. Heuristic ICAN
        alpha_h = heuristic_ican_selection(
            channels, sat_pos, ue_pos, S, cfg.C, cfg.I, cfg.K,
            cfg.P_max_watt, cfg.noise_power_watt
        )
        rate_h, gdop_h = evaluate_selection(
            channels, sat_pos, ue_pos, alpha_h, S, cfg.C, cfg.I,
            cfg.P_max_watt, cfg.noise_power_watt, 0.5
        )
        results[S]['Heuristic ICAN'] = {'rate': rate_h, 'gdop': gdop_h}

        # 5. Coalitional ICAN
        alpha_coal = coalitional_ican_selection(
            channels, sat_pos, ue_pos, S, cfg.C, cfg.I, cfg.K,
            cfg.P_max_watt, cfg.noise_power_watt
        )
        rate_coal, gdop_coal = evaluate_selection(
            channels, sat_pos, ue_pos, alpha_coal, S, cfg.C, cfg.I,
            cfg.P_max_watt, cfg.noise_power_watt, 0.5
        )
        results[S]['Coalitional ICAN'] = {'rate': rate_coal, 'gdop': gdop_coal}

        print(f"    OCF(ρ=1): rate={results[S]['Proposed (ρ=1)']['rate']/1e6:.2f} Mbps, "
              f"GDOP={results[S]['Proposed (ρ=1)']['gdop']:.2f}")

    return results


def simulate_fig8():
    """
    仿真 Fig. 8: 通信速率与导航 GDOP 的权衡
    参数: S=12, C=7, γ_com=12.5 Mbps, γ_nav=6
    X轴: 权重因子 ρ (0-1)
    不同曲线: I=5, 6, 7
    """
    print("Simulating Fig. 8: Communication-Navigation trade-off...")
    results = {}

    for I_val in cfg.I_range:
        print(f"  I = {I_val}")
        results[I_val] = {}
        sat_pos, ue_pos, channels = build_system_variable(S=cfg.S, C=cfg.C, seed=cfg.SEED)

        for rho_val in cfg.rho_range:
            alpha = ocf_satellite_selection(
                channels, sat_pos, ue_pos, cfg.S, cfg.C, I_val, cfg.K,
                cfg.P_max_watt, cfg.noise_power_watt,
                cfg.gamma_com, cfg.gamma_nav, rho_val, max_iter=cfg.J_OCF
            )
            rate, gdop = evaluate_selection(
                channels, sat_pos, ue_pos, alpha, cfg.S, cfg.C, I_val,
                cfg.P_max_watt, cfg.noise_power_watt, rho_val
            )
            results[I_val][rho_val] = {'rate': rate, 'gdop': gdop}

        print(f"    ρ=0: rate={results[I_val][0.0]['rate']/1e6:.2f} Mbps, "
              f"GDOP={results[I_val][0.0]['gdop']:.2f}")
        print(f"    ρ=1: rate={results[I_val][1.0]['rate']/1e6:.2f} Mbps, "
              f"GDOP={results[I_val][1.0]['gdop']:.2f}")

    return results
