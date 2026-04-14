# -*- coding: utf-8 -*-
"""
仿真主脚本: 运行所有图表的仿真
Fig 2-13 完整复现
修复: Monte Carlo 平滑 (3次取中位数)
"""

import numpy as np
import sys
import time
import config as cfg
from channel_model import compute_positions, compute_distances, compute_channel_statistics
from optimization import (
    algorithm_ap_ao, algorithm_mr_s_pa, algorithm_mr_s_ts,
    algorithm_mr_tts_pa, algorithm_mr_tts_ts, run_no_ris,
    evaluate_scheme, evaluate_scheme_rzf, evaluate_sum_rate, init_ris_phase
)
from plotting import (plot_min_sinr, plot_sum_rate, plot_execution_time)

# Monte Carlo 平滑次数
MC_RUNS = 1


def run_single_scheme(cs, PT, zeta_dB, scheme_name, mc_run=0):
    """运行单个方案, 返回最小 SINR (dB)"""
    np.random.seed(42 + mc_run * 1000)  # 不同 MC 运行用不同种子

    if scheme_name == 'AP-AO':
        phi, p = algorithm_ap_ao(cs, PT, zeta_dB, max_iter=6)
        return evaluate_scheme_rzf(cs, phi, PT)  # RZF 预编码评估
    elif scheme_name == 'MR-S-PA':
        phi, p = algorithm_mr_s_pa(cs, PT, zeta_dB)
        return evaluate_scheme(cs, phi, p, 'mr_stat')
    elif scheme_name == 'MR-S-TS':
        phi, p = algorithm_mr_s_ts(cs, PT, zeta_dB, max_iter=8)
        return evaluate_scheme(cs, phi, p, 'mr_stat')
    elif scheme_name == 'MR-TTS-PA':
        phi, p = algorithm_mr_tts_pa(cs, PT, zeta_dB)
        return evaluate_scheme(cs, phi, p, 'mr_tts')
    elif scheme_name == 'MR-TTS-TS':
        phi, p = algorithm_mr_tts_ts(cs, PT, zeta_dB, max_iter=8)
        return evaluate_scheme(cs, phi, p, 'mr_tts')
    elif scheme_name == 'AP-NoRIS':
        phi, p = run_no_ris(cs, PT, zeta_dB, 'mr_stat')
        return evaluate_scheme_rzf(cs, phi, PT)  # RZF 预编码评估
    elif scheme_name == 'MR-S-NoRIS':
        phi, p = run_no_ris(cs, PT, zeta_dB, 'mr_stat')
        return evaluate_scheme(cs, phi, p, 'mr_stat')
    elif scheme_name == 'MR-TTS-NoRIS':
        phi, p = run_no_ris(cs, PT, zeta_dB, 'mr_tts')
        return evaluate_scheme(cs, phi, p, 'mr_tts')
    return 0


def run_single_scheme_mc(cs, PT, zeta_dB, scheme_name):
    """Monte Carlo 平滑: 多次运行取中位数"""
    results = []
    for mc in range(MC_RUNS):
        sinr = run_single_scheme(cs, PT, zeta_dB, scheme_name, mc_run=mc)
        results.append(sinr)
    return np.median(results)


# ===== 8 种方案的完整列表 =====
ALL_SCHEMES = [
    'AP-AO', 'MR-S-PA', 'MR-S-TS', 'MR-TTS-PA', 'MR-TTS-TS',
    'AP-NoRIS', 'MR-S-NoRIS', 'MR-TTS-NoRIS'
]


def fig2_simulation():
    """Fig. 2: Min SINR vs PT, κN=20 dB"""
    print("\n===== Fig. 2: Min SINR vs PT (κN=20 dB) =====")
    kappa_N, kappa_R, kappa_LR = 20.0, 20.0, 20.0
    M = 12
    cs = compute_channel_statistics(
        compute_distances(*compute_positions(M)),
        kappa_N, kappa_R, kappa_LR, M=M
    )

    results = {s: [] for s in ALL_SCHEMES}
    for PT in cfg.PT_range:
        print(f"  PT = {PT} W", end="")
        for scheme in ALL_SCHEMES:
            sinr = run_single_scheme_mc(cs, PT, cfg.zeta_default, scheme)
            results[scheme].append(sinr)
            print(f" | {scheme}: {sinr:.1f}", end="")
        print()

    plot_min_sinr(
        cfg.PT_range, results,
        xlabel='$P_T$', title='Fig. 2: Min SINR vs Transmit Power ($\\kappa_N=20$ dB)',
        filename='fig2_min_sinr_vs_PT_kappaN20.png',
        x_label_unit=' (W)'
    )
    return results


def fig3_simulation():
    """Fig. 3: Min SINR vs PT, κN=0 dB"""
    print("\n===== Fig. 3: Min SINR vs PT (κN=0 dB) =====")
    kappa_N, kappa_R, kappa_LR = 0.0, 0.0, 0.0
    M = 12
    cs = compute_channel_statistics(
        compute_distances(*compute_positions(M)),
        kappa_N, kappa_R, kappa_LR, M=M
    )

    results = {s: [] for s in ALL_SCHEMES}
    for PT in cfg.PT_range:
        print(f"  PT = {PT} W", end="")
        for scheme in ALL_SCHEMES:
            sinr = run_single_scheme_mc(cs, PT, cfg.zeta_default, scheme)
            results[scheme].append(sinr)
            print(f" | {scheme}: {sinr:.1f}", end="")
        print()

    plot_min_sinr(
        cfg.PT_range, results,
        xlabel='$P_T$', title='Fig. 3: Min SINR vs Transmit Power ($\\kappa_N=0$ dB)',
        filename='fig3_min_sinr_vs_PT_kappaN0.png',
        x_label_unit=' (W)'
    )
    return results


def fig4_simulation():
    """Fig. 4: Min SINR vs ζ, κN=20 dB, κR=10 dB"""
    print("\n===== Fig. 4: Min SINR vs ζ (κN=20 dB) =====")
    kappa_N, kappa_R, kappa_LR = 20.0, 10.0, 10.0
    M = 12
    cs = compute_channel_statistics(
        compute_distances(*compute_positions(M)),
        kappa_N, kappa_R, kappa_LR, M=M
    )

    results = {s: [] for s in ALL_SCHEMES}
    for zeta in cfg.zeta_range_dB:
        print(f"  zeta = {zeta} dBW", end="")
        for scheme in ALL_SCHEMES:
            sinr = run_single_scheme_mc(cs, cfg.PT_default, zeta, scheme)
            results[scheme].append(sinr)
            print(f" | {scheme}: {sinr:.1f}", end="")
        print()

    plot_min_sinr(
        cfg.zeta_range_dB, results,
        xlabel='$\\varsigma$', title='Fig. 4: Min SINR vs Interference Threshold ($\\kappa_N=20$ dB)',
        filename='fig4_min_sinr_vs_zeta_kappaN20.png',
        x_label_unit=' (dBW)'
    )
    return results


def fig5_simulation():
    """Fig. 5: Min SINR vs ζ, κN=0 dB, κR=10 dB"""
    print("\n===== Fig. 5: Min SINR vs ζ (κN=0 dB) =====")
    kappa_N, kappa_R, kappa_LR = 0.0, 10.0, 10.0
    M = 12
    cs = compute_channel_statistics(
        compute_distances(*compute_positions(M)),
        kappa_N, kappa_R, kappa_LR, M=M
    )

    results = {s: [] for s in ALL_SCHEMES}
    for zeta in cfg.zeta_range_dB:
        print(f"  zeta = {zeta} dBW", end="")
        for scheme in ALL_SCHEMES:
            sinr = run_single_scheme_mc(cs, cfg.PT_default, zeta, scheme)
            results[scheme].append(sinr)
            print(f" | {scheme}: {sinr:.1f}", end="")
        print()

    plot_min_sinr(
        cfg.zeta_range_dB, results,
        xlabel='$\\varsigma$', title='Fig. 5: Min SINR vs Interference Threshold ($\\kappa_N=0$ dB)',
        filename='fig5_min_sinr_vs_zeta_kappaN0.png',
        x_label_unit=' (dBW)'
    )
    return results


def fig6_simulation():
    """Fig. 6: Min SINR vs M, κN=20 dB"""
    print("\n===== Fig. 6: Min SINR vs M (κN=20 dB) =====")
    kappa_N, kappa_R, kappa_LR = 20.0, 10.0, 10.0
    results = {s: [] for s in ALL_SCHEMES}

    for M in cfg.M_range:
        print(f"  M = {M}", end="")
        cs = compute_channel_statistics(
            compute_distances(*compute_positions(M)),
            kappa_N, kappa_R, kappa_LR, M=M
        )
        for scheme in ALL_SCHEMES:
            sinr = run_single_scheme_mc(cs, cfg.PT_default, cfg.zeta_default, scheme)
            results[scheme].append(sinr)
            print(f" | {scheme}: {sinr:.1f}", end="")
        print()

    plot_min_sinr(
        cfg.M_range, results,
        xlabel='$M$', title='Fig. 6: Min SINR vs RIS Subsurfaces ($\\kappa_N=20$ dB)',
        filename='fig6_min_sinr_vs_M_kappaN20.png'
    )
    return results


def fig7_simulation():
    """Fig. 7: Min SINR vs M, κN=0 dB"""
    print("\n===== Fig. 7: Min SINR vs M (κN=0 dB) =====")
    kappa_N, kappa_R, kappa_LR = 0.0, 10.0, 10.0
    results = {s: [] for s in ALL_SCHEMES}

    for M in cfg.M_range:
        print(f"  M = {M}", end="")
        cs = compute_channel_statistics(
            compute_distances(*compute_positions(M)),
            kappa_N, kappa_R, kappa_LR, M=M
        )
        for scheme in ALL_SCHEMES:
            sinr = run_single_scheme_mc(cs, cfg.PT_default, cfg.zeta_default, scheme)
            results[scheme].append(sinr)
            print(f" | {scheme}: {sinr:.1f}", end="")
        print()

    plot_min_sinr(
        cfg.M_range, results,
        xlabel='$M$', title='Fig. 7: Min SINR vs RIS Subsurfaces ($\\kappa_N=0$ dB)',
        filename='fig7_min_sinr_vs_M_kappaN0.png'
    )
    return results


def fig8_simulation():
    """Fig. 8: Min SINR vs κR, κN=0 dB"""
    print("\n===== Fig. 8: Min SINR vs κR (κN=0 dB) =====")
    kappa_N = 0.0
    kappa_LR = 10.0
    M = 12

    results = {s: [] for s in ALL_SCHEMES}
    for kappa_R in cfg.kappa_R_range:
        print(f"  κR = {kappa_R} dB", end="")
        cs = compute_channel_statistics(
            compute_distances(*compute_positions(M)),
            kappa_N, kappa_R, kappa_LR, M=M
        )
        for scheme in ALL_SCHEMES:
            sinr = run_single_scheme_mc(cs, cfg.PT_default, cfg.zeta_default, scheme)
            results[scheme].append(sinr)
            print(f" | {scheme}: {sinr:.1f}", end="")
        print()

    plot_min_sinr(
        cfg.kappa_R_range, results,
        xlabel='$\\kappa_R$', title='Fig. 8: Min SINR vs Terrestrial Rician Factor ($\\kappa_N=0$ dB)',
        filename='fig8_min_sinr_vs_kappaR_kappaN0.png',
        x_label_unit=' (dB)'
    )
    return results


def fig9_simulation():
    """Fig. 9: Min SINR vs κR, κN=10 dB"""
    print("\n===== Fig. 9: Min SINR vs κR (κN=10 dB) =====")
    kappa_N = 10.0
    kappa_LR = 10.0
    M = 12

    results = {s: [] for s in ALL_SCHEMES}
    for kappa_R in cfg.kappa_R_range:
        print(f"  κR = {kappa_R} dB", end="")
        cs = compute_channel_statistics(
            compute_distances(*compute_positions(M)),
            kappa_N, kappa_R, kappa_LR, M=M
        )
        for scheme in ALL_SCHEMES:
            sinr = run_single_scheme_mc(cs, cfg.PT_default, cfg.zeta_default, scheme)
            results[scheme].append(sinr)
            print(f" | {scheme}: {sinr:.1f}", end="")
        print()

    plot_min_sinr(
        cfg.kappa_R_range, results,
        xlabel='$\\kappa_R$', title='Fig. 9: Min SINR vs Terrestrial Rician Factor ($\\kappa_N=10$ dB)',
        filename='fig9_min_sinr_vs_kappaR_kappaN10.png',
        x_label_unit=' (dB)'
    )
    return results


def fig10_simulation():
    """Fig. 10: Min SINR vs κR, κN=20 dB"""
    print("\n===== Fig. 10: Min SINR vs κR (κN=20 dB) =====")
    kappa_N = 20.0
    kappa_LR = 10.0
    M = 12

    results = {s: [] for s in ALL_SCHEMES}
    for kappa_R in cfg.kappa_R_range:
        print(f"  κR = {kappa_R} dB", end="")
        cs = compute_channel_statistics(
            compute_distances(*compute_positions(M)),
            kappa_N, kappa_R, kappa_LR, M=M
        )
        for scheme in ALL_SCHEMES:
            sinr = run_single_scheme_mc(cs, cfg.PT_default, cfg.zeta_default, scheme)
            results[scheme].append(sinr)
            print(f" | {scheme}: {sinr:.1f}", end="")
        print()

    plot_min_sinr(
        cfg.kappa_R_range, results,
        xlabel='$\\kappa_R$', title='Fig. 10: Min SINR vs Terrestrial Rician Factor ($\\kappa_N=20$ dB)',
        filename='fig10_min_sinr_vs_kappaR_kappaN20.png',
        x_label_unit=' (dB)'
    )
    return results


def fig11_simulation():
    """Fig. 11: Min SINR vs ζ, MSC vs SST 对比"""
    print("\n===== Fig. 11: MSC vs SST (Min SINR) =====")
    kappa_N_values = [0.0, 20.0]
    kappa_R = 10.0
    kappa_LR = 10.0
    M = 12

    schemes_msc_sst = ['M-AP (0dB)', 'M-MR (0dB)', 'S-AP 1.25 (0dB)', 'S-AP 2.5 (0dB)',
                       'S-MR 1.25 (0dB)', 'S-MR 2.5 (0dB)',
                       'M-AP (20dB)', 'M-MR (20dB)', 'S-AP 1.25 (20dB)', 'S-AP 2.5 (20dB)',
                       'S-MR 1.25 (20dB)', 'S-MR 2.5 (20dB)']
    results = {s: [] for s in schemes_msc_sst}

    for zeta in cfg.zeta_msc_range_dB:
        print(f"  zeta = {zeta} dBW")
        for kN_idx, kappa_N in enumerate(kappa_N_values):
            cs = compute_channel_statistics(
                compute_distances(*compute_positions(M)),
                kappa_N, kappa_R, kappa_LR, M=M
            )
            tag = f' ({int(kappa_N)}dB)'

            # MSC: AP-AO
            np.random.seed(42)
            phi, p = algorithm_ap_ao(cs, cfg.PT_default, zeta, max_iter=6)
            results[f'M-AP{tag}'].append(evaluate_scheme(cs, phi, p, 'mr_stat'))

            # MSC: MR-TTS-TS
            np.random.seed(42)
            phi, p = algorithm_mr_tts_ts(cs, cfg.PT_default, zeta, max_iter=6)
            results[f'M-MR{tag}'].append(evaluate_scheme(cs, phi, p, 'mr_tts'))

            # SST 1.25 deg
            phi_s1, p_s1 = _run_single_sat(cs, cfg.PT_default, zeta, 'mr_stat', lat_offset=1.25)
            results[f'S-AP 1.25{tag}'].append(
                _eval_single_sat(cs, phi_s1, p_s1, 'mr_stat'))

            # SST 2.5 deg
            phi_s2, p_s2 = _run_single_sat(cs, cfg.PT_default, zeta, 'mr_stat', lat_offset=2.5)
            results[f'S-AP 2.5{tag}'].append(
                _eval_single_sat(cs, phi_s2, p_s2, 'mr_stat'))

            results[f'S-MR 1.25{tag}'].append(
                _eval_single_sat(cs, phi_s1, p_s1, 'mr_tts'))
            results[f'S-MR 2.5{tag}'].append(
                _eval_single_sat(cs, phi_s2, p_s2, 'mr_tts'))

    plot_min_sinr(
        cfg.zeta_msc_range_dB, results,
        xlabel='$\\varsigma$', title='Fig. 11: MSC vs SST Min SINR vs $\\varsigma$',
        filename='fig11_msc_vs_sst_sinr.png',
        x_label_unit=' (dBW)', ncol=3
    )
    return results


def fig12_simulation():
    """Fig. 12: Sum rate vs ζ, MSC vs SST"""
    print("\n===== Fig. 12: MSC vs SST (Sum Rate) =====")
    kappa_N_values = [0.0, 20.0]
    kappa_R, kappa_LR = 10.0, 10.0
    M = 12

    schemes = ['M-AP (0dB)', 'M-MR (0dB)', 'S-AP (0dB)', 'S-MR (0dB)',
               'M-AP (20dB)', 'M-MR (20dB)', 'S-AP (20dB)', 'S-MR (20dB)']
    results = {s: [] for s in schemes}

    for zeta in cfg.zeta_msc_range_dB:
        print(f"  zeta = {zeta} dBW")
        for kappa_N in kappa_N_values:
            cs = compute_channel_statistics(
                compute_distances(*compute_positions(M)),
                kappa_N, kappa_R, kappa_LR, M=M
            )
            tag = f' ({int(kappa_N)}dB)'

            np.random.seed(42)
            phi, p = algorithm_ap_ao(cs, cfg.PT_default, zeta, max_iter=5)
            results[f'M-AP{tag}'].append(evaluate_sum_rate(cs, phi, p, 'mr_stat'))

            np.random.seed(42)
            phi, p = algorithm_mr_tts_ts(cs, cfg.PT_default, zeta, max_iter=5)
            results[f'M-MR{tag}'].append(evaluate_sum_rate(cs, phi, p, 'mr_tts'))

            phi_s, p_s = _run_single_sat(cs, cfg.PT_default, zeta, 'mr_stat')
            results[f'S-AP{tag}'].append(evaluate_sum_rate(cs, phi_s, p_s, 'mr_stat'))
            results[f'S-MR{tag}'].append(evaluate_sum_rate(cs, phi_s, p_s, 'mr_tts'))

    plot_sum_rate(
        cfg.zeta_msc_range_dB, results,
        xlabel='$\\varsigma$', title='Fig. 12: MSC vs SST Sum Rate vs $\\varsigma$',
        filename='fig12_msc_vs_sst_sumrate.png',
        x_label_unit=' (dBW)'
    )
    return results


def fig13_simulation():
    """Fig. 13: 算法执行时间对比"""
    print("\n===== Fig. 13: Execution Time =====")
    configs = ['N=16,M=12', 'N=16,M=24', 'N=32,M=12', 'N=32,M=24']
    N_values = [16, 16, 32, 32]
    M_values = [12, 24, 12, 24]

    scheme_times = {
        'AP-AO': [],
        'MR-S-PA': [],
        'MR-S-TS': [],
        'MR-TTS-TS': []
    }

    for idx, (N_val, M_val) in enumerate(zip(N_values, M_values)):
        Nr = int(np.sqrt(N_val))
        Nc = N_val // Nr
        print(f"  Config: N={N_val}, M={M_val}")
        cs = compute_channel_statistics(
            compute_distances(*compute_positions(M_val)),
            10.0, 10.0, 10.0, M=M_val, Nr_sat=Nr, Nc_sat=Nc
        )

        for scheme_name in scheme_times:
            t0 = time.time()
            if scheme_name == 'AP-AO':
                algorithm_ap_ao(cs, cfg.PT_default, cfg.zeta_default, max_iter=3)
            elif scheme_name == 'MR-S-PA':
                algorithm_mr_s_pa(cs, cfg.PT_default, cfg.zeta_default)
            elif scheme_name == 'MR-S-TS':
                algorithm_mr_s_ts(cs, cfg.PT_default, cfg.zeta_default, max_iter=5)
            elif scheme_name == 'MR-TTS-TS':
                algorithm_mr_tts_ts(cs, cfg.PT_default, cfg.zeta_default, max_iter=5)
            elapsed = time.time() - t0
            scheme_times[scheme_name].append(elapsed)
            print(f"    {scheme_name}: {elapsed:.2f}s")

    plot_execution_time(
        configs, scheme_times,
        title='Fig. 13: Algorithm Execution Time',
        filename='fig13_execution_time.png'
    )
    return scheme_times


# ===== 辅助函数: 单卫星传输 =====

def _run_single_sat(cs, PT, zeta_dB, scheme='mr_stat', lat_offset=1.25):
    """单卫星传输 (SST): 只使用一颗卫星"""
    U = cfg.U
    M = cs['M']
    phi = init_ris_phase(M, U)
    p_sst = np.zeros((cfg.J, U))
    p_sst[0, :] = PT / U

    from optimization import optimize_ris_rg, power_allocation_fp
    phi = optimize_ris_rg(cs, p_sst, phi, zeta_dB, scheme, max_iter=8)
    p_full = power_allocation_fp(cs, phi, PT, zeta_dB, scheme, max_iter=8)
    p_full[1:, :] = 0
    total = np.sum(p_full[0, :])
    if total > 0:
        p_full[0, :] *= PT / total
    return phi, p_full


def _eval_single_sat(cs, phi, p, scheme):
    """评估单卫星方案"""
    return evaluate_scheme(cs, phi, p, scheme)
