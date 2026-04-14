"""
一键复现脚本 — 论文: LLM-Aided Spectrum-Sharing LEO Satellite Communications
运行: python run_reproduction.py
"""

import numpy as np
import sys
import os
import time

# 确保在项目目录运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from config import (RE, DSD_DEFAULT, ALPHA, LAMBDA_E_FS, PE_OVER_PS)
from channel_model import verify_channel_model
from outage_probability import (pout1_analytical, pout1_analytical_array,
                                pout1_montecarlo,
                                pout2_analytical,
                                pout2_montecarlo_batch)
from resource_allocation import (SequentialScheme, HybridScheme, LLMScheme,
                                 compute_transmitted_data_over_time,
                                 compute_delay_summary)
import plotting

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(42)


def run_fig8():
    """Fig. 8: Pout,1 vs PS, dSD = 600/800/1000 km"""
    print("\n" + "="*60)
    print("Fig. 8: Pout,1 (频谱共享) vs PS, 不同 dSD")
    print("="*60)

    ps_range = np.arange(5, 31, 2.5)
    dSD_list = [600e3, 800e3, 1000e3]
    gamma_th_dB = 10  # 固定 γth

    pout_data = {}
    for dSD in dSD_list:
        dSD_km = int(dSD / 1e3)
        print(f"  dSD = {dSD_km} km:")

        # 解析曲线
        pout_ana = pout1_analytical_array(ps_range, gamma_th_dB, dSD)
        print(f"    Analytical: [{np.min(pout_ana):.2e}, {np.max(pout_ana):.2e}]")

        # 蒙特卡洛
        pout_sim = np.array([pout1_montecarlo(ps, gamma_th_dB, dSD,
                                               n_trials=200000, rng=rng)
                             for ps in ps_range])
        print(f"    Simulation: [{np.nanmin(pout_sim):.2e}, {np.nanmax(pout_sim):.2e}]")

        pout_data[dSD_km] = {'analytical': pout_ana, 'sim': pout_sim}

    plotting.plot_fig8(ps_range, pout_data, OUTPUT_DIR)
    return pout_data


def run_fig9():
    """Fig. 9: Pout,2 vs γth, dSD = 600/800/1000 km"""
    print("\n" + "="*60)
    print("Fig. 9: Pout,2 (固定频段) vs γth, 不同 dSD")
    print("="*60)

    gamma_range = np.arange(-5, 21, 2)  # 稀疏网格加速
    dSD_list = [600e3, 800e3, 1000e3]
    PS_dB = 20

    pout_data = {}
    for dSD in dSD_list:
        dSD_km = int(dSD / 1e3)
        print(f"  dSD = {dSD_km} km:")

        # 解析
        pout_ana = np.array([pout2_analytical(g, PS_dB, dSD,
                                               PE_over_PS=PE_OVER_PS)
                              for g in gamma_range])
        print(f"    Analytical: [{np.min(pout_ana):.2e}, {np.max(pout_ana):.2e}]")

        # 蒙特卡洛
        pout_sim = pout2_montecarlo_batch(gamma_range, PS_dB, dSD,
                                           PE_over_PS=PE_OVER_PS,
                                           n_trials=15000, rng=rng)
        print(f"    Simulation: [{np.nanmin(pout_sim):.2e}, {np.nanmax(pout_sim):.2e}]")

        pout_data[dSD_km] = {'analytical': pout_ana, 'sim': pout_sim}

    plotting.plot_fig9(gamma_range, pout_data, OUTPUT_DIR)
    return pout_data


def run_fig10():
    """Fig. 10: Pout,2 vs γth, 不同 λe·Δfs"""
    print("\n" + "="*60)
    print("Fig. 10: Pout,2 vs γth, 不同 λe·Δfs")
    print("="*60)

    gamma_range = np.arange(-5, 21, 2)
    lambda_list = [3, 5, 7]
    PS_dB = 20
    dSD = 800e3

    pout_data = {}
    for lam in lambda_list:
        print(f"  λe·Δfs = {lam}:")

        # 解析
        pout_ana = np.array([pout2_analytical(g, PS_dB, dSD,
                                               lambda_e_fs=lam,
                                               PE_over_PS=PE_OVER_PS)
                              for g in gamma_range])
        print(f"    Analytical: [{np.min(pout_ana):.2e}, {np.max(pout_ana):.2e}]")

        # 蒙特卡洛
        pout_sim = pout2_montecarlo_batch(gamma_range, PS_dB, dSD,
                                           lambda_e_fs=lam,
                                           PE_over_PS=PE_OVER_PS,
                                           n_trials=10000, rng=rng)
        print(f"    Simulation: [{np.nanmin(pout_sim):.2e}, {np.nanmax(pout_sim):.2e}]")

        pout_data[lam] = {'analytical': pout_ana, 'sim': pout_sim}

    plotting.plot_fig10(gamma_range, pout_data, OUTPUT_DIR)
    return pout_data


def run_fig11():
    """Fig. 11: LLM (Pout,1) vs Fixed-band (Pout,2), PS = 10/20/30 dB"""
    print("\n" + "="*60)
    print("Fig. 11: LLM决策 vs 固定频段 OP 对比")
    print("="*60)

    gamma_range = np.arange(-5, 21, 2)
    PS_list = [10, 20, 30]
    dSD = 800e3

    pout_data = {}
    for ps in PS_list:
        print(f"  PS = {ps} dBW:")

        # LLM (频谱共享): Pout,1
        pout_llm = np.array([pout1_analytical(ps, g, dSD) for g in gamma_range])
        print(f"    LLM:    [{np.min(pout_llm):.2e}, {np.max(pout_llm):.2e}]")

        # Fixed (固定频段): Pout,2
        pout_fixed = np.array([pout2_analytical(g, ps, dSD,
                                                  PE_over_PS=PE_OVER_PS)
                                for g in gamma_range])
        print(f"    Fixed:  [{np.min(pout_fixed):.2e}, {np.max(pout_fixed):.2e}]")

        pout_data[ps] = {'llm': pout_llm, 'fixed': pout_fixed}

    plotting.plot_fig11(gamma_range, pout_data, OUTPUT_DIR)
    return pout_data


def run_fig12():
    """Fig. 12: 传输数据量 vs 时间"""
    print("\n" + "="*60)
    print("Fig. 12: 三种方案传输数据量对比")
    print("="*60)

    schemes = {
        'Sequential': SequentialScheme(),
        'Hybrid': HybridScheme(),
        'LLM Decision': LLMScheme(),
    }

    max_time = max(s.compute()['total_time'] for s in schemes.values())
    t_seq = np.linspace(0, max_time * 1.05, 500)

    data_schemes = {}
    for name, scheme in schemes.items():
        data = compute_transmitted_data_over_time(scheme, t_seq)
        data_schemes[name] = data
        total = scheme.compute()
        print(f"  {name}: 总时间={total['total_time']:.2f}s")

    plotting.plot_fig12(t_seq, data_schemes, OUTPUT_DIR)
    return data_schemes


def run_fig13():
    """Fig. 13: 传输等待时延堆叠柱状图"""
    print("\n" + "="*60)
    print("Fig. 13: 三种方案传输等待时延对比")
    print("="*60)

    delay_summary = compute_delay_summary()

    delay_data = {}
    for name in ['Sequential', 'Hybrid', 'LLM Decision']:
        result = delay_summary[name]
        delay_data[name] = {
            'wait': result['wait_time'],
            'trans': result['trans_time'],
        }
        total_delay = sum(result['wait_time'][s] + result['trans_time'][s]
                         for s in ['A', 'B', 'C', 'D'])
        print(f"  {name}: 总时延 = {total_delay:.2f}s")
        for s in ['A', 'B', 'C', 'D']:
            print(f"    {s}: wait={result['wait_time'][s]:.4f}s, "
                  f"trans={result['trans_time'][s]:.4f}s")

    plotting.plot_fig13(delay_data, OUTPUT_DIR)
    return delay_data


# ==================== 主函数 ====================

if __name__ == "__main__":
    print("="*60)
    print("Paper Reproduction: LLM-Aided Spectrum-Sharing")
    print("LEO Satellite Communications")
    print("IEEE JSAC, Vol. 44, 2026")
    print("="*60)

    # 验证信道模型
    print("\n--- Channel Model Verification ---")
    verify_channel_model()

    t_start = time.time()

    # 快速模块: Fig. 12, 13 (纯计算，无需大量仿真)
    run_fig12()
    run_fig13()

    # Fig. 8: Pout,1 蒙特卡洛
    run_fig8()

    # Fig. 9, 10, 11: Pout,2 解析 + 蒙特卡洛 (耗时较长)
    print("\nNote: Fig. 9/10/11 involve Gaussian-Chebyshev quadrature...")
    run_fig9()
    run_fig10()
    run_fig11()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"All figures generated! Total time: {elapsed:.1f}s")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'='*60}")
