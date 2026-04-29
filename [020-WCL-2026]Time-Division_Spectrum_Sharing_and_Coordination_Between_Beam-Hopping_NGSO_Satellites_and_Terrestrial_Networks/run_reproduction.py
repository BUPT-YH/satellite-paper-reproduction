"""
一键复现脚本
论文: Time-Division Spectrum Sharing and Coordination Between
      Beam-Hopping NGSO Satellites and Terrestrial Networks
期刊: IEEE WCL, 2026
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import USER_TOTAL_RANGE, TIMESLOT_LEN_RANGE
from bhss_core import generate_scenario
from simulation import simulate_fig5, compute_time_sync_efficiency
from plotting import plot_fig5, plot_fig6


def main():
    print('='*60)
    print('BHSS 论文复现')
    print('IEEE Wireless Communications Letters, 2026')
    print('='*60)

    print('\n[1/4] 初始化仿真场景...')
    scenario = generate_scenario()
    n_dist_terr = len(scenario['disturbed_terr_idx'])
    n_dist_sat = len(scenario['disturbed_sat_idx'])
    print(f'  地面小区: {60}, 受干扰: {n_dist_terr}')
    print(f'  卫星小区: {128}, 受干扰: {n_dist_sat}')

    print('\n[2/4] 仿真 Fig.5 — 服务容量 vs 用户数...')
    terr_results, sat_results = simulate_fig5(scenario)

    print('\n  地面小区平均服务容量 (Mbps):')
    print(f'  {"Users":>8}', end='')
    for s in ['Interfered', 'Fixed Freq Div', 'DSS', 'BHSS']:
        print(f'  {s:>16}', end='')
    print()
    for i, u in enumerate(USER_TOTAL_RANGE):
        print(f'  {u:>8}', end='')
        for s in ['Interfered', 'Fixed Freq Div', 'DSS', 'BHSS']:
            print(f'  {terr_results[s][i]:>16.2f}', end='')
        print()

    fig5_path = plot_fig5(terr_results, sat_results, USER_TOTAL_RANGE)

    print('\n[3/4] 仿真 Fig.6 — 时间同步效率 vs 时隙长度...')
    terr_eff, sat_eff = compute_time_sync_efficiency(scenario)

    T_range_ms = TIMESLOT_LEN_RANGE * 1000
    print('\n  地面小区时间同步效率 (%):')
    print(f'  {"T(ms)":>8}', end='')
    for m in ['Ideal', 'Proposed', 'Timeslot-based', 'General Sync', 'Terr-prior']:
        print(f'  {m:>14}', end='')
    print()
    for i, t in enumerate(T_range_ms):
        print(f'  {t:>8.1f}', end='')
        for m in ['Ideal', 'Proposed', 'Timeslot-based', 'General Sync', 'Terr-prior']:
            print(f'  {terr_eff[m][i]*100:>14.1f}', end='')
        print()

    fig6_path = plot_fig6(terr_eff, sat_eff, T_range_ms)

    print(f'\n[4/4] 完成! 输出:')
    print(f'  {fig5_path}')
    print(f'  {fig6_path}')


if __name__ == '__main__':
    main()
