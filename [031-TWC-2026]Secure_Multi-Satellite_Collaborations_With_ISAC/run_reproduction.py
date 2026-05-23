"""
一键复现脚本 — Fig. 3, Fig. 6, Fig. 9
论文: Secure Multi-Satellite Collaborations With ISAC (IEEE TWC, 2026)
"""
import os
import sys
import time
import numpy as np

# 确保在项目目录下运行
project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import config as cfg
from simulation import simulate_fig3, simulate_fig6, simulate_fig9
from plotting import plot_fig3, plot_fig6, plot_fig9

OUTPUT_DIR = os.path.join(project_dir, 'output')


def main():
    print('=' * 60)
    print('ISAC-MSC 论文复现')
    print('Secure Multi-Satellite Collaborations With ISAC')
    print('IEEE TWC, 2026')
    print('=' * 60)

    n_mc = 200  # 蒙特卡洛次数
    print(f'\n蒙特卡洛仿真次数: {n_mc}')
    print(f'输出目录: {OUTPUT_DIR}\n')

    # ===== Fig. 3: 感知 SNR vs P_m =====
    print('[1/3] Fig. 3: Sensing SNR vs. Power Budget P_m')
    print(f'  P_m 范围: {cfg.Pm_dBW_range} dBW')
    print(f'  M0 = {cfg.M0}')
    t0 = time.time()
    fig3_data = simulate_fig3(n_mc=n_mc)
    fig3_path = plot_fig3(fig3_data, OUTPUT_DIR)
    print(f'  耗时: {time.time() - t0:.1f}s\n')

    # ===== Fig. 6: 感知 SNR vs M_0 =====
    print('[2/3] Fig. 6: Sensing SNR vs. Cooperative Satellites M_0')
    print(f'  M0 范围: {cfg.M0_range}')
    print(f'  P_m = 25 dBW')
    t0 = time.time()
    fig6_data = simulate_fig6(n_mc=n_mc, P_dBW=25)
    fig6_path = plot_fig6(fig6_data, OUTPUT_DIR)
    print(f'  耗时: {time.time() - t0:.1f}s\n')

    # ===== Fig. 9: CRB vs P_m (不同 M_0) =====
    print('[3/3] Fig. 9: CRB vs. Power Budget (DP-JSC-BF)')
    print(f'  P_m 范围: {cfg.Pm_dBW_range} dBW')
    print(f'  M0 值: [1, 3, 5, 7, 9]')
    t0 = time.time()
    fig9_data = simulate_fig9(n_mc=n_mc, M0_values=[1, 3, 5, 7, 9])
    fig9_path = plot_fig9(fig9_data, OUTPUT_DIR)
    print(f'  耗时: {time.time() - t0:.1f}s\n')

    # ===== 结果验证 =====
    print('=' * 60)
    print('复现结果验证')
    print('=' * 60)

    # Fig. 3 数值检查
    print('\nFig. 3 关键数值 (P_m=25 dBW):')
    for label, (Pm, snr, _) in sorted(fig3_data.items()):
        idx_25 = np.argmin(np.abs(Pm - 25))
        print(f'  {label:15s}: SNR = {snr[idx_25]:.1f} dB')

    # Fig. 6 数值检查
    print('\nFig. 6 关键数值 (M_0=6):')
    for label, (M0_arr, snr, _) in sorted(fig6_data.items()):
        idx_6 = np.argmin(np.abs(M0_arr - 6))
        print(f'  {label:15s}: SNR = {snr[idx_6]:.1f} dB')

    # Fig. 9 数值检查
    print('\nFig. 9 关键数值 (P_m=25 dBW):')
    for key, (Pm, crb, _, M0) in sorted(fig9_data.items()):
        idx_25 = np.argmin(np.abs(Pm - 25))
        print(f'  M0={M0}: CRB = {crb[idx_25]:.2f} m')

    # 趋势检查
    print('\n趋势检查:')
    dp_jsc = fig3_data.get('DP-JSC-BF', (None, None, None))[1]
    shp_pa = fig3_data.get('SHP-PA', (None, None, None))[1]
    if dp_jsc is not None and shp_pa is not None:
        gap = dp_jsc[-1] - shp_pa[-1]
        print(f'  DP-JSC-BF vs SHP-PA (P_m=30): gap = {gap:.1f} dB (预期 > 0)')

    print('\n' + '=' * 60)
    print('复现完成!')
    print(f'输出文件:')
    print(f'  {fig3_path}')
    print(f'  {fig6_path}')
    print(f'  {fig9_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
