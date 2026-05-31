"""
run_reproduction.py — 一键复现脚本
论文: Co-Existence Analysis of Terrestrial and Non-Terrestrial Networks
      in S-Band Using Stochastic Geometry (IEEE TCOM 2026)

复现目标:
  - 图5: Case I Coverage Probability vs SINR Threshold (城市/农村 × 三种卫星高度 × 100%/25%负载)
  - 图8: Case II Coverage Probability vs SINR Threshold (不同N_u和r_iso组合)
  - 图11: Case I vs Case II 对比
"""

import os
import sys
import time
import numpy as np

# 添加项目目录到路径
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from config import (
    ALTITUDES, T_DB_MIN, T_DB_MAX, T_DB_POINTS,
    R_TN_URBAN, R_TN_RURAL, DISD_URBAN, DISD_RURAL,
    N_U_VALUES, LOAD_FULL, LOAD_PARTIAL, N_C,
    X_0_CENTER
)
from stochastic_geometry import (
    coverage_probability_case1,
    coverage_probability_case2,
    coverage_probability_no_ntn,
    mc_coverage_case1,
)
import plotting as pl
import matplotlib.pyplot as plt


# 输出目录
OUTPUT_DIR = os.path.join(project_dir, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SINR阈值网格 (dB)
T_db_grid = np.linspace(T_DB_MIN, T_DB_MAX, T_DB_POINTS)


def reproduce_figure5():
    """
    复现图5: Case I Coverage Probability vs SINR Threshold
    (a) 100% load, (b) 25% load
    """
    print("=" * 60)
    print("复现图5: Case I 覆盖概率 vs SINR阈值")
    print("=" * 60)

    scenarios = [
        ('urban', 'Urban', R_TN_URBAN),
        ('rural', 'Rural', R_TN_RURAL),
    ]

    loads = [
        (LOAD_FULL, '100% Load'),
        (LOAD_PARTIAL, '25% Load'),
    ]

    for load_factor, load_label in loads:
        fig, axes = plt.subplots(1, 2, figsize=pl.FIG_DOUBLE)

        for idx, (scenario, scenario_label, r_TN) in enumerate(scenarios):
            ax = axes[idx]

            # 无NTN干扰基线
            print(f"\n  [{scenario_label}] 无NTN干扰基线 ({load_label})...")
            t_start = time.time()
            baseline = []
            for T_db in T_db_grid:
                pc = coverage_probability_no_ntn(T_db, r_TN, X_0_CENTER, load_factor)
                baseline.append(pc)
            baseline = np.array(baseline)
            print(f"    耗时: {time.time() - t_start:.1f}s")

            ax.plot(T_db_grid, baseline, 'k--', linewidth=1.2, label='No NTN (baseline)')

            # 三种卫星高度
            for i, alt in enumerate(ALTITUDES):
                print(f"  [{scenario_label}] 卫星高度 {alt} km ({load_label})...")
                t_start = time.time()
                pc_vals = []
                for T_db in T_db_grid:
                    pc = coverage_probability_case1(T_db, alt, r_TN, X_0_CENTER, load_factor)
                    pc_vals.append(pc)
                pc_vals = np.array(pc_vals)
                print(f"    耗时: {time.time() - t_start:.1f}s, "
                      f"Pc范围: [{pc_vals.min():.4f}, {pc_vals.max():.4f}]")

                ax.plot(T_db_grid, pc_vals,
                        color=pl.COLORS[i],
                        marker=pl.MARKERS[i],
                        markersize=3,
                        markevery=10,
                        linestyle='-',
                        linewidth=1.2,
                        label=f'$h_{{sat}}$ = {alt} km')

            pl.setup_axis(ax, 'SINR Threshold $T$ (dB)',
                          'Coverage Probability $P_c(T)$',
                          title=f'{scenario_label} - {load_label}')
            ax.set_ylim([-0.02, 1.02])
            ax.legend(loc='upper right', fontsize=7)

        fig.tight_layout()
        fname = f'fig5_case1_coverage_{"full" if load_factor == 1.0 else "25pct"}load.png'
        pl.save_fig(fig, os.path.join(OUTPUT_DIR, fname))
        plt.close(fig)

    print("\n图5复现完成!")


def reproduce_figure8():
    """
    复现图8: Case II Coverage Probability vs SINR Threshold
    (a) 100% load, (b) 25% load
    """
    print("\n" + "=" * 60)
    print("复现图8: Case II 覆盖概率 vs SINR阈值")
    print("=" * 60)

    scenarios = [
        ('urban', 'Urban', R_TN_URBAN, DISD_URBAN),
        ('rural', 'Rural', R_TN_RURAL, DISD_RURAL),
    ]

    loads = [
        (LOAD_FULL, '100% Load'),
        (LOAD_PARTIAL, '25% Load'),
    ]

    # Case II参数组合 (论文图8: Nu={100,1000,2000} x r_iso={0, 2*dISD})
    case2_configs = [
        {'n_u': 100, 'r_iso_factor': 0, 'label': '$N_u$=100, $r_{iso}$=0'},
        {'n_u': 1000, 'r_iso_factor': 0, 'label': '$N_u$=1000, $r_{iso}$=0'},
        {'n_u': 2000, 'r_iso_factor': 0, 'label': '$N_u$=2000, $r_{iso}$=0'},
        {'n_u': 100, 'r_iso_factor': 2, 'label': '$N_u$=100, $r_{iso}$=2$d_{ISD}$'},
        {'n_u': 1000, 'r_iso_factor': 2, 'label': '$N_u$=1000, $r_{iso}$=2$d_{ISD}$'},
        {'n_u': 2000, 'r_iso_factor': 2, 'label': '$N_u$=2000, $r_{iso}$=2$d_{ISD}$'},
    ]

    altitude = 600  # 图8使用600km

    for load_factor, load_label in loads:
        fig, axes = plt.subplots(1, 2, figsize=pl.FIG_DOUBLE)

        for idx, (scenario, scenario_label, r_TN, disd) in enumerate(scenarios):
            ax = axes[idx]

            # 无NTN干扰基线
            print(f"\n  [{scenario_label}] 无NTN干扰基线 ({load_label})...")
            baseline = []
            for T_db in T_db_grid:
                pc = coverage_probability_no_ntn(T_db, r_TN, X_0_CENTER, load_factor)
                baseline.append(pc)
            ax.plot(T_db_grid, baseline, 'k--', linewidth=1.2, label='No NTN')

            # Case II 各参数组合
            for i, cfg in enumerate(case2_configs):
                r_iso = cfg['r_iso_factor'] * disd
                label = cfg['label']
                print(f"  [{scenario_label}] {label} ({load_label})...")
                t_start = time.time()

                pc_vals = []
                for T_db in T_db_grid:
                    pc = coverage_probability_case2(T_db, altitude, r_TN, r_iso,
                                                     X_0_CENTER, cfg['n_u'], load_factor)
                    pc_vals.append(pc)
                pc_vals = np.array(pc_vals)
                print(f"    耗时: {time.time() - t_start:.1f}s, "
                      f"Pc范围: [{pc_vals.min():.4f}, {pc_vals.max():.4f}]")

                ax.plot(T_db_grid, pc_vals,
                        color=pl.COLORS[i],
                        marker=pl.MARKERS[i],
                        markersize=3,
                        markevery=10,
                        linestyle=pl.LINESTYLES[i],
                        linewidth=1.2,
                        label=label)

            pl.setup_axis(ax, 'SINR Threshold $T$ (dB)',
                          'Coverage Probability $P_c(T)$',
                          title=f'{scenario_label} - {load_label}')
            ax.set_ylim([-0.02, 1.02])
            ax.legend(loc='upper right', fontsize=6.5)

        fig.tight_layout()
        fname = f'fig8_case2_coverage_{"full" if load_factor == 1.0 else "25pct"}load.png'
        pl.save_fig(fig, os.path.join(OUTPUT_DIR, fname))
        plt.close(fig)

    print("\n图8复现完成!")


def reproduce_figure11():
    """
    复现图11: Case I vs Case II 对比 — 单图，8条曲线
    4条Case II曲线 (600km, Urban场景) + 4条Case I曲线 (不同高度和场景)
    """
    print("\n" + "=" * 60)
    print("复现图11: Case I vs Case II 对比 (单图)")
    print("=" * 60)

    # 图11曲线配置: 单图中的8条曲线
    # Case II使用Urban场景参数, Case I各自指定场景
    fig11_configs = [
        # Case II curves (600km, Urban)
        {'case': 2, 'alt': 600, 'n_u': 100, 'r_iso_factor': 0, 'scenario': 'urban',
         'label': '$N_u$=100, $r_{iso}$=0, Case II', 'color_idx': 0, 'style_idx': 0},
        {'case': 2, 'alt': 600, 'n_u': 100, 'r_iso_factor': 2, 'scenario': 'urban',
         'label': '$N_u$=100, $r_{iso}$=2$d_{ISD}$, Case II', 'color_idx': 1, 'style_idx': 1},
        {'case': 2, 'alt': 600, 'n_u': 2000, 'r_iso_factor': 0, 'scenario': 'urban',
         'label': '$N_u$=2000, $r_{iso}$=0, Case II', 'color_idx': 2, 'style_idx': 2},
        {'case': 2, 'alt': 600, 'n_u': 2000, 'r_iso_factor': 2, 'scenario': 'urban',
         'label': '$N_u$=2000, $r_{iso}$=2$d_{ISD}$, Case II', 'color_idx': 3, 'style_idx': 3},
        # Case I curves
        {'case': 1, 'alt': 1200, 'n_u': None, 'r_iso_factor': 0, 'scenario': 'urban',
         'label': '$a$=1200km, Case I, Urban', 'color_idx': 4, 'style_idx': 4},
        {'case': 1, 'alt': 200, 'n_u': None, 'r_iso_factor': 0, 'scenario': 'urban',
         'label': '$a$=200km, Case I, Urban', 'color_idx': 5, 'style_idx': 0},
        {'case': 1, 'alt': 1200, 'n_u': None, 'r_iso_factor': 0, 'scenario': 'rural',
         'label': '$a$=1200km, Case I, Rural', 'color_idx': 6, 'style_idx': 1},
        {'case': 1, 'alt': 200, 'n_u': None, 'r_iso_factor': 0, 'scenario': 'rural',
         'label': '$a$=200km, Case I, Rural', 'color_idx': 0, 'style_idx': 2},
    ]

    scenario_params = {
        'urban': {'r_TN': R_TN_URBAN, 'disd': DISD_URBAN, 'label': 'Urban'},
        'rural': {'r_TN': R_TN_RURAL, 'disd': DISD_RURAL, 'label': 'Rural'},
    }

    load_factor = LOAD_FULL

    fig, ax = plt.subplots(1, 1, figsize=pl.FIG_SINGLE)

    for cfg in fig11_configs:
        sp = scenario_params[cfg['scenario']]
        r_TN = sp['r_TN']
        disd = sp['disd']
        label = cfg['label']
        print(f"  {label} ({cfg['scenario']})...")
        t_start = time.time()

        pc_vals = []
        for T_db in T_db_grid:
            if cfg['case'] == 1:
                pc = coverage_probability_case1(T_db, cfg['alt'], r_TN,
                                                 X_0_CENTER, load_factor)
            else:
                r_iso = cfg.get('r_iso_factor', 0) * disd
                pc = coverage_probability_case2(T_db, cfg['alt'], r_TN, r_iso,
                                                 X_0_CENTER, cfg['n_u'], load_factor)
            pc_vals.append(pc)
        pc_vals = np.array(pc_vals)
        print(f"    耗时: {time.time() - t_start:.1f}s, "
              f"Pc范围: [{pc_vals.min():.4f}, {pc_vals.max():.4f}]")

        ax.plot(T_db_grid, pc_vals,
                color=pl.COLORS[cfg['color_idx']],
                marker=pl.MARKERS[cfg['style_idx']],
                markersize=3,
                markevery=10,
                linestyle=pl.LINESTYLES[cfg['style_idx']],
                linewidth=1.2,
                label=label)

    # No SS baseline (Urban)
    baseline = []
    for T_db in T_db_grid:
        pc = coverage_probability_no_ntn(T_db, R_TN_URBAN, X_0_CENTER, load_factor)
        baseline.append(pc)
    ax.plot(T_db_grid, baseline, 'k--', linewidth=1.2, label='No SS')

    pl.setup_axis(ax, 'SINR Threshold $T$ (dB)',
                  'Coverage Probability $P_c(T)$',
                  title='Case I vs Case II')
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc='upper right', fontsize=5.5)

    fig.tight_layout()
    pl.save_fig(fig, os.path.join(OUTPUT_DIR, 'fig11_case1_vs_case2.png'))
    plt.close(fig)

    print("\n图11复现完成!")


def run_mc_validation():
    """
    运行少量Monte Carlo仿真验证解析结果
    """
    print("\n" + "=" * 60)
    print("Monte Carlo 仿真验证 (少量验证点)")
    print("=" * 60)

    # 选取几个验证点
    test_points = [
        {'T_db': 0, 'alt': 600, 'r_TN': R_TN_URBAN, 'x_0': X_0_CENTER,
         'load': LOAD_FULL, 'label': 'Urban, 600km, T=0dB, 100%load'},
        {'T_db': 10, 'alt': 600, 'r_TN': R_TN_URBAN, 'x_0': X_0_CENTER,
         'load': LOAD_FULL, 'label': 'Urban, 600km, T=10dB, 100%load'},
        {'T_db': 0, 'alt': 600, 'r_TN': R_TN_RURAL, 'x_0': X_0_CENTER,
         'load': LOAD_FULL, 'label': 'Rural, 600km, T=0dB, 100%load'},
    ]

    for tp in test_points:
        print(f"\n  验证: {tp['label']}")

        # 解析结果
        pc_analytical = coverage_probability_case1(
            tp['T_db'], tp['alt'], tp['r_TN'], tp['x_0'], tp['load'])
        print(f"    解析: P_c = {pc_analytical:.4f}")

        # MC仿真 (减少次数以节省时间)
        pc_mc = mc_coverage_case1(
            tp['T_db'], tp['alt'], tp['r_TN'], tp['x_0'], tp['load'],
            n_trials=50000)
        print(f"    MC仿真: P_c = {pc_mc:.4f}")

        diff = abs(pc_analytical - pc_mc)
        print(f"    偏差: {diff:.4f}")


if __name__ == '__main__':
    print("=" * 60)
    print("论文复现: Co-Existence Analysis of TN and NTN in S-Band")
    print("IEEE Transactions on Communications, 2026")
    print("=" * 60)

    total_start = time.time()

    # 1. 复现图5
    reproduce_figure5()

    # 2. 复现图8
    reproduce_figure8()

    # 3. 复现图11
    reproduce_figure11()

    # 4. MC验证
    run_mc_validation()

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"全部复现完成! 总耗时: {total_time:.1f}s")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
