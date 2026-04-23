"""
一键复现脚本
运行所有仿真并生成图表
"""
import numpy as np
import sys
import os

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from simulation import simulate_fig2, simulate_fig5, simulate_fig8
from plotting import plot_fig2, plot_fig5, plot_fig8

# 确保输出目录存在
os.makedirs('output', exist_ok=True)


def main():
    print("=" * 60)
    print("论文复现: Beamforming Design and Satellite Selection")
    print("         for Realizing the ICAN in LEO Satellite Networks")
    print("期刊: IEEE TWC, 2026")
    print("=" * 60)
    print()

    # ===== Fig. 2: 波束赋形对比 =====
    print("[1/3] Fig. 2: Rate performance under different beamforming schemes")
    results_fig2 = simulate_fig2()
    plot_fig2(cfg.C_range, results_fig2, output_dir='output')

    print()

    # ===== Fig. 5: 卫星选择对比 =====
    print("[2/3] Fig. 5: Rate and GDOP under different satellite selection schemes")
    results_fig5 = simulate_fig5()
    plot_fig5(cfg.S_range, results_fig5, output_dir='output')

    print()

    # ===== Fig. 8: 通导权衡 =====
    print("[3/3] Fig. 8: Trade-off between communication rate and navigation GDOP")
    results_fig8 = simulate_fig8()
    plot_fig8(cfg.rho_range, results_fig8, output_dir='output')

    print()
    print("=" * 60)
    print("复现完成！所有图表已保存到 output/ 目录")
    print("=" * 60)

    # 打印结果汇总
    print("\n===== Fig. 2 结果汇总 =====")
    print(f"{'UE数':>6s} | {'DC':>8s} | {'MRT':>8s} | {'ZF':>8s} | {'MMSE':>8s} | {'ST-ZF':>8s} | (Mbps)")
    for C in cfg.C_range:
        rates = [results_fig2[C][s]/1e6 for s in ['Proposed DC', 'MRT', 'ZF', 'MMSE', 'ST-ZF']]
        print(f"{C:>6d} | " + " | ".join(f"{r:>8.2f}" for r in rates))

    print("\n===== Fig. 5 结果汇总 (S=12) =====")
    if 12 in results_fig5:
        for scheme in ['Proposed (ρ=1)', 'Proposed (ρ=0.5)', 'Proposed (ρ=0)',
                       'Comm-oriented', 'Nav-oriented', 'Heuristic ICAN', 'Coalitional ICAN']:
            r = results_fig5[12][scheme]
            print(f"  {scheme:>25s}: Rate={r['rate']/1e6:.2f} Mbps, GDOP={r['gdop']:.2f}")


if __name__ == '__main__':
    main()
