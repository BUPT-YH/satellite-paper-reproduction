"""
一键复现脚本 — 精确 FCC 参数版
论文: Satellite Selection for In-Band Coexistence of Dense LEO Networks
期刊: IEEE TWC, 2026

复现目标:
  Fig. 4 — 基线 INR CDF
  Fig. 5 — 提出方案 INR CDF

用法:
  python run_reproduction.py
"""

import os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from simulation import (
    run_baseline_simulation,
    run_proposed_simulation,
)
from plotting import plot_fig4, plot_fig5

OUTPUT = 'output'
os.makedirs(OUTPUT, exist_ok=True)


def main():
    print("=" * 70)
    print("论文复现: Satellite Selection for In-Band Coexistence of Dense LEO Networks")
    print("期刊: IEEE TWC, 2026  |  精确 FCC 轨道参数 (6+3 壳层)")
    print("=" * 70)

    print("\n[1/2] Fig. 4: 基线 INR CDF...")
    t0 = time.time()
    baseline = run_baseline_simulation(duration_sec=30, time_step=2.0)
    f4 = plot_fig4(baseline, OUTPUT)
    print(f"  耗时 {time.time()-t0:.0f}s\n")

    print("[2/2] Fig. 5: 提出方案 INR CDF...")
    t0 = time.time()
    proposed = run_proposed_simulation(duration_sec=30, time_step=2.0,
                                       inr_th_db=-6.0,
                                       inr_max_th_list=[-6.0, 0.0, 3.0, float('inf')])
    f5 = plot_fig5(proposed, OUTPUT)
    print(f"  耗时 {time.time()-t0:.0f}s\n")

    print("=" * 70)
    print("复现完成!")
    print(f"  Fig. 4: {f4}")
    print(f"  Fig. 5: {f5}")
    print("=" * 70)


if __name__ == '__main__':
    main()
