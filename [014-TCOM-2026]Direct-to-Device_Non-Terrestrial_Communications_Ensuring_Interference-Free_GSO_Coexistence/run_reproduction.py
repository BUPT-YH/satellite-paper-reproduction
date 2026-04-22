"""
论文复现 - 一键运行脚本
Reproduce Fig. 4 and Fig. 6 from:
  Direct-to-Device Non-Terrestrial Communications Ensuring Interference-Free GSO Coexistence
  IEEE TCOM 2026
"""
import os
import sys
import time

# 确保在正确目录下运行
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from simulation import simulate_fig4a, simulate_fig4b, simulate_fig6
from plotting import plot_fig4a, plot_fig4b, plot_fig6


def main():
    print("=" * 60)
    print("Paper Reproduction: D2D Non-Terrestrial Communications")
    print("IEEE TCOM 2026 - Jalali et al.")
    print("=" * 60)
    print()

    os.makedirs('output', exist_ok=True)

    total_start = time.time()

    # ========== Fig. 4(a): 禁区卫星数量 ==========
    print("\n[1/3] Simulating Fig. 4(a)...")
    t0 = time.time()
    times, num_fz, num_vis = simulate_fig4a(cone_angle=2.0)
    plot_fig4a(times, num_fz, 'output/fig4a_fz_satellites.png')
    print(f"  Time: {time.time() - t0:.1f}s")

    # ========== Fig. 4(b): 禁区百分比 vs 锥角 ==========
    print("\n[2/3] Simulating Fig. 4(b)...")
    t0 = time.time()
    cone_angles, pct_low, pct_high = simulate_fig4b()
    plot_fig4b(cone_angles, pct_low, pct_high, 'output/fig4b_fz_percentage.png')
    print(f"  Time: {time.time() - t0:.1f}s")

    # ========== Fig. 6: EPFD CCDF + SE CDF ==========
    print("\n[3/3] Simulating Fig. 6...")
    t0 = time.time()
    (epfd_range, ccdf_homn, ccdf_evmn, ccdf_lnmx,
     se_range, cdf_se_homn, cdf_se_evmn, cdf_se_lnmx) = simulate_fig6()
    plot_fig6(epfd_range, ccdf_homn, ccdf_evmn, ccdf_lnmx,
              se_range, cdf_se_homn, cdf_se_evmn, cdf_se_lnmx,
              'output/fig6_epfd_se.png')
    print(f"  Time: {time.time() - t0:.1f}s")

    total_time = time.time() - total_start
    print()
    print("=" * 60)
    print(f"All simulations completed in {total_time:.1f}s")
    print("Output saved to: output/")
    print("=" * 60)


if __name__ == '__main__':
    main()
