"""
一键复现脚本
"""
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import (
    compute_all_signal_params,
    simulate_fig3,
    simulate_fig4_5,
    simulate_fig6,
    simulate_fig7,
    simulate_fig8,
)
from plotting import plot_fig3, plot_fig4, plot_fig5, plot_fig6, plot_fig7, plot_fig8


def main():
    print("=" * 60)
    print("Reproduction: Info Geometry for NGSO CFI Analysis")
    print("IEEE TWC, Vol. 24, No. 10, 2025")
    print("=" * 60)

    np.random.seed(42)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Step 0
    print("\n[Step 0] Computing signal parameters...")
    A0, sigma0, A_interf, sigma_interf = compute_all_signal_params()

    # Fig. 3
    print("\n" + "=" * 50)
    t0 = time.time()
    avg15, avg100 = simulate_fig3(A0, sigma0)
    plot_fig3(avg15, avg100, output_dir)
    print(f"  Time: {time.time()-t0:.1f}s")

    # Fig. 4 & 5
    print("\n" + "=" * 50)
    t0 = time.time()
    airm_r, skld_r = simulate_fig4_5(A0, sigma0)
    plot_fig4(airm_r[15][0], airm_r[15][1], airm_r[100][0], airm_r[100][1], output_dir)
    plot_fig5(skld_r[15][0], skld_r[15][1], skld_r[100][0], skld_r[100][1], output_dir)
    print(f"  Time: {time.time()-t0:.1f}s")

    # Fig. 6
    print("\n" + "=" * 50)
    t0 = time.time()
    jr_a, jr_s, sp6 = simulate_fig6(A0, sigma0, A_interf, sigma_interf)
    plot_fig6(jr_a, jr_s, sp6, output_dir)
    print(f"  Time: {time.time()-t0:.1f}s")

    # Fig. 7
    print("\n" + "=" * 50)
    t0 = time.time()
    pa, ps, pe, sp7 = simulate_fig7(A0, sigma0, A_interf, sigma_interf)
    plot_fig7(pa, ps, pe, sp7, output_dir)
    print(f"  Time: {time.time()-t0:.1f}s")

    # Fig. 8
    print("\n" + "=" * 50)
    t0 = time.time()
    lon_g, lat_g, Z = simulate_fig8()
    plot_fig8(lon_g, lat_g, Z, output_dir)
    print(f"  Time: {time.time()-t0:.1f}s")

    print("\n" + "=" * 60)
    print("All figures generated! Check output/ directory.")
    print("=" * 60)


if __name__ == '__main__':
    main()
