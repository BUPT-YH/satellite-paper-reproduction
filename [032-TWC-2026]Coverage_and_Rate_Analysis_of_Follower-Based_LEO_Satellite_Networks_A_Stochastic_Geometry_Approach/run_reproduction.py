"""
Run-all reproduction script
Coverage and Rate Analysis of Follower-Based LEO Satellite Networks
"""

import time
import numpy as np
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config as cfg
import stochastic_geometry as sg
import monte_carlo as mc
import simulation as sim
import plotting as plt_module


def print_banner():
    print("=" * 70)
    print("  Paper Reproduction: Coverage and Rate Analysis of Follower-Based")
    print("  LEO Satellite Networks: A Stochastic Geometry Approach")
    print("  Journal: IEEE TWC 2026")
    print("=" * 70)


def print_config():
    cfg.print_config()


def verify_basic_functions():
    print("\n" + "-" * 60)
    print("Verifying basic functions...")
    print("-" * 60)

    from scipy import integrate
    result_lu, _ = integrate.quad(sg.pdf_theta_LU, 0, cfg.theta_max, limit=200)
    print(f"  pdf_theta_LU integral = {result_lu:.6f} (should be 1.0)")

    result_min, _ = integrate.quad(sg.pdf_theta_min, 0, cfg.theta_max - cfg.theta_cap, limit=200)
    print(f"  pdf_theta_min integral = {result_min:.6f}")

    result_max, _ = integrate.quad(sg.pdf_theta_max_contact, cfg.theta_cap, cfg.theta_max + cfg.theta_cap, limit=200)
    print(f"  pdf_theta_max integral = {result_max:.6f}")

    print(f"  F_W(1) = {sg.cdf_W(1.0):.6f}")
    print(f"  F_W(100) = {sg.cdf_W(100.0):.6f} (should be ~1.0)")

    print(f"  xi_LU = {cfg.xi_LU:.4e}")
    print(f"  xi_FU = {cfg.xi_FU:.4e}")
    print(f"  xi_LF = {cfg.xi_LF:.4e}")

    print(f"\n  Quick MC validation (gamma_th = -5 dB, 10000 samples):")
    mc_l = mc.mc_outage_leader(-5, 10000)
    print(f"    MC Leader outage: {mc_l:.6e}")
    ana_l = sg.outage_leader(-5)
    print(f"    Analytical Leader outage: {ana_l:.6e}")

    if abs(mc_l - ana_l) / max(mc_l, ana_l, 1e-10) < 0.3:
        print("    [OK] MC matches analytical")
    else:
        print("    [WARN] MC differs from analytical (may need more samples)")

    print("  Basic function verification done!")


def run_all_figures():
    results = {}

    # Fig. 2: Outage probability
    print("\n" + "=" * 60)
    print("Starting Fig. 2: Outage Probability")
    print("=" * 60)
    t0 = time.time()
    results['fig2'] = sim.run_fig2_outage_probability()
    t_fig2 = time.time() - t0
    print(f"  Fig. 2 done, elapsed: {t_fig2:.1f}s")

    plt_module.plot_fig2(results['fig2'])

    # Fig. 4: Average rate vs N_F
    print("\n" + "=" * 60)
    print("Starting Fig. 4: Average Rate vs N_F")
    print("=" * 60)
    t0 = time.time()
    results['fig4'] = sim.run_fig4_avg_rate_vs_nf()
    t_fig4 = time.time() - t0
    print(f"  Fig. 4 done, elapsed: {t_fig4:.1f}s")

    plt_module.plot_fig4(results['fig4'])

    # Fig. 6: L.F vs N.F
    print("\n" + "=" * 60)
    print("Starting Fig. 6: L.F vs N.F Rate Comparison")
    print("=" * 60)
    t0 = time.time()
    results['fig6'] = sim.run_fig6_lf_vs_nf()
    t_fig6 = time.time() - t0
    print(f"  Fig. 6 done, elapsed: {t_fig6:.1f}s")

    plt_module.plot_fig6(results['fig6'])

    return results, {'fig2': t_fig2, 'fig4': t_fig4, 'fig6': t_fig6}


def print_summary(results, timing):
    print("\n" + "=" * 70)
    print("  Reproduction Summary")
    print("=" * 70)

    # Fig. 2
    fig2 = results['fig2']
    print("\n  Fig. 2 - Outage Probability:")
    gamma_points = [-10, -5, 0, 5]
    for g in gamma_points:
        idx = np.argmin(np.abs(fig2['gamma_th_dB'] - g))
        print(f"    gamma_th={g:+3d} dB: Leader={fig2['theorem1'][idx]:.4e}, "
              f"Cluster={fig2['theorem2'][idx]:.4e}, "
              f"Upper={fig2['upper_bound'][idx]:.4e}, "
              f"Lower={fig2['lower_bound'][idx]:.4e}")

    # Fig. 4
    fig4 = results['fig4']
    print("\n  Fig. 4 - Average Rate (Gbps):")
    for rho in fig4['rho_FU_dBW']:
        rates = fig4['middle'][rho] / 1e9
        print(f"    rho_FU={rho}dBW: NF=0->{rates[0]:.3f}, "
              f"NF=10->{rates[len(rates)//2]:.3f}, "
              f"NF=20->{rates[-1]:.3f} Gbps")

    # Fig. 6
    fig6 = results['fig6']
    print("\n  Fig. 6 - L.F vs N.F (Gbps):")
    for rho in fig6['rho_total_dBW']:
        nf_rate = fig6['nf_rates'][rho] / 1e9
        lf_max = fig6['lf_rates'][rho][-1] / 1e9
        gain = (lf_max - nf_rate) / nf_rate * 100 if nf_rate > 0 else 0
        print(f"    rho={rho}dBW: N.F={nf_rate:.3f}, L.F(max_NF)={lf_max:.3f}, "
              f"gain={gain:+.1f}%")

    print(f"\n  Timing: Fig.2={timing['fig2']:.0f}s, Fig.4={timing['fig4']:.0f}s, "
          f"Fig.6={timing['fig6']:.0f}s")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    print(f"\n  Output saved in: {output_dir}/")
    print("=" * 70)


def main():
    t_total_start = time.time()

    print_banner()
    print_config()

    verify_basic_functions()

    print("\n  Plotting diagnostics...")
    plt_module.plot_diagnostic_pdf()

    results, timing = run_all_figures()

    print_summary(results, timing)

    t_total = time.time() - t_total_start
    print(f"\n  All done! Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
