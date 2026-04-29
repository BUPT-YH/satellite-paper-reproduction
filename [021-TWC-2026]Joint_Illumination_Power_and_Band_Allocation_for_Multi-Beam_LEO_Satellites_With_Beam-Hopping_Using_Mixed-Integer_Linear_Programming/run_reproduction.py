"""
一键复现脚本 — 运行所有仿真并生成图表
Fig. 5: 权重参数评估
Fig. 6: 缩减场景对比 (DB, GA, MILPsplit)
Fig. 8: 扩展场景对比 (DB, GA, MILPsplit)
"""
import os
import sys
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from config import REDUCED, ENLARGED, N_SEEDS, BETA
from system_model import BeamHoppingSystem
from db_baseline import run_db_baseline
from ga_baseline import run_ga_baseline
from optimizer_heuristic import run_optimized_method
from plotting import plot_reduced_scenario, plot_enlarged_scenario, plot_weighting_study


def run_scenario(scenario, methods, n_seeds=3):
    """对给定场景运行多种方法，返回平均 KPI"""
    user_counts = scenario["NU_range"]
    results = {m: {"UC": [], "EC": [], "TTS": []} for m in methods}

    for nu in user_counts:
        print(f"\n  NU = {nu}...", end="", flush=True)
        sc = dict(scenario)
        sc["NU"] = nu

        seed_kpis = {m: {"UC": [], "EC": [], "TTS": []} for m in methods}

        for seed in range(n_seeds):
            system = BeamHoppingSystem(sc, seed=seed)

            for method in methods:
                if method == "DB":
                    kpi = run_db_baseline(system, beta=BETA)
                elif method == "GA":
                    kpi = run_ga_baseline(system, beta=BETA)
                elif method == "MILPsplit":
                    kpi = run_optimized_method(system, beta=BETA)
                seed_kpis[method]["UC"].append(kpi["UC"])
                seed_kpis[method]["EC"].append(kpi["EC"])
                seed_kpis[method]["TTS"].append(kpi["TTS"])

            sys.stdout.write(f" s{seed}")
            sys.stdout.flush()

        for method in methods:
            results[method]["UC"].append(np.mean(seed_kpis[method]["UC"]))
            results[method]["EC"].append(np.mean(seed_kpis[method]["EC"]))
            results[method]["TTS"].append(np.mean(seed_kpis[method]["TTS"]))

    return results, user_counts


def run_weighting_study(scenario, beta_values, n_seeds=3):
    """权重参数研究 (Fig. 5)"""
    sc = dict(scenario)
    sc["NU"] = 20

    uc_list, ec_list, tts_list = [], [], []

    for beta in beta_values:
        print(f"  beta={beta:.2f}...", end="")
        s_uc, s_ec, s_tts = [], [], []
        for seed in range(n_seeds):
            system = BeamHoppingSystem(sc, seed=seed)
            kpi = run_optimized_method(system, beta=beta)
            s_uc.append(kpi["UC"])
            s_ec.append(kpi["EC"])
            s_tts.append(kpi["TTS"])
        uc_list.append(np.mean(s_uc))
        ec_list.append(np.mean(s_ec))
        tts_list.append(np.mean(s_tts))
        print(f" UC={np.mean(s_uc):.1f}%")

    return beta_values, uc_list, ec_list, tts_list


def main():
    print("=" * 60)
    print("论文复现: Joint Illumination, Power, and Band Allocation")
    print("  for Multi-Beam LEO Satellites With Beam-Hopping")
    print("  Using Mixed-Integer Linear Programming")
    print("=" * 60)

    # ===== Fig. 5: 权重参数评估 =====
    print("\n[Fig. 5] Weighting parameter study...")
    beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    bvs, uc_f5, ec_f5, tts_f5 = run_weighting_study(REDUCED, beta_values, n_seeds=2)

    # 归一化
    vals = np.array([uc_f5, ec_f5, tts_f5])
    vmax = vals.max()
    if vmax > 0:
        vals = vals / vmax
    plot_weighting_study(bvs, vals[0], vals[1], vals[2], OUTPUT_DIR)

    # ===== Fig. 6: 缩减场景 =====
    print("\n[Fig. 6] Reduced scenario comparison...")
    methods_r = ["DB", "GA", "MILPsplit"]
    res_r, uc_r = run_scenario(REDUCED, methods_r, n_seeds=3)

    plot_reduced_scenario(
        {m: res_r[m]["UC"] for m in methods_r},
        {m: res_r[m]["EC"] for m in methods_r},
        {m: res_r[m]["TTS"] for m in methods_r},
        uc_r, OUTPUT_DIR
    )

    # ===== Fig. 8: 扩展场景 =====
    print("\n[Fig. 8] Enlarged scenario comparison...")
    methods_e = ["DB", "GA", "MILPsplit"]
    res_e, uc_e = run_scenario(ENLARGED, methods_e, n_seeds=2)

    plot_enlarged_scenario(
        {m: res_e[m]["UC"] for m in methods_e},
        {m: res_e[m]["EC"] for m in methods_e},
        {m: res_e[m]["TTS"] for m in methods_e},
        uc_e, OUTPUT_DIR
    )

    # ===== 打印结果汇总 =====
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)

    for name, res, uc, methods in [
        ("Reduced", res_r, uc_r, methods_r),
        ("Enlarged", res_e, uc_e, methods_e)
    ]:
        print(f"\n--- {name} Scenario ---")
        print(f"{'Users':>6} | " + " | ".join(f"{m:>20}" for m in methods))
        print(f"{'':>6} | " + " | ".join(f"{'UC':>6} {'EC':>6} {'TTS':>5}" for _ in methods))
        for i, nu in enumerate(uc):
            vals = " | ".join(
                f"{res[m]['UC'][i]:>6.1f} {res[m]['EC'][i]:>6.1f} {res[m]['TTS'][i]:>5.1f}"
                for m in methods
            )
            print(f"{nu:>6} | {vals}")

    print(f"\n所有图表已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
