"""
主仿真逻辑模块
Coverage and Rate Analysis of Follower-Based LEO Satellite Networks

实现:
- Fig. 2: 中断概率 vs γ_th
- Fig. 4: 平均速率 vs N_F 和 ρ_FU
- Fig. 6: L.F vs N.F 速率对比
"""

import numpy as np
import time
import config as cfg
import stochastic_geometry as sg
import monte_carlo as mc


def run_fig2_outage_probability():
    """
    Fig. 2: 中断概率验证

    横轴: γ_th (-10 ~ 5 dB)
    纵轴: 中断概率
    曲线:
    - Theorem 1 (N_F=0) — 实线
    - Theorem 2 (N_F=10) — 实线
    - Corollary 1 上界 — 虚线
    - Corollary 1 下界 — 虚线
    - Monte Carlo仿真点 — marker
    """
    print("\n" + "=" * 60)
    print("Fig. 2: 中断概率验证")
    print("=" * 60)

    gamma_range = cfg.gamma_th_range_dB

    # 存储结果
    results = {
        'gamma_th_dB': gamma_range,
        'theorem1': [],       # Leader中断 (N_F=0)
        'theorem2': [],       # Cluster中断 (N_F=10)
        'upper_bound': [],    # 上界
        'lower_bound': [],    # 下界
        'mc_leader': [],      # MC Leader中断
        'mc_cluster': [],     # MC Cluster中断
    }

    n_mc = 50000  # MC仿真次数 (减少以加快速度)

    for i, g_th in enumerate(gamma_range):
        print(f"\r  Progress: {i+1}/{len(gamma_range)} | gamma_th = {g_th:.1f} dB", end="", flush=True)

        # 解析解
        p1 = sg.outage_leader(g_th)
        results['theorem1'].append(p1)

        p2 = sg.outage_cluster(g_th, N_F=10)
        results['theorem2'].append(p2)

        pu = sg.outage_upper_bound(g_th)
        results['upper_bound'].append(pu)

        pl = sg.outage_lower_bound(g_th)
        results['lower_bound'].append(pl)

        # Monte Carlo仿真 (只对部分点做仿真以加速)
        if i % 3 == 0 or i == len(gamma_range) - 1:
            mc_l = mc.mc_outage_leader(g_th, n_mc)
            mc_c = mc.mc_outage_cluster(g_th, 10, n_mc)
            results['mc_leader'].append((g_th, mc_l))
            results['mc_cluster'].append((g_th, mc_c))

    print("\n  完成!")

    # 转换为numpy数组
    for key in ['theorem1', 'theorem2', 'upper_bound', 'lower_bound']:
        results[key] = np.array(results[key])

    return results


def run_fig4_avg_rate_vs_nf():
    """
    Fig. 4: 平均速率 vs N_F 和 ρ_FU

    横轴: N_F (0 ~ 20, step 2)
    纵轴: 平均数据速率 (Gbps)
    曲线组: ρ_FU = 5, 10, 15, 20 dBW
    每组包含: 解析曲线 + 上界 + 下界 + 中值
    """
    print("\n" + "=" * 60)
    print("Fig. 4: Average Rate vs N_F and rho_FU")
    print("=" * 60)

    N_F_range = cfg.N_F_range
    rho_FU_range = cfg.rho_FU_range_dBW

    results = {
        'N_F': N_F_range,
        'rho_FU_dBW': rho_FU_range,
        'analytical': {},   # {rho_dBW: array of rates}
        'upper': {},
        'lower': {},
        'middle': {},
    }

    for rho_idx, rho_dBW in enumerate(rho_FU_range):
        print(f"\n  rho_FU = {rho_dBW} dBW:")

        # Temporarily modify Follower power
        original_xi_FU = cfg.xi_FU
        original_rho_FU_dBW = cfg.rho_FU_dBW
        original_rho_FU = cfg.rho_FU

        cfg.rho_FU_dBW = rho_dBW
        cfg.rho_FU = 10 ** (rho_dBW / 10)
        cfg.xi_FU = cfg.rho_FU * cfg.G * cfg.zeta_U * (cfg.nu / (4 * np.pi))**2 / cfg.sigma_U_sq / 1e6

        analytical_rates = []
        upper_rates = []
        lower_rates = []
        middle_rates = []

        for nf_idx, nf in enumerate(N_F_range):
            print(f"    N_F = {nf}...", end=" ", flush=True)

            # 计算解析解和上下界
            if nf == 0:
                # 只有Leader
                r_lu = sg.avg_rate_leader()
                analytical_rates.append(r_lu)
                upper_rates.append(r_lu)
                lower_rates.append(r_lu)
                middle_rates.append(r_lu)
            else:
                # 使用上下界 + 中值近似
                r_upper, r_lower, r_middle = sg.avg_rate_cluster_bounds(nf)
                analytical_rates.append(r_middle)  # 用中值近似作为解析解
                upper_rates.append(r_upper)
                lower_rates.append(r_lower)
                middle_rates.append(r_middle)

            print("完成")

        results['analytical'][rho_dBW] = np.array(analytical_rates)
        results['upper'][rho_dBW] = np.array(upper_rates)
        results['lower'][rho_dBW] = np.array(lower_rates)
        results['middle'][rho_dBW] = np.array(middle_rates)

        # 恢复原始参数
        cfg.xi_FU = original_xi_FU
        cfg.rho_FU_dBW = original_rho_FU_dBW
        cfg.rho_FU = original_rho_FU

    return results


def run_fig6_lf_vs_nf():
    """
    Fig. 6: L.F vs N.F 速率对比

    横轴: N_F (0 ~ 20)
    纵轴: 平均数据速率 (Gbps)
    两种方案对比:
    - L.F (leader-follower): ρ_LU^(1) = 10, 15, 20, 25, 30 dBW
    - N.F (non-follower): 对应相同总功率的单leader方案
    """
    print("\n" + "=" * 60)
    print("Fig. 6: L.F vs N.F 速率对比")
    print("=" * 60)

    N_F_range = np.arange(0, 21, 2)
    rho_total_range = cfg.rho_LU_total_range_dBW

    results = {
        'N_F': N_F_range,
        'rho_total_dBW': rho_total_range,
        'lf_rates': {},     # {rho_dBW: array} Leader-Follower方案速率
        'nf_rates': {},     # {rho_dBW: float} Non-Follower方案速率
    }

    for rho_total in rho_total_range:
        print(f"\n  Total power rho = {rho_total} dBW:")

        # N.F scheme: all power to leader
        print(f"    N.F scheme...", end=" ", flush=True)
        nf_rate = sg.avg_rate_non_follower(rho_total)
        results['nf_rates'][rho_total] = nf_rate
        print(f"{nf_rate/1e9:.4f} Gbps")

        # L.F scheme: ρ_LU^(1)为leader下行功率，LF功率来自独立预算
        # N.F: 所有功率给LU链路; L.F: 相同LU功率 + N_F个follower额外贡献
        original_xi_LU = cfg.xi_LU
        original_rho_LU_dBW = cfg.rho_LU_dBW
        original_rho_LU = cfg.rho_LU

        cfg.rho_LU_dBW = rho_total
        cfg.rho_LU = 10 ** (rho_total / 10)
        cfg.xi_LU = cfg.rho_LU * cfg.G * cfg.zeta_U * (cfg.nu / (4 * np.pi))**2 / cfg.sigma_U_sq / 1e6

        lf_rates = []
        for nf in N_F_range:
            print(f"    L.F N_F={nf}...", end=" ", flush=True)

            if nf == 0:
                r = sg.avg_rate_leader()
            else:
                _, _, r_middle = sg.avg_rate_cluster_bounds(nf)
                r = r_middle

            lf_rates.append(r)
            print(f"{r/1e9:.4f} Gbps")

        results['lf_rates'][rho_total] = np.array(lf_rates)

        # 恢复原始参数
        cfg.xi_LU = original_xi_LU
        cfg.rho_LU_dBW = original_rho_LU_dBW
        cfg.rho_LU = original_rho_LU

    return results


if __name__ == "__main__":
    # 快速测试
    print("快速测试仿真逻辑...")

    # 测试单个点的中断概率
    print("\nLeader outage probability (gamma_th = -5 dB):")
    t0 = time.time()
    p_out = sg.outage_leader(-5)
    t1 = time.time()
    print(f"  解析值: {p_out:.6e}, 耗时: {t1-t0:.2f}s")

    t0 = time.time()
    p_mc = mc.mc_outage_leader(-5, 50000)
    t1 = time.time()
    print(f"  MC仿真: {p_mc:.6e}, 耗时: {t1-t0:.2f}s")

    # 测试Cluster中断概率
    print("\nCluster outage probability (gamma_th = -5 dB, N_F=10):")
    t0 = time.time()
    p_cluster = sg.outage_cluster(-5, N_F=10)
    t1 = time.time()
    print(f"  解析值: {p_cluster:.6e}, 耗时: {t1-t0:.2f}s")

    t0 = time.time()
    p_mc_cluster = mc.mc_outage_cluster(-5, 10, 50000)
    t1 = time.time()
    print(f"  MC仿真: {p_mc_cluster:.6e}, 耗时: {t1-t0:.2f}s")
