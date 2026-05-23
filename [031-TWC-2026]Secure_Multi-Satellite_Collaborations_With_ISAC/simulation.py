"""
仿真框架 — 蒙特卡洛仿真 + 参数扫描
论文: Secure Multi-Satellite Collaborations With ISAC
"""
import numpy as np
import config as cfg
from isac_msc import run_single_trial

# 9 种算法组合
ALLOC_METHODS = ['SHP', 'DP', 'CP']
BF_METHODS = ['PA', 'IA', 'JSC-BF']


def get_algo_label(alloc, bf):
    return f'{alloc}-{bf}'


def simulate_fig3(n_mc=None, Pm_range=None, M0=None, seed_base=0):
    """
    Fig. 3: 感知 SNR vs 功率预算 P_m

    返回:
        results: dict, key=算法名, value=(Pm_array, snr_mean_array, snr_std_array)
    """
    if n_mc is None:
        n_mc = cfg.n_mc
    if Pm_range is None:
        Pm_range = cfg.Pm_dBW_range
    if M0 is None:
        M0 = cfg.M0

    results = {}

    for alloc in ALLOC_METHODS:
        for bf in BF_METHODS:
            label = get_algo_label(alloc, bf)
            snr_means = []
            snr_stds = []

            for P_dBW in Pm_range:
                snr_trials = []
                for mc in range(n_mc):
                    seed = seed_base + mc * 100 + hash(label) % 1000
                    snr_db, _, _ = run_single_trial(
                        P_dBW=P_dBW, M0=M0,
                        alloc_method=alloc, bf_method=bf,
                        seed=seed
                    )
                    snr_trials.append(snr_db)

                snr_means.append(np.mean(snr_trials))
                snr_stds.append(np.std(snr_trials))

            results[label] = (
                np.array(Pm_range),
                np.array(snr_means),
                np.array(snr_stds)
            )
            print(f'  {label}: P_m={Pm_range[-1]}dBW -> SNR={snr_means[-1]:.1f} dB')

    return results


def simulate_fig6(n_mc=None, M0_range=None, P_dBW=25, seed_base=10000):
    """
    Fig. 6: 感知 SNR vs 协作卫星数 M_0

    返回:
        results: dict, key=算法名, value=(M0_array, snr_mean_array, snr_std_array)
    """
    if n_mc is None:
        n_mc = cfg.n_mc
    if M0_range is None:
        M0_range = cfg.M0_range

    results = {}

    for alloc in ALLOC_METHODS:
        for bf in BF_METHODS:
            label = get_algo_label(alloc, bf)
            snr_means = []
            snr_stds = []

            for M0 in M0_range:
                snr_trials = []
                for mc in range(n_mc):
                    seed = seed_base + mc * 100 + hash(label) % 1000
                    snr_db, _, _ = run_single_trial(
                        P_dBW=P_dBW, M0=int(M0),
                        alloc_method=alloc, bf_method=bf,
                        seed=seed
                    )
                    snr_trials.append(snr_db)

                snr_means.append(np.mean(snr_trials))
                snr_stds.append(np.std(snr_trials))

            results[label] = (
                np.array(M0_range),
                np.array(snr_means),
                np.array(snr_stds)
            )
            print(f'  {label}: M0={M0_range[-1]} -> SNR={snr_means[-1]:.1f} dB')

    return results


def simulate_fig9(n_mc=None, Pm_range=None, M0_values=None, seed_base=20000):
    """
    Fig. 9: CRB 定位误差 vs P_m, 不同 M_0
    使用最优算法 DP-JSC-BF

    返回:
        results: dict, key=f'M0={m0}', value=(Pm_array, crb_mean_array)
    """
    if n_mc is None:
        n_mc = cfg.n_mc
    if Pm_range is None:
        Pm_range = cfg.Pm_dBW_range
    if M0_values is None:
        M0_values = [1, 3, 5, 7, 9]

    alloc = 'DP'
    bf = 'JSC-BF'
    results = {}

    for M0 in M0_values:
        crb_means = []
        crb_stds = []

        for P_dBW in Pm_range:
            crb_trials = []
            for mc in range(n_mc):
                seed = seed_base + mc * 100 + M0 * 10
                _, crb, _ = run_single_trial(
                    P_dBW=P_dBW, M0=int(M0),
                    alloc_method=alloc, bf_method=bf,
                    seed=seed
                )
                crb_trials.append(crb)

            crb_means.append(np.mean(crb_trials))
            crb_stds.append(np.std(crb_trials))

        key = f'M0={M0}'
        results[key] = (
            np.array(Pm_range),
            np.array(crb_means),
            np.array(crb_stds),
            M0
        )
        print(f'  {key}: P_m={Pm_range[-1]}dBW -> CRB={crb_means[-1]:.2f} m')

    return results
