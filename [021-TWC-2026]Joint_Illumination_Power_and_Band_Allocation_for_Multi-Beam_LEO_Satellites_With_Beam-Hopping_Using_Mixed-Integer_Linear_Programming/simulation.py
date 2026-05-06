"""
主仿真逻辑 — 运行场景对比仿真，收集 KPI 结果
"""
import numpy as np
from config import N_SEEDS, BETA
from system_model import BeamHoppingSystem
from db_baseline import run_db_baseline
from ga_baseline import run_ga_baseline
from optimizer_heuristic import run_optimized_method


def run_single_scenario(scenario, nu, methods, n_seeds=3, beta=BETA):
    """
    对给定用户数运行多种方法，返回平均 KPI
    methods: list of 'DB', 'GA', 'MILPsplit'
    """
    sc = dict(scenario)
    sc["NU"] = nu

    seed_kpis = {m: {"UC": [], "EC": [], "TTS": []} for m in methods}

    for seed in range(n_seeds):
        system = BeamHoppingSystem(sc, seed=seed)
        for method in methods:
            if method == "DB":
                kpi = run_db_baseline(system, beta=beta)
            elif method == "GA":
                kpi = run_ga_baseline(system, beta=beta)
            elif method == "MILPsplit":
                kpi = run_optimized_method(system, beta=beta)
            seed_kpis[method]["UC"].append(kpi["UC"])
            seed_kpis[method]["EC"].append(kpi["EC"])
            seed_kpis[method]["TTS"].append(kpi["TTS"])

    results = {}
    for method in methods:
        results[method] = {
            "UC": np.mean(seed_kpis[method]["UC"]),
            "EC": np.mean(seed_kpis[method]["EC"]),
            "TTS": np.mean(seed_kpis[method]["TTS"]),
        }
    return results


def run_full_scenario(scenario, methods=None, n_seeds=3, beta=BETA):
    """
    对完整场景（所有用户数）运行仿真
    返回 (results_dict, user_counts)
    """
    if methods is None:
        methods = ["DB", "GA", "MILPsplit"]

    user_counts = scenario["NU_range"]
    results = {m: {"UC": [], "EC": [], "TTS": []} for m in methods}

    for nu in user_counts:
        res = run_single_scenario(scenario, nu, methods, n_seeds, beta)
        for method in methods:
            results[method]["UC"].append(res[method]["UC"])
            results[method]["EC"].append(res[method]["EC"])
            results[method]["TTS"].append(res[method]["TTS"])

    return results, user_counts


def run_weighting_study(scenario, nu, beta_values, n_seeds=3):
    """
    权重参数研究：固定用户数，变化 beta
    返回 (beta_values, uc_list, ec_list, tts_list)
    """
    uc_list, ec_list, tts_list = [], [], []

    for beta in beta_values:
        res = run_single_scenario(scenario, nu, ["MILPsplit"], n_seeds, beta)
        uc_list.append(res["MILPsplit"]["UC"])
        ec_list.append(res["MILPsplit"]["EC"])
        tts_list.append(res["MILPsplit"]["TTS"])

    return beta_values, uc_list, ec_list, tts_list
