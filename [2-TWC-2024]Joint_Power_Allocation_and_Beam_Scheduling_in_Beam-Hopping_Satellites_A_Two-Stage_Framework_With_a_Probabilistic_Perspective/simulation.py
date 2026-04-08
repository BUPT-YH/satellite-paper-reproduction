"""
仿真模块 - 运行完整仿真流程
生成 Figure 4, Figure 11, Table V 所需的数据
"""
import numpy as np
from config import (
    N_BEAMS, M_SLOTS, BANDWIDTH, K_ACTIVE_DEFAULT,
    DEMAND_DENSITIES, N_INSTANCES, COMPENSATION_EPSILON,
    NOISE_FLOOR, BASELINE_POWER, generate_demands, get_system
)
from inverse_matrix_optimization import algorithm1_solve, compute_power_from_rho, compute_energy
from beam_scheduling import (
    rounding_algorithm, compute_modified_power,
    mpmm_scheduling, compute_capacity
)
def run_convergence_experiment(density=0.3, K=K_ACTIVE_DEFAULT, seed=42):
    """Figure 4: Algorithm 1 收敛性实验"""
    H, A, b, _ = get_system(N_BEAMS, seed=seed)
    d = generate_demands(N_BEAMS, density, seed=seed)
    d_compensated = d / COMPENSATION_EPSILON
    rho, p, energy_history = algorithm1_solve(
        A, b, d_compensated, K=K, B=BANDWIDTH, verbose=False
    )
    energy_db = [10 * np.log10(max(e, 1e-10)) for e in energy_history]
    return energy_db
def run_convergence_multiple_densities(K=K_ACTIVE_DEFAULT):
    """Figure 4: 多个需求密度下的收敛曲线"""
    results = {}
    for density in DEMAND_DENSITIES:
        print(f"  运行密度 r={density}...")
        energy_db = run_convergence_experiment(density=density, K=K)
        results[density] = energy_db
    return results
def compute_proposed_solution(H, d, A, b, K, B=BANDWIDTH):
    """运行完整的提出方法 (两阶段框架)"""
    d_comp = d / COMPENSATION_EPSILON
    rho, p, _ = algorithm1_solve(A, b, d_comp, K=K, verbose=False)
    if rho is None or p is None:
        return None
    d_hat = rounding_algorithm(rho, M_SLOTS, K)
    p_hat, rho_hat = compute_modified_power(d_hat, M_SLOTS, A, b, d_comp)
    X = mpmm_scheduling(rho_hat, p_hat, d_hat, M_SLOTS, K, A)
    capacity = compute_capacity(X, p_hat, H, M_SLOTS, B, NOISE_FLOOR)
    energy = np.sum(rho_hat * p_hat)
    return {
        'energy': energy,
        'capacity': capacity,
        'ratio': capacity / d,
        'rho_hat': rho_hat,
        'p_hat': p_hat,
    }
def compute_baseline_solution(H, d, A, b, K, B=BANDWIDTH):
    """基线方法: 固定功率 + 贪心调度"""
    N = len(d)
    X = np.zeros((N, M_SLOTS), dtype=int)
    remaining = d.copy()
    for t in range(M_SLOTS):
        if np.sum(remaining > 0) == 0:
            break
        active_count = min(K, int(np.sum(remaining > 0)))
        indices = np.argsort(-remaining)[:active_count]
        X[indices, t] = 1
        for n in indices:
            signal = BASELINE_POWER * H[n, n]**2
            interference = sum(
                BASELINE_POWER * H[j, n]**2
                for j in indices if j != n
            )
            sinr = signal / (interference + NOISE_FLOOR)
            rate = B * np.log2(1 + sinr) / M_SLOTS
            remaining[n] -= rate
    capacity = compute_capacity(
        X, np.ones(N) * BASELINE_POWER, H, M_SLOTS, B, NOISE_FLOOR
    )
    energy = BASELINE_POWER * np.sum(X)
    return {
        'energy': energy,
        'capacity': capacity,
        'ratio': capacity / d,
    }
def run_single_instance(density, K, seed=42):
    """运行单个仿真实例"""
    H, A, b, _ = get_system(N_BEAMS, seed=seed)
    d = generate_demands(N_BEAMS, density, seed=seed)
    proposed = compute_proposed_solution(H, d, A, b, K)
    baseline = compute_baseline_solution(H, d, A, b, K)
    if proposed is None:
        proposed = {'energy': 1e10, 'ratio': np.ones(N_BEAMS)}
    return {
        'proposed_energy': proposed['energy'],
        'proposed_ratio': proposed['ratio'],
        'baseline_energy': baseline['energy'],
        'baseline_ratio': baseline['ratio'],
    }
def run_performance_comparison(K=K_ACTIVE_DEFAULT):
    """Figure 11: 多个需求密度下的性能对比"""
    results = {}
    for density in DEMAND_DENSITIES:
        print(f"\n=== 需求密度 r={density} ===")
        pe, pr, be, br = [], [], [], []
        for inst in range(N_INSTANCES):
            result = run_single_instance(density, K, seed=inst)
            pe.append(result['proposed_energy'])
            pr.extend(result['proposed_ratio'].flatten())
            be.append(result['baseline_energy'])
            br.extend(result['baseline_ratio'].flatten())
        pe, be = np.array(pe), np.array(be)
        valid = (be > 0) & np.isfinite(pe) & np.isfinite(be)
        energy_ratio = np.where(valid, pe / be, np.nan)
        results[density] = {
            'proposed_energies': pe,
            'proposed_ratios': np.array(pr),
            'baseline_energies': be,
            'baseline_ratios': np.array(br),
            'energy_ratio': energy_ratio,
        }
    return results
def run_table_v(K=K_ACTIVE_DEFAULT):
    """Table V: 定量性能对比"""
    print("\n=== 生成 Table V 数据 ===")
    table_data = {}
    for density in DEMAND_DENSITIES:
        print(f"  需求密度 r={density}...")
        jp, jb_list, ers = [], [], []
        for inst in range(N_INSTANCES):
            result = run_single_instance(density, K, seed=inst)
            for ratios, store in [
                (result['proposed_ratio'].flatten(), jp),
                (result['baseline_ratio'].flatten(), jb_list),
            ]:
                valid = ratios[np.isfinite(ratios) & (ratios > 0)]
                if len(valid) > 0:
                    store.append(
                        (np.sum(valid))**2 / (len(valid) * np.sum(valid**2))
                    )
            if result['baseline_energy'] > 0 and np.isfinite(result['proposed_energy']):
                er = result['proposed_energy'] / result['baseline_energy']
                if np.isfinite(er) and 0 < er < 10:
                    ers.append(er)
        table_data[density] = {
            'jain_proposed': jp,
            'jain_baseline': jb_list,
            'energy_ratios': ers,
        }
    return table_data


if __name__ == "__main__":
    print("快速测试...")
    result = run_single_instance(0.3, K=26, seed=42)
    pe, be = result['proposed_energy'], result['baseline_energy']
    print(f"Proposed energy: {pe:.4f}")
    print(f"Baseline energy: {be:.4f}")
    print(f"Energy ratio: {pe/be:.4f}")
    print(f"Proposed C/D mean: {np.nanmean(result['proposed_ratio']):.4f}")
    print(f"Baseline C/D mean: {np.nanmean(result['baseline_ratio']):.4f}")
