"""
绘图模块 - 生成论文中的图表
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 中文字体和全局样式配置
rcParams['font.family'] = ['Times New Roman', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.size'] = 12

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_figure4(convergence_results):
    """
    Figure 4: Algorithm 1 收敛性
    不同需求密度下的能耗收敛曲线
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    densities = sorted(convergence_results.keys())

    for idx, density in enumerate(densities):
        energy_db = convergence_results[density]
        iterations = list(range(len(energy_db)))
        ax.plot(iterations, energy_db,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                markersize=5, linewidth=1.5,
                label=f'r = {density}')

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Consumed Energy (dB)', fontsize=13)
    ax.set_title('Convergence of Algorithm 1', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'figure4_convergence.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Figure 4 saved: {filepath}")
    return filepath


def plot_figure11(performance_results):
    """
    Figure 11: 与基线的性能对比
    左: 能耗比 CDF; 右: 需求匹配 CDF
    """
    densities = sorted(performance_results.keys())
    n_densities = len(densities)

    fig, axes = plt.subplots(n_densities, 2, figsize=(14, 4 * n_densities))
    if n_densities == 1:
        axes = axes.reshape(1, 2)

    colors_proposed = '#1f77b4'
    colors_baseline = '#ff7f0e'

    for idx, density in enumerate(densities):
        data = performance_results[density]

        # 左: 能耗比 CDF
        ax_left = axes[idx, 0]
        energy_ratio = data['energy_ratio']
        energy_ratio = energy_ratio[np.isfinite(energy_ratio) & (energy_ratio > 0)]

        if len(energy_ratio) > 0:
            sorted_er = np.sort(energy_ratio)
            cdf = np.arange(1, len(sorted_er) + 1) / len(sorted_er)
            ax_left.plot(sorted_er, cdf, color=colors_proposed, linewidth=2,
                         label='Proposed')
            ax_left.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5,
                            label='Ideal')

        ax_left.set_xlabel('Energy Ratio', fontsize=12)
        ax_left.set_ylabel('CDF', fontsize=12)
        ax_left.set_title(f'Energy Consumption (r={density})', fontsize=13)
        ax_left.legend(fontsize=10)
        ax_left.grid(True, alpha=0.3)

        # 右: 需求匹配 CDF
        ax_right = axes[idx, 1]
        proposed_ratios = data['proposed_ratios']
        baseline_ratios = data['baseline_ratios']

        # Proposed
        valid_p = proposed_ratios[np.isfinite(proposed_ratios) & (proposed_ratios > 0)]
        if len(valid_p) > 0:
            sorted_p = np.sort(valid_p)
            cdf_p = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
            ax_right.plot(sorted_p, cdf_p, color=colors_proposed, linewidth=2,
                          label='Proposed')

        # Baseline
        valid_b = baseline_ratios[np.isfinite(baseline_ratios) & (baseline_ratios > 0)]
        if len(valid_b) > 0:
            sorted_b = np.sort(valid_b)
            cdf_b = np.arange(1, len(sorted_b) + 1) / len(sorted_b)
            ax_right.plot(sorted_b, cdf_b, color=colors_baseline, linewidth=2,
                          label='Baseline')

        ax_right.axvline(x=1.0, color='red', linestyle='--', alpha=0.7,
                         label='Perfect match')
        ax_right.set_xlabel('Capacity/Demand Ratio', fontsize=12)
        ax_right.set_ylabel('CDF', fontsize=12)
        ax_right.set_title(f'Demand Matching (r={density})', fontsize=13)
        ax_right.legend(fontsize=10)
        ax_right.grid(True, alpha=0.3)
        ax_right.set_xlim(0, 2.5)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'figure11_performance_comparison.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Figure 11 saved: {filepath}")
    return filepath


def print_table_v(table_data):
    """
    打印 Table V: 定量性能对比
    """
    print("\n" + "=" * 80)
    print("TABLE V: Performance Comparison")
    print("=" * 80)
    print(f"{'Density':>10} | {'Jain (Proposed)':>16} | {'Jain (Baseline)':>16} | "
          f"{'Energy Ratio':>14} | {'Mean ER':>10} | {'Std ER':>10}")
    print("-" * 80)

    for density in sorted(table_data.keys()):
        data = table_data[density]

        jain_p = np.mean(data['jain_proposed']) if data['jain_proposed'] else 0
        jain_b = np.mean(data['jain_baseline']) if data['jain_baseline'] else 0
        er = data['energy_ratios']

        if er:
            er_mean = np.mean(er)
            er_std = np.std(er)
            er_min = np.min(er)
            er_max = np.max(er)
        else:
            er_mean = er_std = er_min = er_max = 0

        print(f"{'r='+str(density):>10} | {jain_p:>16.4f} | {jain_b:>16.4f} | "
              f"{er_min:.3f} - {er_max:.3f} | {er_mean:>10.4f} | {er_std:>10.4f}")

    print("=" * 80)
    return table_data
