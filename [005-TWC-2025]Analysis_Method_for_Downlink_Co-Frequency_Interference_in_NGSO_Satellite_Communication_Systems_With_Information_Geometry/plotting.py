"""
绘图模块 - 统一风格的论文复现图表
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 全局字体配置
matplotlib.rcParams['font.family'] = ['SimHei', 'Times New Roman', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.dpi'] = 150

# 颜色方案 (专业且色盲友好)
COLORS = {
    'blue': '#2176AE',
    'red': '#D64933',
    'orange': '#F5A623',
    'green': '#33A1C9',
    'purple': '#7B68EE',
    'gray': '#888888',
    'black': '#333333',
}


def save_fig(fig, filename, output_dir='output'):
    """保存图片到 output 目录"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存: {filepath}")


def plot_fig3(avg_dtc_15, avg_dtc_100, output_dir='output'):
    """Fig. 3 - AIRM 度量下平均 DTC 随迭代次数变化"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    iterations = np.arange(1, len(avg_dtc_15) + 1)
    ax.plot(iterations, avg_dtc_15, 'b-o', markersize=4, linewidth=1.5,
            label='Sampling Points = 15')
    ax.plot(iterations, avg_dtc_100, 'r-s', markersize=4, linewidth=1.5,
            label='Sampling Points = 100')

    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Average DTC')
    ax.set_title('Fig. 3: Average DTC vs. Iterations (AIRM Metric)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    save_fig(fig, 'fig3_airm_avg_dtc_vs_iterations.png', output_dir)


def plot_fig4(dtc_values_15, threshold_15, dtc_values_100, threshold_100, output_dir='output'):
    """Fig. 4 - AIRM 度量下的 DTC 散点图和阈值"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 采样点 15
    mc_indices = np.arange(1, len(dtc_values_15) + 1)
    ax1.scatter(mc_indices, dtc_values_15, s=8, c=COLORS['blue'], alpha=0.6)
    ax1.axhline(y=threshold_15, color='red', linewidth=2, linestyle='--',
                label=f'Threshold = {threshold_15:.2f}')
    ax1.set_xlabel('Monte Carlo Index')
    ax1.set_ylabel('DTC (AIRM)')
    ax1.set_title('Sampling Points = 15')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 采样点 100
    mc_indices = np.arange(1, len(dtc_values_100) + 1)
    ax2.scatter(mc_indices, dtc_values_100, s=8, c=COLORS['blue'], alpha=0.6)
    ax2.axhline(y=threshold_100, color='red', linewidth=2, linestyle='--',
                label=f'Threshold = {threshold_100:.2f}')
    ax2.set_xlabel('Monte Carlo Index')
    ax2.set_ylabel('DTC (AIRM)')
    ax2.set_title('Sampling Points = 100')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Fig. 4: DTC and Threshold (AIRM Metric)', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig4_airm_dtc_scatter_threshold.png', output_dir)


def plot_fig5(dtc_values_15, threshold_15, dtc_values_100, threshold_100, output_dir='output'):
    """Fig. 5 - SKLD 度量下的 DTC 散点图和阈值"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 采样点 15
    mc_indices = np.arange(1, len(dtc_values_15) + 1)
    ax1.scatter(mc_indices, dtc_values_15, s=8, c=COLORS['blue'], alpha=0.6)
    ax1.axhline(y=threshold_15, color='red', linewidth=2, linestyle='--',
                label=f'Threshold = {threshold_15:.2f}')
    ax1.set_xlabel('Monte Carlo Index')
    ax1.set_ylabel('DTC (SKLD)')
    ax1.set_title('Sampling Points = 15')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 采样点 100
    mc_indices = np.arange(1, len(dtc_values_100) + 1)
    ax2.scatter(mc_indices, dtc_values_100, s=8, c=COLORS['blue'], alpha=0.6)
    ax2.axhline(y=threshold_100, color='red', linewidth=2, linestyle='--',
                label=f'Threshold = {threshold_100:.2f}')
    ax2.set_xlabel('Monte Carlo Index')
    ax2.set_ylabel('DTC (SKLD)')
    ax2.set_title('Sampling Points = 100')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Fig. 5: DTC and Threshold (SKLD Metric)', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig5_skld_dtc_scatter_threshold.png', output_dir)


def plot_fig6(jr_dtc_data_airm, jr_dtc_data_skld, sampling_points, output_dir='output'):
    """Fig. 6 - JR-DTC 随采样点数变化
    jr_dtc_data: dict, key=case名, value=(lower_bounds, upper_bounds)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    case_colors = {
        'Case0': COLORS['blue'],
        'Case1': COLORS['red'],
        'Case2': COLORS['orange'],
        'Case3': COLORS['green'],
    }
    case_labels = {
        'Case0': 'Case 0 (No interference)',
        'Case1': 'Case 1 (Sat 1 only)',
        'Case2': 'Case 2 (Sat 2 only)',
        'Case3': 'Case 3 (Both)',
    }

    # AIRM
    for case_name in ['Case0', 'Case1', 'Case2', 'Case3']:
        lower, upper = jr_dtc_data_airm[case_name]
        ax1.fill_between(sampling_points, lower, upper, alpha=0.3,
                         color=case_colors[case_name], label=case_labels[case_name])
        ax1.plot(sampling_points, lower, color=case_colors[case_name], linewidth=1)
        ax1.plot(sampling_points, upper, color=case_colors[case_name], linewidth=1)
    ax1.set_xlabel('Number of Sampling Points')
    ax1.set_ylabel('DTC (AIRM)')
    ax1.set_title('AIRM Metric')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # SKLD
    for case_name in ['Case0', 'Case1', 'Case2', 'Case3']:
        lower, upper = jr_dtc_data_skld[case_name]
        ax2.fill_between(sampling_points, lower, upper, alpha=0.3,
                         color=case_colors[case_name], label=case_labels[case_name])
        ax2.plot(sampling_points, lower, color=case_colors[case_name], linewidth=1)
        ax2.plot(sampling_points, upper, color=case_colors[case_name], linewidth=1)
    ax2.set_xlabel('Number of Sampling Points')
    ax2.set_ylabel('DTC (SKLD)')
    ax2.set_title('SKLD Metric')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Fig. 6: JR-DTC vs. Sampling Points', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig6_jr_dtc_vs_sampling.png', output_dir)


def plot_fig7(detection_prob_airm, detection_prob_skld, detection_prob_energy,
              sampling_points, output_dir='output'):
    """Fig. 7 - 正确判断概率 vs 采样点数"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    case_colors = {
        'Case0': COLORS['blue'],
        'Case1': COLORS['red'],
        'Case2': COLORS['orange'],
        'Case3': COLORS['green'],
        'Energy': COLORS['purple'],
    }
    case_markers = {
        'Case0': 'o',
        'Case1': 's',
        'Case2': '^',
        'Case3': 'D',
        'Energy': 'v',
    }
    case_labels = {
        'Case0': 'Case 0 (No interference)',
        'Case1': 'Case 1 (Sat 1 only)',
        'Case2': 'Case 2 (Sat 2 only)',
        'Case3': 'Case 3 (Both)',
        'Energy': 'Energy-based',
    }

    # AIRM
    for case_name in ['Case0', 'Case1', 'Case2', 'Case3']:
        ax1.plot(sampling_points, detection_prob_airm[case_name],
                 color=case_colors[case_name], marker=case_markers[case_name],
                 markersize=5, linewidth=1.5, label=case_labels[case_name])
    ax1.plot(sampling_points, detection_prob_energy,
             color=case_colors['Energy'], marker=case_markers['Energy'],
             markersize=5, linewidth=1.5, linestyle='--', label='Energy-based')
    ax1.set_xlabel('Number of Sampling Points')
    ax1.set_ylabel('Detection Probability')
    ax1.set_title('AIRM Metric')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_ylim([0.7, 1.02])
    ax1.grid(True, alpha=0.3)

    # SKLD
    for case_name in ['Case0', 'Case1', 'Case2', 'Case3']:
        ax2.plot(sampling_points, detection_prob_skld[case_name],
                 color=case_colors[case_name], marker=case_markers[case_name],
                 markersize=5, linewidth=1.5, label=case_labels[case_name])
    ax2.plot(sampling_points, detection_prob_energy,
             color=case_colors['Energy'], marker=case_markers['Energy'],
             markersize=5, linewidth=1.5, linestyle='--', label='Energy-based')
    ax2.set_xlabel('Number of Sampling Points')
    ax2.set_ylabel('Detection Probability')
    ax2.set_title('SKLD Metric')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.set_ylim([0.7, 1.02])
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Fig. 7: Detection Probability vs. Sampling Points', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig7_detection_probability.png', output_dir)


def plot_fig8(lon_grid, lat_grid, Z_potential, output_dir='output'):
    """Fig. 8 - 仿射嵌入 3D 曲面图"""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(lon_grid, lat_grid, Z_potential,
                           cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)

    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_zlabel('Potential Function Φ')
    ax.set_title('Fig. 8: Affine Embedding 3D Surface\n(Starlink Interference on 3 OneWeb Earth Stations)')

    fig.colorbar(surf, shrink=0.5, aspect=10, label='Potential Function Φ')

    save_fig(fig, 'fig8_affine_embedding_3d.png', output_dir)
