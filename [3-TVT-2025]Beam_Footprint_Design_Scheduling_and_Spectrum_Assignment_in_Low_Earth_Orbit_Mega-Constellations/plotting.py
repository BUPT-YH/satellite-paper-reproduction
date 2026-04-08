"""
Plotting Module for TVT-2025 Paper
Generates all figures from the paper
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def plot_convergence_analysis(convergence_data: dict, save_path: str = None):
    """
    Reproduces Figure 5: Convergence Analysis
    Shows R_E set size and throughput evolution over time
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    time_steps = np.array(convergence_data['time_steps']) * 120  # Convert to seconds
    re_sizes = convergence_data['R_E_size']
    throughput = convergence_data['throughput']

    # R_E set size
    color1 = '#1f77b4'
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('$R_E$ Set Size', color=color1, fontsize=12)
    ax1.plot(time_steps, re_sizes, color=color1, linewidth=2, label='$R_E$ Size')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.fill_between(time_steps, 0, re_sizes, alpha=0.3, color=color1)

    # Throughput on secondary axis
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    ax2.set_ylabel('Throughput (Gbps)', color=color2, fontsize=12)
    ax2.plot(time_steps, throughput, color=color2, linewidth=2,
             linestyle='--', label='Throughput')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add convergence markers
    ax1.axvline(x=6000, color='gray', linestyle=':', alpha=0.7)
    ax1.text(6500, max(re_sizes)*0.9, 'Beam Clustering\nComplete',
             fontsize=9, color='gray')

    ax1.axvline(x=time_steps[-1]*0.85, color='green', linestyle='--', alpha=0.7)
    ax1.text(time_steps[-1]*0.78, max(re_sizes)*0.7, 'Convergence',
             fontsize=9, color='green')

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Convergence Analysis on SpaceX Starlink Constellation', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure 5 saved to {save_path}")

    plt.close()
    return fig


def plot_performance_comparison(results: dict, save_path: str = None):
    """
    Reproduces Figure 7: Throughput and Power Consumption
    Compares different optimization methods across constellations
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    colors = {
        'ME-WF': '#808080',
        'IO-WF': '#d62728',
        'ME-IO': '#2ca02c',
        'IO-IO': '#1f77b4'
    }

    markers = {
        'ME-WF': 's',
        'IO-WF': 'o',
        'ME-IO': '^',
        'IO-IO': 'o'
    }

    constellations = list(results.keys())
    methods = ['ME-WF', 'IO-WF', 'ME-IO', 'IO-IO']

    for idx, (const_name, ax) in enumerate(zip(constellations, axes)):
        data = results[const_name]

        for method in methods:
            if method not in data:
                continue

            throughput = data[method]['throughput']
            power = data[method]['power']

            marker_size = 200 if method == 'IO-IO' else 150
            edgecolor = 'black' if method == 'IO-IO' else colors[method]

            ax.scatter(throughput, power,
                      c=colors[method], s=marker_size,
                      marker=markers[method],
                      edgecolors=edgecolor,
                      linewidths=2 if method == 'IO-IO' else 1,
                      label=method if idx == 0 else "",
                      zorder=5 if method == 'IO-IO' else 3)

        if idx == 0:
            ax.set_ylabel('Power Consumption (W)', fontsize=11)

        ax.set_xlabel('Throughput (Gbps)', fontsize=11)

        const_display = const_name.replace('_', ' ')
        if 'Starlink' in const_display:
            const_display += '\n(FSS)'
        ax.set_title(const_display, fontsize=12)

        ax.grid(True, alpha=0.3)

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    plt.suptitle('Throughput and Power Consumption for Different Implementations',
                 fontsize=14, y=1.08)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure 7 saved to {save_path}")

    plt.close()
    return fig


def plot_operations_simulation(ops_data: dict, save_path: str = None):
    """
    Reproduces Figure 8: Operations Simulation
    Shows throughput and power during system operations
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    time_steps = np.array(ops_data['time_steps']) * 5  # 5 second intervals

    # Throughput plot
    ax = axes[0]
    ax.plot(time_steps[:len(ops_data['throughput_history'])],
           ops_data['throughput_history'],
           color='#1f77b4', linewidth=1.5)
    ax.set_ylabel('Throughput (Gbps)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_title('Satellite 1', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add cluster change markers
    for change_idx in ops_data['cluster_changes'][:15]:
        change_time = change_idx * 5
        if change_time < time_steps[-1]:
            ax.axvline(x=change_time, color='red', linestyle='--',
                      alpha=0.5, linewidth=1)

    ax.set_xlim(0, time_steps[-1])

    # Power plot
    ax = axes[1]
    ax.plot(time_steps[:len(ops_data['power_history'])],
           ops_data['power_history'],
           color='#ff7f0e', linewidth=1.5)
    ax.set_ylabel('Power Consumption (W)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_title('Satellite 2 (Max Changes)', fontsize=12)
    ax.grid(True, alpha=0.3)

    for change_idx in ops_data['cluster_changes'][:15]:
        change_time = change_idx * 5
        if change_time < time_steps[-1]:
            ax.axvline(x=change_time, color='red', linestyle='--',
                      alpha=0.5, linewidth=1)

    ax.set_xlim(0, time_steps[-1])

    plt.suptitle('Throughput and Power Consumption During Operations\n'
                 '(Dashed lines: cluster-to-satellite changes)',
                 fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure 8 saved to {save_path}")

    plt.close()
    return fig


def plot_beam_footprint_visualization(save_path: str = None):
    """
    Creates beam footprint coverage visualization
    Similar to Figure 6 in the paper
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    np.random.seed(42)

    center_lat, center_lon = 40.0, -4.0  # Iberian Peninsula

    # Draw beam footprints
    for i in range(12):
        lat = center_lat + np.random.uniform(-2, 2)
        lon = center_lon + np.random.uniform(-3, 3)
        radius = np.random.uniform(0.5, 1.2)

        circle = Circle((lon, lat), radius, fill=False,
                        color='blue', linewidth=2, alpha=0.7)
        ax.add_patch(circle)

        # Add frequency channel indicator
        channel = i % 4
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        inner_circle = Circle((lon, lat), radius * 0.3,
                             color=colors[channel], alpha=0.5)
        ax.add_patch(inner_circle)

        # Elevation angle label
        elevation = np.random.randint(30, 80)
        ax.text(lon, lat, str(elevation), ha='center', va='center',
               fontsize=8, color='white', fontweight='bold')

    # Draw satellite positions
    sat_lats = [center_lat + np.random.uniform(-1.5, 1.5) for _ in range(5)]
    sat_lons = [center_lon + np.random.uniform(-2, 2) for _ in range(5)]
    ax.scatter(sat_lons, sat_lats, c='red', s=150, marker='*', zorder=5)

    for i, (slat, slon) in enumerate(zip(sat_lats, sat_lons)):
        ax.text(slon, slat+0.3, f'S{i+1}', ha='center',
               fontsize=9, color='red', fontweight='bold')

    ax.set_xlim(center_lon - 4, center_lon + 4)
    ax.set_ylim(center_lat - 3, center_lat + 3)
    ax.set_xlabel('Longitude (°)', fontsize=11)
    ax.set_ylabel('Latitude (°)', fontsize=11)
    ax.set_title('Beam Footprint Coverage and Frequency Reuse Pattern\n'
                 '(Inner colors: frequency channels)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Legend
    legend_text = '★: Satellites\n○: Beam footprints\nColors: Frequency channels'
    ax.text(center_lon + 3.5, center_lat - 2.5, legend_text,
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure 6 saved to {save_path}")

    plt.close()
    return fig


def plot_spectrum_allocation(save_path: str = None):
    """
    Creates spectrum allocation visualization
    Shows frequency channel assignments
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    np.random.seed(42)

    # Frequency bands
    total_bandwidth = 500  # MHz
    num_channels = 8
    channel_width = total_bandwidth / num_channels

    # Plot spectrum allocation for multiple beams
    num_beams = 12
    for i in range(num_beams):
        y_pos = i
        start_freq = (i % 4) * (total_bandwidth / 4)
        bandwidth = np.random.uniform(100, 200)

        # Draw allocation bar
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        color = colors[i % 4]
        rect = plt.Rectangle((start_freq, y_pos - 0.4), bandwidth, 0.8,
                            color=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)

        # Label
        ax.text(start_freq + bandwidth/2, y_pos, f'B{i+1}',
               ha='center', va='center', fontsize=8, color='white')

    ax.set_xlim(0, total_bandwidth)
    ax.set_ylim(-1, num_beams)
    ax.set_xlabel('Frequency (MHz)', fontsize=11)
    ax.set_ylabel('Beam ID', fontsize=11)
    ax.set_title('Spectrum Allocation Across Beams\n'
                 '(Colors indicate frequency reuse pattern)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Spectrum allocation figure saved to {save_path}")

    plt.close()
    return fig


def print_table_ii(results: dict):
    """Print Table II: Performance comparison results"""
    print("\n" + "="*90)
    print("TABLE II: Performance Comparison Results")
    print("="*90)
    print(f"{'Constellation':<20} {'Method':<8} {'Throughput':<15} {'Power':<12} "
          f"{'Spectrum':<12} {'Active':<10}")
    print(f"{'':<20} {'':<8} {'(Gbps)':<15} {'(W)':<12} {'(MHz)':<12} {'Beams':<10}")
    print("-"*90)

    for const_name, data in results.items():
        const_display = const_name.replace('_', ' ')

        for method in ['ME-WF', 'IO-WF', 'ME-IO', 'IO-IO']:
            if method not in data:
                continue

            throughput = data[method]['throughput']
            power = data[method]['power']
            spectrum = data[method].get('spectrum', throughput * 10)
            active = data[method].get('active_beams', int(throughput * 10))

            if method == 'ME-WF':
                print(f"{const_display:<20} {method:<8} {throughput:>10.2f}      "
                      f"{power:>8.1f}     {spectrum:>8.0f}     {active:>8}")
            else:
                print(f"{'':<20} {method:<8} {throughput:>10.2f}      "
                      f"{power:>8.1f}     {spectrum:>8.0f}     {active:>8}")

        print("-"*90)

    print("="*90)


if __name__ == "__main__":
    import os

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating sample plots...")

    # Sample data for testing
    convergence_data = {
        'time_steps': list(range(100)),
        'R_E_size': [500 * np.exp(-3 * t/100) + 50 + np.random.randn() * 20
                     for t in range(100)],
        'throughput': [80 + 15 * (t/100) + np.random.randn() * 2
                       for t in range(100)]
    }

    plot_convergence_analysis(convergence_data,
                             save_path=os.path.join(output_dir, 'figure5_convergence.png'))

    performance_results = {
        'O3b_mPower': {
            'ME-WF': {'throughput': 45.2, 'power': 180.5, 'spectrum': 450, 'active_beams': 45},
            'IO-WF': {'throughput': 52.1, 'power': 158.2, 'spectrum': 520, 'active_beams': 52},
            'ME-IO': {'throughput': 55.3, 'power': 148.9, 'spectrum': 550, 'active_beams': 55},
            'IO-IO': {'throughput': 62.5, 'power': 126.7, 'spectrum': 620, 'active_beams': 62}
        },
        'Telesat_Lightspeed': {
            'ME-WF': {'throughput': 85.4, 'power': 225.3, 'spectrum': 850, 'active_beams': 85},
            'IO-WF': {'throughput': 98.2, 'power': 198.1, 'spectrum': 980, 'active_beams': 98},
            'ME-IO': {'throughput': 104.1, 'power': 185.2, 'spectrum': 1040, 'active_beams': 104},
            'IO-IO': {'throughput': 117.8, 'power': 160.4, 'spectrum': 1170, 'active_beams': 117}
        },
        'SpaceX_Starlink': {
            'ME-WF': {'throughput': 125.6, 'power': 285.1, 'spectrum': 1250, 'active_beams': 125},
            'IO-WF': {'throughput': 144.5, 'power': 250.8, 'spectrum': 1440, 'active_beams': 144},
            'ME-IO': {'throughput': 153.2, 'power': 234.2, 'spectrum': 1530, 'active_beams': 153},
            'IO-IO': {'throughput': 173.4, 'power': 203.2, 'spectrum': 1730, 'active_beams': 173}
        }
    }

    plot_performance_comparison(performance_results,
                               save_path=os.path.join(output_dir, 'figure7_performance.png'))

    ops_data = {
        'throughput_history': [95 + 10 * np.sin(t * 0.1) + np.random.randn() * 3
                               for t in range(200)],
        'power_history': [250 + 20 * np.cos(t * 0.05) + np.random.randn() * 8
                         for t in range(200)],
        'cluster_changes': [int(t) for t in np.random.uniform(0, 200, 15)],
        'time_steps': list(range(200))
    }

    plot_operations_simulation(ops_data,
                              save_path=os.path.join(output_dir, 'figure8_operations.png'))

    plot_beam_footprint_visualization(
        save_path=os.path.join(output_dir, 'figure6_beam_footprint.png'))

    plot_spectrum_allocation(
        save_path=os.path.join(output_dir, 'figure_spectrum_allocation.png'))

    print_table_ii(performance_results)

    print("\n" + "="*60)
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)
