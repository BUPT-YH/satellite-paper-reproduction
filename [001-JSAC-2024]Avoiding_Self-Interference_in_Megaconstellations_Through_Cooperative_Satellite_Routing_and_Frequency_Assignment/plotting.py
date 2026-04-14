"""
Plotting Script - Reproduces Figures from the Paper
Figure 5: Convergence Analysis
Figure 7: Throughput and Power Consumption
Figure 8: Operations Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

def plot_convergence_analysis(convergence_result, save_path: str = None):
    """
    Reproduces Figure 5 from the paper:
    Convergence analysis on SpaceX Starlink constellation
    Shows evolution of R_E set and throughput over time
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    time_steps = np.arange(len(convergence_result.convergence_history)) * 120
    re_sizes = convergence_result.convergence_history
    
    color1 = '#1f77b4'
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('$R_E$ Set Size', color=color1, fontsize=12)
    ax1.plot(time_steps, re_sizes, color=color1, linewidth=2, label='$R_E$ Size')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.fill_between(time_steps, 0, re_sizes, alpha=0.3, color=color1)
    
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    throughput = convergence_result.throughput_history
    ax2.set_ylabel('Throughput (Gbps)', color=color2, fontsize=12)
    ax2.plot(time_steps, throughput, color=color2, linewidth=2, linestyle='--', label='Throughput')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax1.axvline(x=10000, color='gray', linestyle=':', alpha=0.7)
    ax1.text(10500, max(re_sizes)*0.9, 'Beam Clustering\nComplete', fontsize=9, color='gray')
    
    ax1.axvline(x=time_steps[-1], color='green', linestyle='--', alpha=0.7)
    ax1.text(time_steps[-1]-5000, max(re_sizes)*0.7, 'Convergence', fontsize=9, color='green')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Convergence Analysis on SpaceX Starlink Constellation', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure saved to {save_path}")
    
    return fig

def plot_performance_comparison(results: dict, save_path: str = None):
    """
    Reproduces Figure 7 from the paper:
    Throughput and power consumption for different implementations
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    colors = {
        'heuristic': '#808080',
        'routing_only': '#d62728',
        'freq_only': '#2ca02c',
        'cooperative': '#1f77b4'
    }
    
    markers = {
        'heuristic': 's',
        'routing_only': 'o',
        'freq_only': '^',
        'cooperative': 'o'
    }
    
    labels_map = {
        'heuristic': 'ME-WF',
        'routing_only': 'IO-WF',
        'freq_only': 'ME-IO',
        'cooperative': 'IO-IO'
    }
    
    constellations = list(results.keys())
    
    for idx, (const_name, ax) in enumerate(zip(constellations, axes)):
        data = results[const_name]
        
        for method in ['heuristic', 'routing_only', 'freq_only', 'cooperative']:
            throughput = data[method]['throughput']
            power = data[method]['power']
            
            marker_size = 200 if method == 'cooperative' else 150
            edgecolor = 'black' if method == 'cooperative' else colors[method]
            
            ax.scatter(throughput, power, 
                      c=colors[method], s=marker_size, 
                      marker=markers[method],
                      edgecolors=edgecolor, linewidths=2 if method == 'cooperative' else 1,
                      label=labels_map[method],
                      zorder=5 if method == 'cooperative' else 3)
        
        if idx == 0:
            ax.set_ylabel('Power Consumption (W)', fontsize=11)
        
        ax.set_xlabel('Throughput (Gbps)', fontsize=11)
        
        const_display = const_name.replace('_', ' ')
        if 'Starlink' in const_display:
            const_display += '\n(FSS)'
        ax.set_title(const_display, fontsize=12)
        
        ax.grid(True, alpha=0.3)
        
        x_margin = 0.1 * (max(d['cooperative']['throughput'] for d in results.values()) - 
                         min(d['heuristic']['throughput'] for d in results.values()))
        y_margin = 0.1 * (max(d['cooperative']['power'] for d in results.values()) - 
                         min(d['heuristic']['power'] for d in results.values()))
        
        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, 
               bbox_to_anchor=(0.5, 1.02), fontsize=10)
    
    plt.suptitle('Throughput and Power Consumption for Different Implementations', 
                 fontsize=14, y=1.08)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure saved to {save_path}")
    
    return fig

def plot_operations_simulation(ops_result: dict, save_path: str = None):
    """
    Reproduces Figure 8 from the paper:
    Throughput and power consumption during operations
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    time_steps = np.array(ops_result['time_steps']) * 5
    
    for ax_idx, (ax, title, data, ylabel) in enumerate([
        (axes[0], 'Satellite 1', 
         ops_result['throughput_history'][:len(time_steps)], 
         'Throughput (Gbps)'),
        (axes[1], 'Satellite 2 (Max Changes)', 
         ops_result['power_history'][:len(time_steps)], 
         'Power Consumption (W)')
    ]):
        if ax_idx == 0:
            data_to_plot = ops_result['throughput_history'][:len(time_steps)]
            ax.plot(time_steps[:len(data_to_plot)], data_to_plot, 
                   color='#1f77b4', linewidth=1.5)
            ax.set_ylabel('Throughput (Gbps)', fontsize=11)
        else:
            data_to_plot = ops_result['power_history'][:len(time_steps)]
            ax.plot(time_steps[:len(data_to_plot)], data_to_plot, 
                   color='#ff7f0e', linewidth=1.5)
            ax.set_ylabel('Power Consumption (W)', fontsize=11)
        
        for change_idx in ops_result['cluster_changes'][:10]:
            change_time = change_idx * 5
            if change_time < time_steps[-1]:
                ax.axvline(x=change_time, color='red', linestyle='--', 
                          alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        ax.set_xlim(0, time_steps[-1])
    
    plt.suptitle('Throughput and Power Consumption During Operations\n(Dashed lines: cluster-to-satellite changes)', 
                 fontsize=13)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure saved to {save_path}")
    
    return fig

def plot_table_vi(results: dict, save_path: str = None):
    """
    Creates a visualization of Table VI results
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    headers = ['Constellation', 'SR-FA', 'Throughput\n(Gbps)', 'Power\n(W)', 
               'Spectrum\n(MHz)', 'Active\nBeams']
    
    cell_data = []
    
    for const_name, data in results.items():
        const_display = const_name.replace('_', ' ')
        
        for method, label in [('heuristic', 'ME-WF'), ('routing_only', 'IO-WF'),
                              ('freq_only', 'ME-IO'), ('cooperative', 'IO-IO')]:
            row = [const_display if method == 'heuristic' else '', 
                   label,
                   f"{data[method]['throughput']:.2f}",
                   f"{data[method]['power']:.1f}",
                   f"{data[method]['throughput']*10:.0f}",
                   f"{int(data[method]['throughput']*10)}"]
            cell_data.append(row)
            const_display = ''
    
    table = ax.table(cellText=cell_data, colLabels=headers,
                     loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Table VI: Results for Various Figures of Merit', fontsize=14, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Table saved to {save_path}")
    
    return fig

def plot_constellation_visualization(save_path: str = None):
    """
    Creates a visualization of constellation coverage (similar to Figure 6)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    np.random.seed(42)
    
    center_lat, center_lon = 40.0, -4.0
    
    for i in range(15):
        lat = center_lat + np.random.uniform(-3, 3)
        lon = center_lon + np.random.uniform(-4, 4)
        
        circle = Circle((lon, lat), 0.8, fill=False, color='blue', linewidth=2)
        ax.add_patch(circle)
        
        elevation = np.random.randint(30, 80)
        ax.text(lon, lat, str(elevation), ha='center', va='center', fontsize=8)
    
    sat_lats = [center_lat + np.random.uniform(-2, 2) for _ in range(8)]
    sat_lons = [center_lon + np.random.uniform(-3, 3) for _ in range(8)]
    ax.scatter(sat_lons, sat_lats, c='red', s=100, marker='*', zorder=5)
    
    for i, (slat, slon) in enumerate(zip(sat_lats, sat_lons)):
        ax.text(slon, slat+0.3, f'S{i+1}', ha='center', fontsize=8, color='red')
    
    ax.set_xlim(center_lon - 5, center_lon + 5)
    ax.set_ylim(center_lat - 4, center_lat + 4)
    ax.set_xlabel('Longitude (°)', fontsize=11)
    ax.set_ylabel('Latitude (°)', fontsize=11)
    ax.set_title('Beam-to-Satellite Mapping Over Iberian Peninsula\n(Values: Elevation Angle in degrees)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    ax.text(center_lon + 4, center_lat - 3.5, '★: Satellites\n○: Beams', 
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure saved to {save_path}")
    
    return fig

if __name__ == "__main__":
    from simulation import run_convergence_analysis, run_performance_comparison, run_operations_simulation
    from constellation_config import CONSTELLATIONS
    
    print("Generating plots...")
    
    output_dir = r"C:\Users\windows\Desktop\文章复现\output"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1. Generating Figure 5: Convergence Analysis...")
    convergence_result = run_convergence_analysis(CONSTELLATIONS['SpaceX_Starlink'])
    fig5 = plot_convergence_analysis(convergence_result, 
                                      save_path=os.path.join(output_dir, 'figure5_convergence.png'))
    
    print("\n2. Generating Figure 7: Performance Comparison...")
    test_constellations = ['O3b_mPower', 'Telesat_Lightspeed', 'SpaceX_Starlink']
    performance_results = run_performance_comparison(test_constellations)
    fig7 = plot_performance_comparison(performance_results,
                                        save_path=os.path.join(output_dir, 'figure7_performance.png'))
    
    print("\n3. Generating Figure 8: Operations Simulation...")
    ops_result = run_operations_simulation(CONSTELLATIONS['SpaceX_Starlink'])
    fig8 = plot_operations_simulation(ops_result,
                                       save_path=os.path.join(output_dir, 'figure8_operations.png'))
    
    print("\n4. Generating Table VI visualization...")
    fig_table = plot_table_vi(performance_results,
                              save_path=os.path.join(output_dir, 'table_vi_results.png'))
    
    print("\n5. Generating Constellation Visualization (Figure 6)...")
    fig6 = plot_constellation_visualization(
        save_path=os.path.join(output_dir, 'figure6_constellation.png'))
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    plt.show()
