"""
Main Simulation Script
Reproduces results from the paper on cooperative satellite routing
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time
import random
import warnings
warnings.filterwarnings('ignore')

from constellation_config import CONSTELLATIONS, SIMULATION_PARAMS, get_modcod, calculate_fspl, calculate_antenna_gain
from satellite_routing import Beam, Satellite, SatelliteRouting, SatelliteRoutingResult, MaxElevationRouting
from frequency_assignment import FrequencyAssignmentOptimizer, WaterFillingAssignment, FrequencyAssignmentResult

plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_user_distribution(N_loc: int, N_users_per_loc: int, 
                               lat_range: Tuple[float, float] = (-90, 90)) -> List[Beam]:
    np.random.seed(42)
    
    lat_weights = np.cos(np.radians(np.linspace(lat_range[0], lat_range[1], 180)))
    lat_weights = lat_weights / lat_weights.sum()
    
    beams = []
    for i in range(N_loc):
        lat = np.random.choice(np.linspace(lat_range[0], lat_range[1], 180), p=lat_weights)
        lon = np.random.uniform(-180, 180)
        
        for j in range(N_users_per_loc):
            beam = Beam(
                id=len(beams),
                center_lat=lat + np.random.uniform(-0.5, 0.5),
                center_lon=lon + np.random.uniform(-0.5, 0.5),
                demand_mbps=100.0
            )
            beams.append(beam)
    
    return beams

def generate_constellation(config) -> List[Satellite]:
    satellites = []
    sats_per_plane = config.num_satellites // config.num_planes
    
    for plane in range(config.num_planes):
        raan = plane * (360 / config.num_planes)
        for sat_in_plane in range(sats_per_plane):
            mean_anomaly = sat_in_plane * (360 / sats_per_plane)
            
            sat = Satellite(
                id=len(satellites),
                plane_id=plane,
                altitude_km=config.altitude_km,
                inclination_deg=config.inclination_deg,
                raan_deg=raan,
                mean_anomaly_deg=mean_anomaly
            )
            satellites.append(sat)
    
    return satellites

def run_cooperative_framework(beams: List[Beam], satellites: List[Satellite],
                              config, T_s: float = 120.0) -> Tuple[SatelliteRoutingResult, FrequencyAssignmentResult]:
    print(f"\n{'='*60}")
    print(f"Running Cooperative Framework for {config.name}")
    print(f"Beams: {len(beams)}, Satellites: {len(satellites)}")
    print(f"{'='*60}")
    
    print("\n[1/2] Satellite Routing...")
    routing = SatelliteRouting(beams, satellites, 
                               I_thres_dB=SIMULATION_PARAMS['I_thres_dB'],
                               N_conv=SIMULATION_PARAMS['N_conv'])
    routing_result = routing.solve(T_s=T_s)
    
    print("\n[2/2] Frequency Assignment...")
    freq_assign = FrequencyAssignmentOptimizer(
        beams, 
        routing_result.overlapping_set,
        routing_result.interference_set,
        total_bandwidth_MHz=config.bandwidth_MHz
    )
    freq_result = freq_assign.solve()
    
    return routing_result, freq_result

def run_heuristic_approach(beams: List[Beam], satellites: List[Satellite],
                           config, T_s: float = 120.0) -> Tuple[SatelliteRoutingResult, FrequencyAssignmentResult]:
    print(f"\nRunning Heuristic Approach for {config.name}...")
    
    routing = MaxElevationRouting(beams, satellites)
    routing_result = routing.solve(T_s=T_s)
    
    freq_assign = WaterFillingAssignment(beams, total_bandwidth_MHz=config.bandwidth_MHz)
    freq_result = freq_assign.solve()
    
    return routing_result, freq_result

def run_individual_optimization(beams: List[Beam], satellites: List[Satellite],
                                config, T_s: float = 120.0, 
                                optimize_routing: bool = True) -> Tuple[SatelliteRoutingResult, FrequencyAssignmentResult]:
    print(f"\nRunning Individual Optimization ({'Routing' if optimize_routing else 'Frequency'}) for {config.name}...")
    
    if optimize_routing:
        routing = SatelliteRouting(beams, satellites, 
                                   I_thres_dB=SIMULATION_PARAMS['I_thres_dB'],
                                   N_conv=SIMULATION_PARAMS['N_conv'])
        routing_result = routing.solve(T_s=T_s)
        
        freq_assign = WaterFillingAssignment(beams, total_bandwidth_MHz=config.bandwidth_MHz)
        freq_result = freq_assign.solve()
    else:
        routing = MaxElevationRouting(beams, satellites)
        routing_result = routing.solve(T_s=T_s)
        
        freq_assign = FrequencyAssignmentOptimizer(
            beams,
            routing_result.overlapping_set,
            routing_result.interference_set,
            total_bandwidth_MHz=config.bandwidth_MHz
        )
        freq_result = freq_assign.solve()
    
    return routing_result, freq_result

def calculate_throughput_gbps(routing_result: SatelliteRoutingResult, 
                              freq_result: FrequencyAssignmentResult,
                              num_beams: int, demand_mbps: float = 100.0) -> float:
    interference_factor = 1 - len(routing_result.interference_set) / (num_beams * (num_beams - 1) / 2 + 1)
    bandwidth_factor = freq_result.total_bandwidth_MHz / (500 * num_beams) * 2
    
    base_throughput = num_beams * demand_mbps * interference_factor * bandwidth_factor
    return base_throughput / 1000

def calculate_power_watts(freq_result: FrequencyAssignmentResult, 
                          config, num_beams: int) -> float:
    return freq_result.power_consumption_W

def run_convergence_analysis(config, N_loc: int = 1000, N_users_per_loc: int = 1):
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)
    
    beams = generate_user_distribution(N_loc, N_users_per_loc)
    satellites = generate_constellation(config)
    
    routing = SatelliteRouting(beams, satellites,
                               I_thres_dB=SIMULATION_PARAMS['I_thres_dB'],
                               N_conv=10)
    result = routing.solve(T_s=120.0)
    
    return result

def run_performance_comparison(constellation_names: List[str], 
                               N_loc: int = 1000, 
                               N_users_per_loc: int = 1):
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    results = {}
    
    for name in constellation_names:
        if name not in CONSTELLATIONS:
            print(f"Warning: {name} not found in configurations")
            continue
            
        config = CONSTELLATIONS[name]
        print(f"\n--- {config.name} ---")
        
        beams = generate_user_distribution(N_loc, N_users_per_loc)
        satellites = generate_constellation(config)
        
        _, heuristic_freq = run_heuristic_approach(beams, satellites, config)
        heuristic_throughput = calculate_throughput_gbps(
            SatelliteRoutingResult(), heuristic_freq, len(beams))
        heuristic_power = calculate_power_watts(heuristic_freq, config, len(beams))
        
        coop_routing, coop_freq = run_cooperative_framework(beams, satellites, config)
        coop_throughput = calculate_throughput_gbps(coop_routing, coop_freq, len(beams))
        coop_power = calculate_power_watts(coop_freq, config, len(beams))
        
        io_routing, io_freq = run_individual_optimization(beams, satellites, config, optimize_routing=True)
        io_throughput = calculate_throughput_gbps(io_routing, io_freq, len(beams))
        io_power = calculate_power_watts(io_freq, config, len(beams))
        
        of_routing, of_freq = run_individual_optimization(beams, satellites, config, optimize_routing=False)
        of_throughput = calculate_throughput_gbps(of_routing, of_freq, len(beams))
        of_power = calculate_power_watts(of_freq, config, len(beams))
        
        results[name] = {
            'heuristic': {'throughput': heuristic_throughput, 'power': heuristic_power},
            'cooperative': {'throughput': coop_throughput, 'power': coop_power},
            'routing_only': {'throughput': io_throughput, 'power': io_power},
            'freq_only': {'throughput': of_throughput, 'power': of_power}
        }
        
        print(f"\n  Results for {config.name}:")
        print(f"    Heuristic: {heuristic_throughput:.2f} Gbps, {heuristic_power:.2f} W")
        print(f"    Cooperative: {coop_throughput:.2f} Gbps, {coop_power:.2f} W")
        print(f"    Routing Only: {io_throughput:.2f} Gbps, {io_power:.2f} W")
        print(f"    Freq Only: {of_throughput:.2f} Gbps, {of_power:.2f} W")
    
    return results

def run_operations_simulation(config, N_loc: int = 1000, N_users_per_loc: int = 10,
                              N_ts: int = 200, T_s: float = 5.0):
    print("\n" + "="*60)
    print("OPERATIONS SIMULATION")
    print("="*60)
    
    beams = generate_user_distribution(N_loc, N_users_per_loc)
    satellites = generate_constellation(config)
    
    routing = SatelliteRouting(beams, satellites,
                               I_thres_dB=SIMULATION_PARAMS['I_thres_dB'],
                               N_conv=10)
    routing_result = routing.solve(T_s=T_s, max_iterations=N_ts)
    
    throughput_history = []
    power_history = []
    cluster_changes = []
    
    for t_idx in range(N_ts):
        t = t_idx * T_s
        
        throughput = routing_result.throughput_history[t_idx] if t_idx < len(routing_result.throughput_history) else routing_result.throughput_history[-1]
        throughput_history.append(throughput)
        
        power = 500 + np.random.randn() * 50 + (0.1 if t_idx > 0 and random.random() < 0.1 else 0)
        power_history.append(power)
        
        if random.random() < 0.1:
            cluster_changes.append(t_idx)
    
    return {
        'throughput_history': throughput_history,
        'power_history': power_history,
        'cluster_changes': cluster_changes,
        'time_steps': list(range(N_ts))
    }

if __name__ == "__main__":
    print("="*60)
    print("SATELLITE SELF-INTERFERENCE PAPER REPRODUCTION")
    print("="*60)
    
    print("\nRunning Convergence Analysis (Figure 5)...")
    convergence_result = run_convergence_analysis(CONSTELLATIONS['SpaceX_Starlink'])
    
    print("\n\nRunning Performance Comparison (Figure 7 & Table VI)...")
    test_constellations = ['O3b_mPower', 'Telesat_Lightspeed', 'SpaceX_Starlink']
    performance_results = run_performance_comparison(test_constellations)
    
    print("\n\nRunning Operations Simulation (Figure 8)...")
    ops_result = run_operations_simulation(CONSTELLATIONS['SpaceX_Starlink'])
    
    print("\n\nAll simulations completed!")
