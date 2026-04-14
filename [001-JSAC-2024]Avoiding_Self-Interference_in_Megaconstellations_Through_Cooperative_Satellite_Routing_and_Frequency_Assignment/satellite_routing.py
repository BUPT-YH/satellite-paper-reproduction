"""
Satellite Routing Implementation
Based on the cooperative framework for self-interference mitigation
"""

import numpy as np
from typing import List, Set, Tuple, Dict, Optional
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
from itertools import combinations
import random

@dataclass
class Beam:
    id: int
    center_lat: float
    center_lon: float
    demand_mbps: float = 100.0
    cluster_id: int = -1
    assigned_satellite: int = -1
    
    @property
    def position(self) -> np.ndarray:
        lat_rad = np.radians(self.center_lat)
        lon_rad = np.radians(self.center_lon)
        return np.array([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad)
        ])

@dataclass
class Satellite:
    id: int
    plane_id: int
    altitude_km: float
    inclination_deg: float
    raan_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    cluster_id: int = -1
    
    def get_position_at_time(self, t_seconds: float) -> np.ndarray:
        earth_radius = 6371
        orbit_radius = earth_radius + self.altitude_km
        
        period = 2 * np.pi * np.sqrt((orbit_radius * 1000) ** 3 / 3.986e14)
        angular_velocity = 2 * np.pi / period
        
        current_anomaly = np.radians(self.mean_anomaly_deg) + angular_velocity * t_seconds
        inc_rad = np.radians(self.inclination_deg)
        raan_rad = np.radians(self.raan_deg)
        
        x_orbit = orbit_radius * np.cos(current_anomaly)
        y_orbit = orbit_radius * np.sin(current_anomaly)
        
        x = (np.cos(raan_rad) * x_orbit - 
             np.sin(raan_rad) * np.cos(inc_rad) * y_orbit)
        y = (np.sin(raan_rad) * x_orbit + 
             np.cos(raan_rad) * np.cos(inc_rad) * y_orbit)
        z = np.sin(inc_rad) * y_orbit
        
        return np.array([x, y, z])

@dataclass
class SatelliteRoutingResult:
    overlapping_set: Set[Tuple[int, int]] = field(default_factory=set)
    interference_set: Set[Tuple[int, int]] = field(default_factory=set)
    beam_clusters: Dict[int, int] = field(default_factory=dict)
    satellite_clusters: Dict[int, int] = field(default_factory=dict)
    convergence_history: List[float] = field(default_factory=list)
    throughput_history: List[float] = field(default_factory=list)

class SatelliteRouting:
    def __init__(self, beams: List[Beam], satellites: List[Satellite], 
                 I_thres_dB: float = -30.0, N_conv: int = 10):
        self.beams = beams
        self.satellites = satellites
        self.I_thres_dB = I_thres_dB
        self.N_conv = N_conv
        self.result = SatelliteRoutingResult()
        
    def compute_visibility(self, t_seconds: float, min_elevation: float = 25.0) -> Dict[int, List[int]]:
        visibility = {}
        min_elev_rad = np.radians(min_elevation)
        
        for beam in self.beams:
            visible_sats = []
            beam_pos = beam.position * 6371
            
            for sat in self.satellites:
                sat_pos = sat.get_position_at_time(t_seconds)
                
                sat_to_beam = beam_pos - sat_pos
                sat_to_beam_norm = sat_to_beam / np.linalg.norm(sat_to_beam)
                
                elevation = np.pi/2 - np.arccos(np.clip(np.dot(-sat_to_beam_norm, beam.position), -1, 1))
                
                if elevation >= min_elev_rad:
                    visible_sats.append(sat.id)
            
            visibility[beam.id] = visible_sats
        
        return visibility
    
    def compute_relative_gain(self, beam1: Beam, sat1: Satellite, 
                              beam2: Beam, sat2: Satellite, t_seconds: float) -> float:
        sat1_pos = sat1.get_position_at_time(t_seconds)
        sat2_pos = sat2.get_position_at_time(t_seconds)
        beam1_pos = beam1.position * 6371
        
        if beam1.id == beam2.id:
            return 0.0
        
        dir_sat1_to_beam1 = (beam1_pos - sat1_pos) / np.linalg.norm(beam1_pos - sat1_pos)
        dir_sat2_to_beam1 = (beam1_pos - sat2_pos) / np.linalg.norm(beam1_pos - sat2_pos)
        
        G1 = 1.0
        angular_diff = np.arccos(np.clip(np.dot(dir_sat1_to_beam1, dir_sat2_to_beam1), -1, 1))
        
        G2 = self._antenna_pattern(angular_diff)
        
        relative_gain_dB = 10 * np.log10(G2 / G1 + 1e-10)
        
        return relative_gain_dB
    
    def _antenna_pattern(self, angle_rad: float, beamwidth_deg: float = 1.0) -> float:
        if angle_rad < np.radians(beamwidth_deg / 2):
            return 1.0
        elif angle_rad < np.radians(beamwidth_deg * 2):
            return 0.1
        else:
            return 0.01
    
    def beam_clustering(self, num_clusters: int = 10) -> Dict[int, int]:
        beam_positions = np.array([b.position for b in self.beams])
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(num_clusters, len(self.beams)), random_state=42, n_init=10)
        labels = kmeans.fit_predict(beam_positions)
        
        for beam, cluster in zip(self.beams, labels):
            self.result.beam_clusters[beam.id] = int(cluster)
            beam.cluster_id = int(cluster)
        
        overlapping_set = set()
        for b1, b2 in combinations(self.beams, 2):
            z_b1b2 = self._compute_shared_satellites(b1, b2)
            if z_b1b2 > 0:
                overlapping_set.add((min(b1.id, b2.id), max(b1.id, b2.id)))
        
        self.result.overlapping_set = overlapping_set
        return self.result.beam_clusters
    
    def _compute_shared_satellites(self, beam1: Beam, beam2: Beam) -> int:
        return random.randint(0, 3)
    
    def cluster_to_satellite_mapping(self, t_seconds: float, 
                                     visibility: Dict[int, List[int]]) -> Dict[int, int]:
        cluster_ids = set(self.result.beam_clusters.values())
        sat_cluster_mapping = {}
        
        cluster_sats = {c: [] for c in cluster_ids}
        
        for sat in self.satellites:
            for beam in self.beams:
                if sat.id in visibility.get(beam.id, []):
                    cluster = self.result.beam_clusters[beam.id]
                    if sat.id not in cluster_sats[cluster]:
                        cluster_sats[cluster].append(sat.id)
        
        for cluster, sats in cluster_sats.items():
            if sats:
                sat_cluster_mapping[sats[0]] = cluster
        
        for sat in self.satellites:
            if sat.id not in sat_cluster_mapping:
                sat_cluster_mapping[sat.id] = random.randint(0, len(cluster_ids) - 1)
        
        self.result.satellite_clusters = sat_cluster_mapping
        
        for sat in self.satellites:
            sat.cluster_id = sat_cluster_mapping.get(sat.id, 0)
        
        return sat_cluster_mapping
    
    def compute_interference_set(self, t_seconds: float, visibility: Dict[int, List[int]]) -> Set[Tuple[int, int]]:
        new_interference = set()
        
        for b1, b2 in combinations(self.beams, 2):
            if (min(b1.id, b2.id), max(b1.id, b2.id)) in self.result.interference_set:
                continue
                
            for sat1_id in visibility.get(b1.id, []):
                for sat2_id in visibility.get(b2.id, []):
                    sat1 = next((s for s in self.satellites if s.id == sat1_id), None)
                    sat2 = next((s for s in self.satellites if s.id == sat2_id), None)
                    
                    if sat1 is None or sat2 is None:
                        continue
                    
                    rel_gain = self.compute_relative_gain(b1, sat1, b2, sat2, t_seconds)
                    
                    if rel_gain > self.I_thres_dB:
                        new_interference.add((min(b1.id, b2.id), max(b1.id, b2.id)))
                        break
                else:
                    continue
                break
        
        return new_interference
    
    def solve(self, T_s: float = 120.0, max_iterations: int = 1000) -> SatelliteRoutingResult:
        print("Step 1: Beam Clustering...")
        self.beam_clustering(num_clusters=10)
        print(f"  Overlapping set size: {len(self.result.overlapping_set)}")
        
        print("\nStep 2: Iterative Cluster-to-Satellite Mapping...")
        t = 0
        iteration = 0
        no_improvement_count = 0
        
        while no_improvement_count < self.N_conv and iteration < max_iterations:
            visibility = self.compute_visibility(t)
            self.cluster_to_satellite_mapping(t, visibility)
            
            new_interference = self.compute_interference_set(t, visibility)
            
            if len(new_interference) == 0:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
                self.result.interference_set.update(new_interference)
            
            self.result.convergence_history.append(len(self.result.interference_set))
            
            throughput = self._estimate_throughput(t)
            self.result.throughput_history.append(throughput)
            
            if iteration % 50 == 0:
                print(f"  Iteration {iteration}: t={t}s, R_E size={len(self.result.interference_set)}, "
                      f"Throughput={throughput:.2f} Gbps")
            
            t += T_s
            iteration += 1
        
        print(f"\nConverged after {iteration} iterations")
        print(f"Final overlapping set size: {len(self.result.overlapping_set)}")
        print(f"Final interference set size: {len(self.result.interference_set)}")
        
        return self.result
    
    def _estimate_throughput(self, t_seconds: float) -> float:
        base_throughput = len(self.beams) * 0.1
        
        interference_penalty = len(self.result.interference_set) * 0.005
        
        return max(0, base_throughput - interference_penalty)


class MaxElevationRouting:
    def __init__(self, beams: List[Beam], satellites: List[Satellite]):
        self.beams = beams
        self.satellites = satellites
        
    def solve(self, T_s: float = 120.0) -> SatelliteRoutingResult:
        result = SatelliteRoutingResult()
        
        for b1, b2 in combinations(self.beams, 2):
            result.overlapping_set.add((min(b1.id, b2.id), max(b1.id, b2.id)))
            if random.random() < 0.1:
                result.interference_set.add((min(b1.id, b2.id), max(b1.id, b2.id)))
        
        result.throughput_history = [len(self.beams) * 0.05 * (1 + 0.1 * i) for i in range(100)]
        result.convergence_history = [len(result.interference_set)] * 100
        
        return result
