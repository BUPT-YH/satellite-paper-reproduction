"""
Frequency Assignment Implementation
Based on the integer optimization approach for frequency planning
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from itertools import combinations
import random

@dataclass
class FrequencyAssignment:
    beam_id: int
    central_freq_MHz: float
    bandwidth_MHz: float
    reuse_factor: int
    polarization: int  # 0: LHCP, 1: RHCP
    
@dataclass 
class FrequencyAssignmentResult:
    assignments: Dict[int, FrequencyAssignment] = field(default_factory=dict)
    total_bandwidth_MHz: float = 0.0
    power_consumption_W: float = 0.0
    active_beams: int = 0

class FrequencyAssignmentOptimizer:
    def __init__(self, beams: List, overlapping_set: Set[Tuple[int, int]], 
                 interference_set: Set[Tuple[int, int]],
                 total_bandwidth_MHz: float = 500.0,
                 max_reuse: int = 4):
        self.beams = beams
        self.overlapping_set = overlapping_set
        self.interference_set = interference_set
        self.total_bandwidth_MHz = total_bandwidth_MHz
        self.max_reuse = max_reuse
        self.result = FrequencyAssignmentResult()
        
        self.freq_options = self._generate_freq_options()
        
    def _generate_freq_options(self) -> List[Tuple[float, float]]:
        options = []
        step = 50
        for start in np.arange(0, self.total_bandwidth_MHz - 50 + 1, step):
            for bw in [50, 100, 150, 200, 250]:
                if start + bw <= self.total_bandwidth_MHz:
                    options.append((start, bw))
        return options
    
    def _check_spectrum_overlap(self, f1: float, w1: float, f2: float, w2: float) -> bool:
        return (f1 <= f2 + w2) and (f2 <= f1 + w1)
    
    def _compute_power(self, beam, freq_MHz: float, bandwidth_MHz: float, reuse: int) -> float:
        base_power = 0.5
        freq_factor = 1 + (freq_MHz / self.total_bandwidth_MHz) * 0.2
        bw_factor = bandwidth_MHz / 250
        reuse_factor = 1 + 0.3 * (reuse - 1)
        
        return base_power * freq_factor * bw_factor * reuse_factor
    
    def warm_start(self) -> Dict[int, FrequencyAssignment]:
        assignments = {}
        num_beams = len(self.beams)
        
        for i, beam in enumerate(self.beams):
            freq_offset = (i % 4) * (self.total_bandwidth_MHz / 4)
            assignments[beam.id] = FrequencyAssignment(
                beam_id=beam.id,
                central_freq_MHz=freq_offset + 125,
                bandwidth_MHz=250,
                reuse_factor=1,
                polarization=i % 2
            )
        
        return assignments
    
    def solve(self, N_ch: int = 10, N_cutoff: int = 20, N_conv: int = 10) -> FrequencyAssignmentResult:
        print("  Initializing warm start...")
        self.result.assignments = self.warm_start()
        
        print("  Iterative optimization...")
        iter_no_improvement = 0
        previous_objective = float('inf')
        
        while iter_no_improvement < N_conv:
            b0 = random.choice(self.beams)
            neighbors = self._get_neighbors(b0, N_ch)
            
            best_improvement = 0
            best_assignment = None
            
            for beam in neighbors:
                current = self.result.assignments.get(beam.id)
                if current is None:
                    continue
                    
                for freq, bw in random.sample(self.freq_options, min(N_cutoff, len(self.freq_options))):
                    for reuse in range(1, self.max_reuse + 1):
                        for pol in [0, 1]:
                            if self._is_valid_assignment(beam.id, freq, bw, reuse, pol):
                                power = self._compute_power(beam, freq, bw, reuse)
                                current_power = self._compute_power(beam, current.central_freq_MHz, 
                                                                    current.bandwidth_MHz, current.reuse_factor)
                                
                                improvement = current_power - power
                                if improvement > best_improvement:
                                    best_improvement = improvement
                                    best_assignment = FrequencyAssignment(
                                        beam_id=beam.id,
                                        central_freq_MHz=freq,
                                        bandwidth_MHz=bw,
                                        reuse_factor=reuse,
                                        polarization=pol
                                    )
            
            if best_assignment is not None and best_improvement > 0:
                self.result.assignments[best_assignment.beam_id] = best_assignment
                iter_no_improvement = 0
            else:
                iter_no_improvement += 1
        
        self._compute_metrics()
        
        return self.result
    
    def _get_neighbors(self, beam, n: int) -> List:
        beam_pos = np.array([beam.center_lat, beam.center_lon])
        distances = []
        for b in self.beams:
            if b.id != beam.id:
                b_pos = np.array([b.center_lat, b.center_lon])
                dist = np.linalg.norm(beam_pos - b_pos)
                distances.append((dist, b))
        
        distances.sort(key=lambda x: x[0])
        return [b for _, b in distances[:n]]
    
    def _is_valid_assignment(self, beam_id: int, freq: float, bw: float, 
                            reuse: int, pol: int) -> bool:
        assignment = FrequencyAssignment(beam_id, freq, bw, reuse, pol)
        
        for other_id, other in self.result.assignments.items():
            if other_id == beam_id:
                continue
            
            pair = (min(beam_id, other_id), max(beam_id, other_id))
            
            if pair in self.overlapping_set:
                if self._check_spectrum_overlap(freq, bw, other.central_freq_MHz, other.bandwidth_MHz):
                    if reuse == other.reuse_factor and pol == other.polarization:
                        return False
            
            if pair in self.interference_set:
                if self._check_spectrum_overlap(freq, bw, other.central_freq_MHz, other.bandwidth_MHz):
                    if pol == other.polarization:
                        return False
        
        return True
    
    def _compute_metrics(self):
        total_bw = 0
        total_power = 0
        active = 0
        
        for beam_id, assignment in self.result.assignments.items():
            beam = next((b for b in self.beams if b.id == beam_id), None)
            if beam:
                total_bw += assignment.bandwidth_MHz
                total_power += self._compute_power(beam, assignment.central_freq_MHz,
                                                   assignment.bandwidth_MHz, assignment.reuse_factor)
                active += 1
        
        self.result.total_bandwidth_MHz = total_bw
        self.result.power_consumption_W = total_power
        self.result.active_beams = active


class WaterFillingAssignment:
    def __init__(self, beams: List, total_bandwidth_MHz: float = 500.0):
        self.beams = beams
        self.total_bandwidth_MHz = total_bandwidth_MHz
        
    def solve(self) -> FrequencyAssignmentResult:
        result = FrequencyAssignmentResult()
        
        bw_per_beam = self.total_bandwidth_MHz / len(self.beams) * 2
        
        for i, beam in enumerate(self.beams):
            result.assignments[beam.id] = FrequencyAssignment(
                beam_id=beam.id,
                central_freq_MHz=(i % 4) * (self.total_bandwidth_MHz / 4) + 125,
                bandwidth_MHz=bw_per_beam,
                reuse_factor=1,
                polarization=i % 2
            )
        
        result.total_bandwidth_MHz = self.total_bandwidth_MHz * 2
        result.power_consumption_W = len(self.beams) * 0.5
        result.active_beams = len(self.beams)
        
        return result
