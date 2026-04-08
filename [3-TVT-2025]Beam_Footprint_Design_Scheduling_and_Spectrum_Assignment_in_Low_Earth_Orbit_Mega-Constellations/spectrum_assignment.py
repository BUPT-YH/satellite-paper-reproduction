"""
Spectrum Assignment Module
Implements the spectrum assignment optimization from the TVT-2025 paper
Based on the integer optimization formulation in Section V
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from itertools import combinations
import random

@dataclass
class SpectrumAllocation:
    """Spectrum allocation for a beam"""
    beam_id: int
    start_freq_MHz: float
    bandwidth_MHz: float
    frequency_reuse: int
    polarization: int  # 0: LHCP, 1: RHCP, 2: Both

    def overlaps(self, other: 'SpectrumAllocation') -> bool:
        """Check if two allocations overlap in frequency"""
        return (self.start_freq_MHz < other.start_freq_MHz + other.bandwidth_MHz and
                other.start_freq_MHz < self.start_freq_MHz + self.bandwidth_MHz)


@dataclass
class SpectrumAssignmentResult:
    """Result of spectrum assignment optimization"""
    allocations: Dict[int, SpectrumAllocation] = field(default_factory=dict)
    total_spectrum_MHz: float = 0.0
    spectrum_efficiency: float = 0.0
    active_beams: int = 0
    interference_pairs: int = 0
    convergence_history: List[float] = field(default_factory=list)


class SpectrumAssigner:
    """
    Implements the spectrum assignment algorithm
    Based on the integer optimization formulation in equations (19)-(25)
    """

    def __init__(self, beams: List, interference_set: Set[Tuple[int, int]],
                 total_bandwidth_MHz: float = 500.0,
                 max_reuse: int = 4,
                 polarizations: int = 2):
        self.beams = beams
        self.interference_set = interference_set
        self.total_bandwidth_MHz = total_bandwidth_MHz
        self.max_reuse = max_reuse
        self.polarizations = polarizations
        self.result = SpectrumAssignmentResult()

        # Generate frequency options
        self.freq_options = self._generate_frequency_options()

    def _generate_frequency_options(self) -> List[Tuple[float, float]]:
        """Generate possible (start_freq, bandwidth) pairs"""
        options = []
        bandwidth_options = [50, 100, 150, 200, 250]

        for bw in bandwidth_options:
            step = bw // 2  # Overlapping channels
            for start in np.arange(0, self.total_bandwidth_MHz - bw + 1, step):
                options.append((start, bw))

        return options

    def water_filling_baseline(self) -> SpectrumAssignmentResult:
        """
        Water-filling baseline for comparison
        Uniform bandwidth allocation
        """
        print("  Running water-filling baseline...")

        num_beams = len(self.beams)
        bw_per_beam = self.total_bandwidth_MHz / num_beams * 4  # Allow reuse

        for i, beam in enumerate(self.beams):
            freq_offset = (i % 4) * (self.total_bandwidth_MHz / 4)

            allocation = SpectrumAllocation(
                beam_id=beam.id,
                start_freq_MHz=freq_offset,
                bandwidth_MHz=bw_per_beam,
                frequency_reuse=1,
                polarization=i % self.polarizations
            )
            self.result.allocations[beam.id] = allocation

        self.result.active_beams = num_beams
        self.result.total_spectrum_MHz = self.total_bandwidth_MHz

        return self.result

    def optimize_spectrum(self, N_conv: int = 10, max_iterations: int = 1000) -> SpectrumAssignmentResult:
        """
        Iterative optimization for spectrum assignment
        Based on Algorithm 2 in the paper
        """
        print("  Initializing spectrum optimization...")

        # Start with warm start
        self._warm_start()

        print("  Running iterative optimization...")
        no_improvement = 0
        iteration = 0

        while no_improvement < N_conv and iteration < max_iterations:
            # Select random beam to optimize
            beam = random.choice(self.beams)

            # Try to find better allocation
            current_alloc = self.result.allocations.get(beam.id)
            if current_alloc is None:
                iteration += 1
                continue

            best_improvement = 0
            best_alloc = None

            # Try different frequency and bandwidth options
            for freq, bw in random.sample(self.freq_options,
                                          min(20, len(self.freq_options))):
                for reuse in range(1, self.max_reuse + 1):
                    for pol in range(self.polarizations):
                        new_alloc = SpectrumAllocation(
                            beam_id=beam.id,
                            start_freq_MHz=freq,
                            bandwidth_MHz=bw,
                            frequency_reuse=reuse,
                            polarization=pol
                        )

                        # Check feasibility
                        if self._is_feasible(new_alloc):
                            # Compute improvement
                            improvement = self._compute_improvement(
                                current_alloc, new_alloc)

                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_alloc = new_alloc

            # Apply improvement if found
            if best_alloc is not None and best_improvement > 0:
                self.result.allocations[beam.id] = best_alloc
                no_improvement = 0
            else:
                no_improvement += 1

            iteration += 1

            # Track convergence
            objective = self._compute_objective()
            self.result.convergence_history.append(objective)

            if iteration % 50 == 0:
                print(f"    Iteration {iteration}: objective={objective:.2f}")

        # Compute final metrics
        self._compute_metrics()

        print(f"  Spectrum optimization completed in {iteration} iterations")
        return self.result

    def _warm_start(self) -> None:
        """Initialize with greedy frequency assignment"""
        for i, beam in enumerate(self.beams):
            # Divide spectrum into blocks
            block_size = self.total_bandwidth_MHz / 4

            allocation = SpectrumAllocation(
                beam_id=beam.id,
                start_freq_MHz=(i % 4) * block_size,
                bandwidth_MHz=self.total_bandwidth_MHz / 8,
                frequency_reuse=1,
                polarization=i % self.polarizations
            )
            self.result.allocations[beam.id] = allocation

    def _is_feasible(self, new_alloc: SpectrumAllocation) -> bool:
        """Check if allocation satisfies interference constraints"""
        for (b1, b2) in self.interference_set:
            if new_alloc.beam_id == b1 or new_alloc.beam_id == b2:
                other_id = b2 if new_alloc.beam_id == b1 else b1
                other_alloc = self.result.allocations.get(other_id)

                if other_alloc is not None:
                    # Check frequency overlap
                    if new_alloc.overlaps(other_alloc):
                        # Same polarization and reuse causes interference
                        if (new_alloc.polarization == other_alloc.polarization and
                            new_alloc.frequency_reuse == other_alloc.frequency_reuse):
                            return False
                        # Different polarization provides ~20dB isolation
                        elif new_alloc.polarization != other_alloc.polarization:
                            # Still need frequency separation
                            pass

        return True

    def _compute_improvement(self, old_alloc: SpectrumAllocation,
                            new_alloc: SpectrumAllocation) -> float:
        """Compute improvement in spectrum efficiency"""
        # Higher bandwidth is better
        bw_improvement = (new_alloc.bandwidth_MHz - old_alloc.bandwidth_MHz) / 100

        # Lower reuse factor is better (more capacity)
        reuse_improvement = (old_alloc.frequency_reuse - new_alloc.frequency_reuse) * 0.1

        return bw_improvement + reuse_improvement

    def _compute_objective(self) -> float:
        """Compute total spectrum efficiency objective"""
        total_efficiency = 0.0

        for beam_id, alloc in self.result.allocations.items():
            # Efficiency = bandwidth / reuse_factor
            efficiency = alloc.bandwidth_MHz / alloc.frequency_reuse
            total_efficiency += efficiency

        return total_efficiency

    def _compute_metrics(self) -> None:
        """Compute final assignment metrics"""
        total_bw = 0
        active = 0

        for beam_id, alloc in self.result.allocations.items():
            effective_bw = alloc.bandwidth_MHz / max(alloc.frequency_reuse, 1)
            total_bw += effective_bw
            active += 1

        # Effective spectrum with frequency reuse
        self.result.total_spectrum_MHz = total_bw
        self.result.active_beams = active
        self.result.spectrum_efficiency = (
            total_bw / (self.total_bandwidth_MHz * self.max_reuse)
            if self.total_bandwidth_MHz > 0 else 0
        )

        # Count interference pairs
        interference_count = 0
        for (b1, b2) in self.interference_set:
            a1 = self.result.allocations.get(b1)
            a2 = self.result.allocations.get(b2)
            if a1 and a2 and a1.overlaps(a2):
                if a1.polarization == a2.polarization:
                    interference_count += 1

        self.result.interference_pairs = interference_count


class FrequencyReuseOptimizer:
    """
    Optimizes frequency reuse pattern
    Based on the four-color theorem approach
    """

    def __init__(self, beams: List, interference_set: Set[Tuple[int, int]],
                 num_colors: int = 4):
        self.beams = beams
        self.interference_set = interference_set
        self.num_colors = num_colors

    def assign_colors(self) -> Dict[int, int]:
        """
        Assign colors (frequency channels) to beams
        Using greedy graph coloring
        """
        colors = {}

        # Sort beams by degree in interference graph
        beam_degree = {}
        for beam in self.beams:
            degree = sum(1 for (b1, b2) in self.interference_set
                        if b1 == beam.id or b2 == beam.id)
            beam_degree[beam.id] = degree

        sorted_beams = sorted(self.beams, key=lambda b: beam_degree.get(b.id, 0),
                             reverse=True)

        for beam in sorted_beams:
            # Find colors used by neighbors
            neighbor_colors = set()
            for (b1, b2) in self.interference_set:
                if b1 == beam.id:
                    neighbor_colors.add(colors.get(b2, -1))
                elif b2 == beam.id:
                    neighbor_colors.add(colors.get(b1, -1))

            # Assign lowest available color
            for color in range(self.num_colors):
                if color not in neighbor_colors:
                    colors[beam.id] = color
                    break
            else:
                # No color available, use random
                colors[beam.id] = random.randint(0, self.num_colors - 1)

        return colors
