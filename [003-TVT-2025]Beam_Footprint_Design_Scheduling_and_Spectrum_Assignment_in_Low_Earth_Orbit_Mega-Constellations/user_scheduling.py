"""
User Scheduling Module
Implements the user scheduling optimization from the TVT-2025 paper
Based on the integer optimization formulation in Section IV
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from itertools import combinations
import random

@dataclass
class User:
    """Represents a user terminal"""
    id: int
    lat: float
    lon: float
    demand_mbps: float = 100.0
    assigned_beam: int = -1
    assigned_time_slot: int = -1

    @property
    def position(self) -> np.ndarray:
        lat_rad = np.radians(self.lat)
        lon_rad = np.radians(self.lon)
        return np.array([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad)
        ])


@dataclass
class ScheduleResult:
    """Result of user scheduling optimization"""
    user_beam_assignment: Dict[int, int] = field(default_factory=dict)
    user_timeslot_assignment: Dict[int, int] = field(default_factory=dict)
    beam_timeslot_usage: Dict[Tuple[int, int], int] = field(default_factory=dict)
    total_served_users: int = 0
    total_throughput_mbps: float = 0.0
    convergence_history: List[float] = field(default_factory=list)


class UserScheduler:
    """
    Implements the user scheduling algorithm
    Based on the integer optimization formulation in equations (13)-(18)
    """

    def __init__(self, users: List[User], beams: List, satellites: List,
                 num_timeslots: int = 10, timeslot_duration_s: float = 120.0,
                 I_thres_dB: float = -30.0):
        self.users = users
        self.beams = beams
        self.satellites = satellites
        self.num_timeslots = num_timeslots
        self.timeslot_duration_s = timeslot_duration_s
        self.I_thres_dB = I_thres_dB
        self.result = ScheduleResult()

        # Pre-compute beam coverage
        self._compute_coverage_matrix()

    def _compute_coverage_matrix(self):
        """Compute which beams cover which users"""
        self.coverage = {}  # beam_id -> set of user_ids

        for beam in self.beams:
            self.coverage[beam.id] = set()
            for user in self.users:
                if self._user_in_beam_coverage(user, beam):
                    self.coverage[beam.id].add(user.id)

    def _user_in_beam_coverage(self, user: User, beam) -> bool:
        """Check if user is within beam coverage"""
        lat_diff = abs(user.lat - beam.center_lat)
        lon_diff = min(abs(user.lon - beam.center_lon),
                       360 - abs(user.lon - beam.center_lon))
        distance_deg = np.sqrt(lat_diff**2 + lon_diff**2)
        distance_km = distance_deg * 111.32
        return distance_km <= beam.radius_km

    def _compute_interference_set(self) -> Set[Tuple[int, int]]:
        """
        Compute interference set R_E as in equation (1)
        Pairs of beams that cause mutual interference
        """
        interference_set = set()

        for b1, b2 in combinations(self.beams, 2):
            # Compute relative antenna gain
            rel_gain = self._compute_relative_gain(b1, b2)

            if rel_gain > self.I_thres_dB:
                interference_set.add((min(b1.id, b2.id), max(b1.id, b2.id)))

        return interference_set

    def _compute_relative_gain(self, beam1, beam2) -> float:
        """Compute relative antenna gain between two beams"""
        lat_diff = beam1.center_lat - beam2.center_lat
        lon_diff = min(abs(beam1.center_lon - beam2.center_lon),
                      360 - abs(beam1.center_lon - beam2.center_lon))
        distance_deg = np.sqrt(lat_diff**2 + lon_diff**2)

        # Simplified antenna pattern model
        beam_spacing = 1.0  # degrees
        if distance_deg < beam_spacing:
            return 0
        elif distance_deg < 2 * beam_spacing:
            return -10
        elif distance_deg < 4 * beam_spacing:
            return -20
        else:
            return -30

    def greedy_scheduling(self) -> ScheduleResult:
        """
        Greedy heuristic for user scheduling
        Fast but suboptimal - used as baseline
        """
        print("  Running greedy scheduling...")

        # Sort users by demand (highest first)
        sorted_users = sorted(self.users, key=lambda u: u.demand_mbps, reverse=True)

        # Track beam usage per timeslot
        beam_ts_usage = {}  # (beam_id, ts) -> num_users

        for user in sorted_users:
            # Find beams that cover this user
            candidate_beams = []
            for beam in self.beams:
                if user.id in self.coverage.get(beam.id, set()):
                    candidate_beams.append(beam)

            if not candidate_beams:
                continue

            # Find best beam-timeslot pair with minimum load
            best_beam = None
            best_ts = 0
            min_load = float('inf')

            for beam in candidate_beams:
                for ts in range(self.num_timeslots):
                    load = beam_ts_usage.get((beam.id, ts), 0)
                    if load < min_load:
                        min_load = load
                        best_beam = beam
                        best_ts = ts

            if best_beam is not None:
                self.result.user_beam_assignment[user.id] = best_beam.id
                self.result.user_timeslot_assignment[user.id] = best_ts
                beam_ts_usage[(best_beam.id, best_ts)] = min_load + 1

        self.result.beam_timeslot_usage = beam_ts_usage
        self.result.total_served_users = len(self.result.user_beam_assignment)

        return self.result

    def optimize_scheduling(self, N_conv: int = 10, max_iterations: int = 1000) -> ScheduleResult:
        """
        Iterative optimization for user scheduling
        Based on Algorithm 1 in the paper
        """
        print("  Initializing scheduling optimization...")

        # Start with greedy solution
        self.greedy_scheduling()

        # Compute interference set
        interference_set = self._compute_interference_set()
        print(f"    Interference set size: {len(interference_set)}")

        # Iterative improvement
        print("  Running iterative optimization...")
        no_improvement = 0
        iteration = 0
        prev_objective = self._compute_objective()

        while no_improvement < N_conv and iteration < max_iterations:
            # Select random user to reschedule
            user = random.choice(self.users)

            if user.id not in self.result.user_beam_assignment:
                iteration += 1
                continue

            # Find alternative beam-timeslot assignment
            current_beam = self.result.user_beam_assignment[user.id]
            current_ts = self.result.user_timeslot_assignment[user.id]

            # Try all feasible alternatives
            best_improvement = 0
            best_assignment = None

            for beam in self.beams:
                if user.id not in self.coverage.get(beam.id, set()):
                    continue

                for ts in range(self.num_timeslots):
                    # Check interference constraints
                    if self._check_interference(user.id, beam.id, ts, interference_set):
                        # Compute objective improvement
                        improvement = self._compute_local_improvement(
                            user, current_beam, current_ts, beam.id, ts)

                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_assignment = (beam.id, ts)

            # Apply improvement if found
            if best_assignment is not None and best_improvement > 0:
                self.result.user_beam_assignment[user.id] = best_assignment[0]
                self.result.user_timeslot_assignment[user.id] = best_assignment[1]
                no_improvement = 0
            else:
                no_improvement += 1

            iteration += 1

            # Track convergence
            current_objective = self._compute_objective()
            self.result.convergence_history.append(current_objective)

            if iteration % 50 == 0:
                print(f"    Iteration {iteration}: objective={current_objective:.2f}")

        self.result.total_served_users = len(self.result.user_beam_assignment)
        print(f"  Scheduling optimization completed in {iteration} iterations")

        return self.result

    def _check_interference(self, user_id: int, beam_id: int, ts: int,
                           interference_set: Set[Tuple[int, int]]) -> bool:
        """Check if assignment violates interference constraints"""
        for (b1, b2) in interference_set:
            if beam_id == b1 or beam_id == b2:
                other_beam = b2 if beam_id == b1 else b1
                # Check if other beam has users in same timeslot
                for other_user_id, other_beam_id in self.result.user_beam_assignment.items():
                    if other_beam_id == other_beam:
                        other_ts = self.result.user_timeslot_assignment.get(other_user_id, -1)
                        if other_ts == ts:
                            return False
        return True

    def _compute_objective(self) -> float:
        """Compute total throughput objective"""
        total = 0.0
        for user in self.users:
            if user.id in self.result.user_beam_assignment:
                total += user.demand_mbps
        return total

    def _compute_local_improvement(self, user: User,
                                   old_beam: int, old_ts: int,
                                   new_beam: int, new_ts: int) -> float:
        """Compute improvement from changing user assignment"""
        # Prefer assignments to less loaded beam-timeslot pairs
        old_load = self.result.beam_timeslot_usage.get((old_beam, old_ts), 0)
        new_load = self.result.beam_timeslot_usage.get((new_beam, new_ts), 0)
        load_improvement = (old_load - new_load) * user.demand_mbps / 1000
        return load_improvement
