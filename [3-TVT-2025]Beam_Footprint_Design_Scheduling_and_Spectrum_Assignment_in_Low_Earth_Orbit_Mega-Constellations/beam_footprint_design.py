"""
Beam Footprint Design Module
Implements the beam footprint optimization from the TVT-2025 paper
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from itertools import combinations
import random

@dataclass
class BeamFootprint:
    """Represents a beam footprint on ground"""
    id: int
    center_lat: float
    center_lon: float
    radius_km: float
    frequency_GHz: float
    bandwidth_MHz: float
    power_W: float
    satellite_id: int = -1
    user_ids: List[int] = field(default_factory=list)
    demand_mbps: float = 100.0
    throughput_mbps: float = 0.0

    @property
    def position(self) -> np.ndarray:
        """Get 3D position on Earth surface"""
        lat_rad = np.radians(self.center_lat)
        lon_rad = np.radians(self.center_lon)
        return np.array([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad)
        ])

    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if a point is within the beam footprint"""
        lat_diff = abs(self.center_lat - lat)
        lon_diff = min(abs(self.center_lon - lon),
                       360 - abs(self.center_lon - lon))
        distance_deg = np.sqrt(lat_diff**2 + lon_diff**2)
        distance_km = distance_deg * 111.32  # Approximate km per degree
        return distance_km <= self.radius_km


@dataclass
class Satellite:
    """Represents a LEO satellite"""
    id: int
    plane_id: int
    altitude_km: float
    inclination_deg: float
    raan_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    num_beams: int = 10
    max_power_W: float = 100.0
    bandwidth_MHz: float = 250.0

    def get_position_at_time(self, t_seconds: float) -> np.ndarray:
        """Calculate satellite position at given time"""
        earth_radius = 6371
        orbit_radius = earth_radius + self.altitude_km

        # Orbital period using Kepler's third law
        period = 2 * np.pi * np.sqrt((orbit_radius * 1000) ** 3 / 3.986e14)
        angular_velocity = 2 * np.pi / period

        # Current position in orbit
        current_anomaly = np.radians(self.mean_anomaly_deg) + angular_velocity * t_seconds
        inc_rad = np.radians(self.inclination_deg)
        raan_rad = np.radians(self.raan_deg)

        # Position in orbital plane
        x_orbit = orbit_radius * np.cos(current_anomaly)
        y_orbit = orbit_radius * np.sin(current_anomaly)

        # Transform to ECEF coordinates
        x = (np.cos(raan_rad) * x_orbit -
             np.sin(raan_rad) * np.cos(inc_rad) * y_orbit)
        y = (np.sin(raan_rad) * x_orbit +
             np.cos(raan_rad) * np.cos(inc_rad) * y_orbit)
        z = np.sin(inc_rad) * y_orbit

        return np.array([x, y, z])

    def get_coverage_at_time(self, t_seconds: float,
                            min_elevation: float = 25.0) -> Tuple[float, float, float]:
        """Get coverage cone center position at given time"""
        pos = self.get_position_at_time(t_seconds)
        lat = np.degrees(np.arcsin(pos[2] / np.linalg.norm(pos)))
        lon = np.degrees(np.arctan2(pos[1], pos[0]))
        return lat, lon, self.altitude_km


class BeamFootprintDesign:
    """
    Implements the beam footprint design algorithm
    Based on the optimization framework in Section III of the paper
    """

    def __init__(self, config, user_locations: List[Tuple[float, float]],
                 I_thres_dB: float = -30.0):
        self.config = config
        self.user_locations = user_locations
        self.I_thres_dB = I_thres_dB
        self.beams: List[BeamFootprint] = []
        self.satellites: List[Satellite] = []

        # Initialize satellites
        self._initialize_satellites()

    def _initialize_satellites(self):
        """Create satellite constellation"""
        sats_per_plane = self.config.num_satellites // self.config.num_planes

        for plane in range(self.config.num_planes):
            raan = plane * (360 / self.config.num_planes)
            for sat_in_plane in range(sats_per_plane):
                mean_anomaly = sat_in_plane * (360 / sats_per_plane)

                sat = Satellite(
                    id=len(self.satellites),
                    plane_id=plane,
                    altitude_km=self.config.altitude_km,
                    inclination_deg=self.config.inclination_deg,
                    raan_deg=raan,
                    mean_anomaly_deg=mean_anomaly,
                    num_beams=self.config.num_beams_per_sat,
                    max_power_W=self.config.max_power_W,
                    bandwidth_MHz=self.config.bandwidth_MHz
                )
                self.satellites.append(sat)

    def design_beam_footprints(self, num_beams: int = 100) -> List[BeamFootprint]:
        """
        Design beam footprints to cover user locations
        Uses clustering-based approach as described in the paper
        """
        from sklearn.cluster import KMeans

        # Cluster user locations
        user_coords = np.array([[lat, lon] for lat, lon in self.user_locations])

        n_clusters = min(num_beams, len(self.user_locations))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(user_coords)

        # Create beam footprints at cluster centers
        beam_centers = kmeans.cluster_centers_

        self.beams = []
        for i, center in enumerate(beam_centers):
            # Calculate beam radius based on actual cluster spread
            cluster_mask = labels == i
            cluster_points = user_coords[cluster_mask]
            if len(cluster_points) > 1:
                distances = np.sqrt(np.sum((cluster_points - center) ** 2, axis=1))
                radius_deg = max(np.max(distances) * 1.2, 0.3)
            else:
                radius_deg = 0.5
            radius_km = radius_deg * 111.32

            beam = BeamFootprint(
                id=i,
                center_lat=center[0],
                center_lon=center[1],
                radius_km=radius_km,
                frequency_GHz=self.config.frequency_GHz,
                bandwidth_MHz=self.config.bandwidth_MHz,
                power_W=self.config.max_power_W / self.config.num_beams_per_sat
            )
            self.beams.append(beam)

        # Assign users to beams
        for user_idx, label in enumerate(labels):
            self.beams[label].user_ids.append(user_idx)

        return self.beams

    def compute_interference_set(self, t_seconds: float = 0) -> set:
        """
        Compute the interference set R_E as defined in equation (1)
        R_E = {(b_i, b_j) : G_ij > I_thres}
        """
        interference_set = set()

        for b1, b2 in combinations(self.beams, 2):
            rel_gain = self._compute_relative_gain(b1, b2, t_seconds)
            if rel_gain > self.I_thres_dB:
                interference_set.add((min(b1.id, b2.id), max(b1.id, b2.id)))

        return interference_set

    def _compute_relative_gain(self, beam1: BeamFootprint, beam2: BeamFootprint,
                               t_seconds: float) -> float:
        """
        Compute relative antenna gain G_ij between two beams
        Based on antenna pattern model in the paper
        """
        # Distance between beam centers
        lat_diff = beam1.center_lat - beam2.center_lat
        lon_diff = min(abs(beam1.center_lon - beam2.center_lon),
                      360 - abs(beam1.center_lon - beam2.center_lon))
        distance_deg = np.sqrt(lat_diff**2 + lon_diff**2)

        # Angular separation in satellite antenna pattern
        angular_sep = distance_deg

        # Simplified antenna pattern (could use more accurate model)
        if angular_sep < self.config.beam_spacing_deg:
            return 0  # Maximum gain (co-channel interference)
        elif angular_sep < 2 * self.config.beam_spacing_deg:
            return -10  # First sidelobe
        elif angular_sep < 4 * self.config.beam_spacing_deg:
            return -20  # Second sidelobe
        else:
            return -30  # Beyond main beam

    def optimize_beam_power(self) -> None:
        """
        Optimize power allocation to beams using water-filling
        Based on equations (10)-(12) in the paper
        """
        # Total power budget: one serving satellite's worth
        total_power = self.config.max_power_W

        # Calculate channel quality for each beam
        channel_qualities = []
        for beam in self.beams:
            num_users = len(beam.user_ids)
            if num_users > 0:
                quality = num_users / max(1, len(self.user_locations)) * 100
            else:
                quality = 0.1
            channel_qualities.append(quality)

        # Water-filling power allocation
        qualities = np.array(channel_qualities)
        total_quality = np.sum(qualities)

        for i, beam in enumerate(self.beams):
            if total_quality > 0:
                allocated_power = total_power * (qualities[i] / total_quality)
            else:
                allocated_power = total_power / len(self.beams)
            beam.power_W = min(allocated_power, self.config.max_power_W)

    def compute_throughput(self, beam: BeamFootprint, snr_dB: float) -> float:
        """
        Compute achievable throughput for a beam
        Using MODCOD selection based on SNR
        """
        from constellation_config import get_modcod

        modcod = get_modcod(snr_dB)
        spectral_eff = modcod['spectral_eff']
        effective_bw = beam.bandwidth_MHz * (1 - SIMULATION_PARAMS.get('rolloff_factor', 0.1))

        throughput_mbps = spectral_eff * effective_bw
        return throughput_mbps


# Import SIMULATION_PARAMS
from constellation_config import SIMULATION_PARAMS
