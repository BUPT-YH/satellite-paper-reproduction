"""
Constellation Configuration for TVT-2025 Paper
Based on: "Beam Footprint Design, Scheduling, and Spectrum Assignment
in Low Earth Orbit Mega-Constellations"
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ConstellationConfig:
    name: str
    num_satellites: int
    altitude_km: float
    inclination_deg: float
    num_planes: int
    num_beams_per_sat: int
    frequency_GHz: float
    bandwidth_MHz: float
    eirp_dBW: float
    max_power_W: float
    antenna_diameter_m: float
    beam_spacing_deg: float
    min_elevation_deg: float
    service_type: str

CONSTELLATIONS = {
    'O3b_mPower': ConstellationConfig(
        name='O3b mPower',
        num_satellites=11,
        altitude_km=8000,
        inclination_deg=0,
        num_planes=1,
        num_beams_per_sat=3500,
        frequency_GHz=17.8,
        bandwidth_MHz=250,
        eirp_dBW=49.4,
        max_power_W=1000,
        antenna_diameter_m=0.3,
        beam_spacing_deg=0.5,
        min_elevation_deg=25,
        service_type='FSS'
    ),
    'Telesat_Lightspeed': ConstellationConfig(
        name='Telesat Lightspeed',
        num_satellites=298,
        altitude_km=1015,
        inclination_deg=98.98,
        num_planes=27,
        num_beams_per_sat=700,
        frequency_GHz=17.8,
        bandwidth_MHz=250,
        eirp_dBW=47.0,
        max_power_W=500,
        antenna_diameter_m=0.4,
        beam_spacing_deg=0.8,
        min_elevation_deg=25,
        service_type='FSS'
    ),
    'SpaceX_Starlink': ConstellationConfig(
        name='SpaceX Starlink',
        num_satellites=4408,
        altitude_km=550,
        inclination_deg=53,
        num_planes=72,
        num_beams_per_sat=200,
        frequency_GHz=13.5,
        bandwidth_MHz=240,
        eirp_dBW=44.5,
        max_power_W=200,
        antenna_diameter_m=0.25,
        beam_spacing_deg=1.0,
        min_elevation_deg=25,
        service_type='FSS'
    ),
    'OneWeb': ConstellationConfig(
        name='OneWeb',
        num_satellites=648,
        altitude_km=1200,
        inclination_deg=87.9,
        num_planes=18,
        num_beams_per_sat=16,
        frequency_GHz=11.5,
        bandwidth_MHz=250,
        eirp_dBW=42.0,
        max_power_W=100,
        antenna_diameter_m=0.35,
        beam_spacing_deg=1.2,
        min_elevation_deg=25,
        service_type='FSS'
    )
}

# MODCOD table for adaptive coding and modulation
MODCOD_TABLE = [
    {'name': 'QPSK 1/2', 'spectral_eff': 0.89, 'snr_req': 1.0, 'obo': 0.0},
    {'name': 'QPSK 2/3', 'spectral_eff': 1.19, 'snr_req': 3.1, 'obo': 0.0},
    {'name': 'QPSK 3/4', 'spectral_eff': 1.33, 'snr_req': 4.1, 'obo': 0.0},
    {'name': 'QPSK 5/6', 'spectral_eff': 1.48, 'snr_req': 5.2, 'obo': 0.0},
    {'name': '8PSK 2/3', 'spectral_eff': 1.78, 'snr_req': 6.4, 'obo': 0.5},
    {'name': '8PSK 3/4', 'spectral_eff': 2.00, 'snr_req': 7.8, 'obo': 0.5},
    {'name': '8PSK 5/6', 'spectral_eff': 2.22, 'snr_req': 9.2, 'obo': 0.5},
    {'name': '16APSK 2/3', 'spectral_eff': 2.37, 'snr_req': 9.6, 'obo': 1.0},
    {'name': '16APSK 3/4', 'spectral_eff': 2.67, 'snr_req': 10.9, 'obo': 1.0},
    {'name': '16APSK 5/6', 'spectral_eff': 2.96, 'snr_req': 12.2, 'obo': 1.0},
    {'name': '32APSK 3/4', 'spectral_eff': 3.33, 'snr_req': 13.6, 'obo': 2.0},
    {'name': '32APSK 4/5', 'spectral_eff': 3.56, 'snr_req': 14.7, 'obo': 2.0},
    {'name': '32APSK 5/6', 'spectral_eff': 3.70, 'snr_req': 15.6, 'obo': 2.0},
]

# Simulation parameters from paper
SIMULATION_PARAMS = {
    'I_thres_dB': -30,           # Interference threshold
    'N_conv': 10,                 # Convergence iterations
    'N_ts': 10,                   # Time slots
    'T_s': 120,                   # Time slot duration (s)
    'min_elevation_deg': 25,      # Minimum elevation angle
    'rolloff_factor': 0.1,        # Roll-off factor
    'margin_dB': 0.5,             # Link margin
    'atmospheric_loss_dB': 1.0,   # Atmospheric loss
    'waveguide_loss_dB': 0.2,     # Waveguide loss
    'feed_loss_dB': 1.1,          # Feed loss
    'system_temperature_K': 290,  # System temperature
    'boltzmann_constant': 1.38e-23,
    'speed_of_light': 3e8,
    'earth_radius_km': 6371,
}

# User distribution parameters
USER_PARAMS = {
    'N_loc': 1000,               # Number of locations
    'N_users_per_loc': 10,       # Users per location
    'demand_mbps': 100,          # User demand in Mbps
}

def get_modcod(snr_dB: float) -> Dict:
    """Select appropriate MODCOD based on SNR"""
    for modcod in MODCOD_TABLE:
        if snr_dB >= modcod['snr_req']:
            return modcod
    return MODCOD_TABLE[0]

def calculate_fspl(distance_km: float, frequency_GHz: float) -> float:
    """Calculate Free Space Path Loss in dB"""
    frequency_hz = frequency_GHz * 1e9
    distance_m = distance_km * 1000
    wavelength = SIMULATION_PARAMS['speed_of_light'] / frequency_hz
    fspl_dB = 20 * np.log10(4 * np.pi * distance_m / wavelength)
    return fspl_dB

def calculate_antenna_gain(diameter_m: float, frequency_GHz: float,
                          efficiency: float = 0.65) -> float:
    """Calculate antenna gain in dB"""
    frequency_hz = frequency_GHz * 1e9
    wavelength = SIMULATION_PARAMS['speed_of_light'] / frequency_hz
    gain_dB = 10 * np.log10(efficiency * (np.pi * diameter_m / wavelength) ** 2)
    return gain_dB

def calculate_beam_radius(altitude_km: float, half_angle_deg: float) -> float:
    """Calculate beam footprint radius on ground in km"""
    half_angle_rad = np.radians(half_angle_deg)
    radius_km = altitude_km * np.tan(half_angle_rad)
    return radius_km
