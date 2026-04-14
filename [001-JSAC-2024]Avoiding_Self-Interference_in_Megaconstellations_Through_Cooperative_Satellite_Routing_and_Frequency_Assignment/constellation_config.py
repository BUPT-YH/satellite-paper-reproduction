"""
Constellation Configuration for Satellite Self-Interference Paper
Based on: "Avoiding Self-Interference in Megaconstellations Through 
Cooperative Satellite Routing and Frequency Assignment"
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
    eirp_density_dBW_Hz: float
    max_power_per_beam_W: float
    antenna_diameter_m: float
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
        eirp_density_dBW_Hz=-5.5,
        max_power_per_beam_W=0.5,
        antenna_diameter_m=0.3,
        service_type='FSS'
    ),
    'ViaSat_LEO': ConstellationConfig(
        name='ViaSat LEO',
        num_satellites=288,
        altitude_km=1300,
        inclination_deg=85,
        num_planes=18,
        num_beams_per_sat=1000,
        frequency_GHz=19.5,
        bandwidth_MHz=500,
        eirp_density_dBW_Hz=-3.5,
        max_power_per_beam_W=2.0,
        antenna_diameter_m=0.5,
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
        eirp_density_dBW_Hz=-4.5,
        max_power_per_beam_W=1.0,
        antenna_diameter_m=0.4,
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
        eirp_density_dBW_Hz=-6.0,
        max_power_per_beam_W=0.5,
        antenna_diameter_m=0.25,
        service_type='FSS'
    ),
    'SpaceX_Starlink_MSS': ConstellationConfig(
        name='SpaceX Starlink MSS',
        num_satellites=4408,
        altitude_km=550,
        inclination_deg=53,
        num_planes=72,
        num_beams_per_sat=200,
        frequency_GHz=1.9,
        bandwidth_MHz=5,
        eirp_density_dBW_Hz=-8.0,
        max_power_per_beam_W=0.2,
        antenna_diameter_m=0.0,
        service_type='MSS'
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
        eirp_density_dBW_Hz=-5.0,
        max_power_per_beam_W=0.3,
        antenna_diameter_m=0.35,
        service_type='FSS'
    )
}

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

SIMULATION_PARAMS = {
    'I_thres_dB': -30,
    'N_conv': 10,
    'N_ts': 10,
    'T_s': 120,
    'min_elevation_deg': 25,
    'rolloff_factor': 0.1,
    'margin_dB': 0.5,
    'atmospheric_loss_dB': 1.0,
    'waveguide_loss_dB': 0.2,
    'feed_loss_dB': 1.1,
    'system_temperature_K': 290,
    'boltzmann_constant': 1.38e-23,
    'speed_of_light': 3e8,
}

def get_modcod(spectral_eff_required: float) -> Dict:
    for modcod in MODCOD_TABLE:
        if modcod['spectral_eff'] >= spectral_eff_required:
            return modcod
    return MODCOD_TABLE[-1]

def calculate_fspl(distance_m: float, frequency_hz: float) -> float:
    wavelength = SIMULATION_PARAMS['speed_of_light'] / frequency_hz
    fspl_dB = 20 * np.log10(4 * np.pi * distance_m / wavelength)
    return fspl_dB

def calculate_antenna_gain(diameter_m: float, frequency_hz: float, efficiency: float = 0.65) -> float:
    wavelength = SIMULATION_PARAMS['speed_of_light'] / frequency_hz
    gain_dB = 10 * np.log10(efficiency * (np.pi * diameter_m / wavelength) ** 2)
    return gain_dB
