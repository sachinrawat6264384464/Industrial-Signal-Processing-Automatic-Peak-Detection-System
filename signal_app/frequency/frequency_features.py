"""
Frequency-Domain Feature Extraction Module.
Calculates Spectral Centroid, Bandwidth, Roll-off, Total Power, and Peak Frequency.
"""
import numpy as np
from typing import Dict, Any
from .fft_analysis import compute_fft

def extract_frequency_features(signal: np.ndarray, sampling_rate: float = 50000.0) -> Dict[str, float]:
    """
    Extracts quantitative spectral features from a 1D signal window.

    Returns dict containing:
        'peak_frequency': Frequency with maximum spectral magnitude (Hz)
        'spectral_centroid': Center of mass of the spectrum (Hz)
        'spectral_bandwidth': Spectral width around centroid (Hz)
        'spectral_rolloff': Frequency below which 85% of spectral energy lies (Hz)
        'total_power': Sum of power spectral density
    """
    if signal is None or len(signal) == 0:
        return {
            "peak_frequency": 0.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_rolloff": 0.0,
            "total_power": 0.0
        }

    freqs, mag, psd = compute_fft(signal, sampling_rate=sampling_rate)
    if len(freqs) == 0 or np.sum(mag) == 0:
        return {
            "peak_frequency": 0.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_rolloff": 0.0,
            "total_power": 0.0
        }

    # Peak Frequency
    peak_idx = np.argmax(mag[1:]) + 1 if len(mag) > 1 else 0
    peak_freq = float(freqs[peak_idx])

    # Spectral Centroid: sum(f * mag) / sum(mag)
    sum_mag = np.sum(mag)
    centroid = float(np.sum(freqs * mag) / sum_mag)

    # Spectral Bandwidth: sqrt(sum((f - centroid)^2 * mag) / sum(mag))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / sum_mag))

    # Spectral Roll-off (85% energy cut-off)
    cum_energy = np.cumsum(psd)
    total_power = float(cum_energy[-1])
    if total_power > 0:
        threshold = 0.85 * total_power
        rolloff_idx = np.where(cum_energy >= threshold)[0]
        rolloff = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else float(freqs[-1])
    else:
        rolloff = 0.0

    return {
        "peak_frequency": peak_freq,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "total_power": total_power
    }
