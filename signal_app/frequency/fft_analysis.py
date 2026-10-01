"""
Fast Fourier Transform (FFT) and Power Spectral Density Analysis Module.
"""
import numpy as np
from scipy.fft import rfft, rfftfreq
from typing import Dict, Any, Tuple

def compute_fft(signal: np.ndarray, sampling_rate: float = 50000.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes real-valued Fast Fourier Transform (rFFT) and Power Spectral Density (PSD).

    Parameters:
        signal: 1D signal array
        sampling_rate: Sampling frequency in Hz

    Returns:
        (freqs, magnitude_spectrum, psd)
    """
    if signal is None or len(signal) == 0:
        return np.array([]), np.array([]), np.array([])

    n = len(signal)
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n)
    windowed_signal = signal * window

    fft_vals = rfft(windowed_signal)
    freqs = rfftfreq(n, d=1.0 / sampling_rate)

    magnitude = np.abs(fft_vals) / n
    # Double magnitude for non-DC components
    magnitude[1:] *= 2.0

    psd = (magnitude ** 2) / (sampling_rate * n)

    return freqs, magnitude, psd

def find_dominant_frequencies(
    freqs: np.ndarray,
    magnitude: np.ndarray,
    top_n: int = 5,
    min_freq: float = 0.5
) -> Dict[str, Any]:
    """
    Extracts top N dominant peak frequencies from the magnitude spectrum.
    """
    if len(freqs) == 0 or len(magnitude) == 0:
        return {"dominant_freqs": [], "magnitudes": []}

    mask = freqs >= min_freq
    f_filtered = freqs[mask]
    m_filtered = magnitude[mask]

    if len(m_filtered) == 0:
        return {"dominant_freqs": [], "magnitudes": []}

    top_indices = np.argsort(m_filtered)[::-1][:top_n]
    dom_freqs = f_filtered[top_indices].tolist()
    dom_mags = m_filtered[top_indices].tolist()

    return {
        "dominant_freqs": dom_freqs,
        "magnitudes": dom_mags
    }
