"""
Adaptive Thresholding Module for Dynamic Noise Estimation.
Supports Rolling Standard Deviation, Median Absolute Deviation (MAD), and Local Noise Estimation.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter

def compute_rolling_std(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Computes rolling standard deviation."""
    if window_size < 2:
        return np.zeros_like(signal)
    mean = uniform_filter1d(signal, size=window_size)
    sq_mean = uniform_filter1d(signal**2, size=window_size)
    var = np.maximum(0.0, sq_mean - mean**2)
    return np.sqrt(var)

def compute_rolling_mad(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Computes Median Absolute Deviation (MAD) robust noise estimate."""
    if window_size % 2 == 0:
        window_size += 1
    med = median_filter(signal, size=window_size)
    dev = np.abs(signal - med)
    mad = median_filter(dev, size=window_size)
    return 1.4826 * mad  # scale factor for normal distribution equivalence

def compute_adaptive_threshold(
    signal: np.ndarray,
    method: str = "rolling_std",
    factor: float = 3.0,
    window_size: int = 50000,
    base_threshold: float = 0.0
) -> np.ndarray:
    """
    Computes time-varying dynamic threshold curve.

    Parameters:
        signal: 1D input array
        method: 'rolling_std', 'mad', 'local_noise'
        factor: multiplier k for threshold (e.g. baseline + k * noise_std)
        window_size: rolling window in samples
        base_threshold: constant offset threshold

    Returns:
        1D threshold array matching signal length
    """
    if signal is None or len(signal) == 0:
        return np.array([])

    method = method.lower()
    window_size = max(10, min(len(signal), window_size))

    if method == "rolling_std":
        noise_level = compute_rolling_std(signal, window_size)
    elif method == "mad":
        noise_level = compute_rolling_mad(signal, window_size)
    else: # local_noise
        noise_level = compute_rolling_std(signal, window_size)

    # Dynamic threshold is base_threshold + factor * noise_level
    threshold = base_threshold + factor * noise_level
    return threshold
