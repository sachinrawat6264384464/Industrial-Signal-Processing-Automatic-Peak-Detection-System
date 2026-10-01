"""
Signal Smoothing and Noise Reduction Module.
Supports Savitzky-Golay, Gaussian, Moving Average, and None.
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from typing import Dict, Any
from ..utils.logging_config import logger

def smooth_savgol(signal: np.ndarray, window_length: int = 51, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay filtering preserves peak heights while smoothing noise."""
    n = len(signal)
    if window_length >= n:
        window_length = n - 1 if (n - 1) % 2 != 0 else n - 2
    if window_length <= polyorder:
        window_length = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)

def smooth_gaussian(signal: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Gaussian kernel smoothing."""
    return gaussian_filter1d(signal, sigma=sigma)

def smooth_moving_average(signal: np.ndarray, window_size: int = 21) -> np.ndarray:
    """Moving average boxcar smoothing."""
    if window_size < 1:
        window_size = 1
    return uniform_filter1d(signal, size=window_size)

def smooth_signal(signal: np.ndarray, method: str = "savgol", **kwargs) -> np.ndarray:
    """
    Common signal smoothing interface.

    Parameters:
        signal: Input 1D numpy array
        method: 'savgol', 'gaussian', 'moving_average', 'none'
        kwargs: Hyperparameters (window_length, polyorder, sigma, window_size)

    Returns:
        Smoothed signal array
    """
    if signal is None or len(signal) == 0:
        return np.array([])

    method = method.lower()
    if method == "savgol":
        win = kwargs.get("window_length", kwargs.get("window_size", 51))
        poly = kwargs.get("polyorder", 3)
        return smooth_savgol(signal, window_length=win, polyorder=poly)
    elif method == "gaussian":
        sigma = kwargs.get("sigma", 5.0)
        return smooth_gaussian(signal, sigma=sigma)
    elif method == "moving_average":
        win = kwargs.get("window_size", 21)
        return smooth_moving_average(signal, window_size=win)
    elif method == "none":
        return signal.copy()
    else:
        logger.warning(f"Unknown smoothing method '{method}', returning original signal")
        return signal.copy()
