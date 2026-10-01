"""
Baseline Estimation and Removal Module.
Supports Rolling Median, Moving Average, Polynomial, and Asymmetric Least Squares (ALS).
"""
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import median_filter, uniform_filter1d
from typing import Tuple, Dict, Any
from ..utils.logging_config import logger

def baseline_rolling_median(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Estimates baseline using rolling median filter."""
    if window_size < 1:
        window_size = 1
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    return median_filter(signal, size=window_size, mode='nearest')

def baseline_moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Estimates baseline using moving average filter."""
    if window_size < 1:
        window_size = 1
    return uniform_filter1d(signal, size=window_size, mode='nearest')

def baseline_polynomial(signal: np.ndarray, poly_order: int = 2) -> np.ndarray:
    """Estimates baseline using polynomial curve fitting."""
    x = np.arange(len(signal))
    coeffs = np.polyfit(x, signal, poly_order)
    return np.polyval(coeffs, x)

def baseline_als(signal: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares (ALS) baseline estimation.
    Ref: Eilers & Boelens (2005)
    """
    L = len(signal)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z.tocsc(), w * signal)
        w = p * (signal > z) + (1 - p) * (signal <= z)
    return z

def remove_baseline(signal: np.ndarray, method: str = "rolling_median", **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Common baseline removal interface.

    Parameters:
        signal: 1D input array
        method: 'rolling_median', 'moving_average', 'polynomial', 'als', or 'none'
        kwargs: method-specific hyper-parameters

    Returns:
        (corrected_signal, baseline)
    """
    if signal is None or len(signal) == 0:
        return np.array([]), np.array([])

    method = method.lower()
    n = len(signal)

    if method == "rolling_median":
        window_size = kwargs.get("window_size", max(51, int(n * 0.05)))
        baseline = baseline_rolling_median(signal, window_size)
    elif method == "moving_average":
        window_size = kwargs.get("window_size", max(51, int(n * 0.05)))
        baseline = baseline_moving_average(signal, window_size)
    elif method == "polynomial":
        poly_order = kwargs.get("poly_order", 2)
        baseline = baseline_polynomial(signal, poly_order)
    elif method == "als":
        lam = kwargs.get("lam", 1e5)
        p = kwargs.get("p", 0.01)
        baseline = baseline_als(signal, lam, p)
    elif method == "none":
        baseline = np.zeros_like(signal)
    else:
        logger.warning(f"Unknown baseline method '{method}', defaulting to rolling median")
        window_size = kwargs.get("window_size", max(51, int(n * 0.05)))
        baseline = baseline_rolling_median(signal, window_size)

    corrected = signal - baseline
    return corrected, baseline
