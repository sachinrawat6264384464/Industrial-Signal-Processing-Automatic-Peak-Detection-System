"""
Signal Normalization Module.
"""
import numpy as np

def normalize_min_max(signal: np.ndarray) -> np.ndarray:
    """Scales signal values to [0, 1] range."""
    s_min = np.min(signal)
    s_max = np.max(signal)
    if s_max == s_min:
        return np.zeros_like(signal)
    return (signal - s_min) / (s_max - s_min)

def normalize_zscore(signal: np.ndarray) -> np.ndarray:
    """Standardizes signal to mean=0, std=1."""
    std = np.std(signal)
    if std == 0:
        return np.zeros_like(signal)
    return (signal - np.mean(signal)) / std

def normalize_peak(signal: np.ndarray) -> np.ndarray:
    """Scales signal by maximum absolute peak amplitude."""
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return np.zeros_like(signal)
    return signal / max_val

def normalize_signal(signal: np.ndarray, method: str = "none") -> np.ndarray:
    """Common normalization interface."""
    if signal is None or len(signal) == 0 or method.lower() in ("none", "off"):
        return signal
    method = method.lower()
    if method == "minmax":
        return normalize_min_max(signal)
    elif method == "zscore":
        return normalize_zscore(signal)
    elif method == "peak":
        return normalize_peak(signal)
    return signal
