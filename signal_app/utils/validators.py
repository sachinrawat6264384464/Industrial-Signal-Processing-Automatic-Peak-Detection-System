"""
Validation Utilities for Input Data and Processing Parameters.
"""
from pathlib import Path
import numpy as np
from typing import Tuple

def validate_file_path(filepath: str) -> Path:
    """Validates that a file path exists and is a file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {filepath}")
    return path

def validate_sampling_rate(fs: float) -> float:
    """Validates sampling rate."""
    if fs <= 0:
        raise ValueError(f"Sampling rate must be positive, got {fs}")
    return float(fs)

def validate_filter_params(fs: float, filter_type: str, cutoff_low: float, cutoff_high: float) -> Tuple[float, float]:
    """Validates filter cutoff frequencies against Nyquist frequency."""
    nyquist = fs / 2.0
    if filter_type in ("lowpass", "bandpass"):
        if cutoff_low <= 0 or cutoff_low >= nyquist:
            raise ValueError(f"Cutoff frequency ({cutoff_low} Hz) must be strictly between 0 and Nyquist ({nyquist} Hz)")
    if filter_type in ("highpass", "bandpass"):
        if cutoff_high <= 0 or cutoff_high >= nyquist:
            raise ValueError(f"Cutoff frequency ({cutoff_high} Hz) must be strictly between 0 and Nyquist ({nyquist} Hz)")
    if filter_type == "bandpass" and cutoff_low >= cutoff_high:
        raise ValueError(f"Lower cutoff ({cutoff_low}) must be less than upper cutoff ({cutoff_high})")
    return cutoff_low, cutoff_high

def validate_numeric_array(arr: np.ndarray) -> np.ndarray:
    """Ensures input array is 1D, non-empty, and contains no NaNs or Infs."""
    if arr is None or arr.size == 0:
        raise ValueError("Input signal array is empty or None")
    arr = np.asarray(arr, dtype=np.float64)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        # Handle NaN/Inf by interpolating or replacing
        nan_mask = np.isnan(arr) | np.isinf(arr)
        arr[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), arr[~nan_mask])
    return arr
