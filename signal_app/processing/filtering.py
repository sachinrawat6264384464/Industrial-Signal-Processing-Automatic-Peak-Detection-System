"""
Digital Filtering Module using SciPy Butterworth Filters.
"""
import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple
from ..utils.logging_config import logger
from ..utils.validators import validate_filter_params

def apply_digital_filter(
    signal: np.ndarray,
    sampling_rate: float,
    filter_type: str = "none",
    cutoff_low: float = 10.0,
    cutoff_high: float = 2000.0,
    order: int = 4
) -> np.ndarray:
    """
    Applies zero-phase Butterworth digital filtering (lowpass, highpass, bandpass).

    Parameters:
        signal: 1D signal array
        sampling_rate: Sampling frequency in Hz
        filter_type: 'none', 'lowpass', 'highpass', 'bandpass'
        cutoff_low: Low cutoff frequency (Hz)
        cutoff_high: High cutoff frequency (Hz)
        order: Filter order

    Returns:
        Filtered 1D signal array
    """
    if signal is None or len(signal) == 0 or filter_type.lower() in ("none", "off"):
        return signal

    filter_type = filter_type.lower()
    nyquist = sampling_rate / 2.0

    try:
        if filter_type == "lowpass":
            cutoff = min(cutoff_low, nyquist * 0.99)
            if cutoff <= 0:
                return signal
            b, a = butter(order, cutoff / nyquist, btype='lowpass')
        elif filter_type == "highpass":
            cutoff = max(cutoff_high, 0.01)
            if cutoff >= nyquist:
                return signal
            b, a = butter(order, cutoff / nyquist, btype='highpass')
        elif filter_type == "bandpass":
            c_low = max(0.01, cutoff_low)
            c_high = min(cutoff_high, nyquist * 0.99)
            if c_low >= c_high:
                logger.warning(f"Invalid bandpass cutoffs [{c_low}, {c_high}], skipping filter")
                return signal
            b, a = butter(order, [c_low / nyquist, c_high / nyquist], btype='bandpass')
        else:
            return signal

        filtered = filtfilt(b, a, signal)
        return filtered
    except Exception as e:
        logger.error(f"Error applying digital filter ({filter_type}): {e}")
        return signal
