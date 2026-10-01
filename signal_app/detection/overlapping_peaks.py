"""
Overlapping Peak Detection and Deconvolution Module.
Uses 1st and 2nd derivative inflection points and local curvature analysis to resolve fused peaks.
"""
import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple

def detect_overlapping_peaks_derivative(
    signal: np.ndarray,
    primary_peaks: np.ndarray,
    min_distance: int = 10,
    prominence_ratio: float = 0.2
) -> np.ndarray:
    """
    Identifies hidden sub-peaks within broad/fused peak regions using 2nd derivative minima.

    Parameters:
        signal: 1D signal array
        primary_peaks: Initial detected peak indices
        min_distance: Minimum distance between sub-peaks
        prominence_ratio: Minimum ratio of peak height relative to max signal

    Returns:
        Refined peak indices array including deconvolved sub-peaks
    """
    if signal is None or len(signal) == 0 or len(primary_peaks) == 0:
        return primary_peaks

    max_sig = np.max(signal) if np.max(signal) > 0 else 1.0

    # 2nd derivative identifies concave points / hidden shoulder peaks
    d2 = np.gradient(np.gradient(signal))
    neg_d2 = -d2

    # Search for sharp negative curvature inflection points with minimum prominence
    d2_prom = np.std(neg_d2) * 2.0
    d2_peaks, _ = find_peaks(neg_d2, distance=min_distance, prominence=d2_prom)

    valid_subpeaks = []
    for p in d2_peaks:
        # Check if local region in signal is a true peak and above minimum amplitude threshold
        if p > 0 and p < len(signal) - 1:
            if signal[p] >= signal[p-1] and signal[p] >= signal[p+1]:
                if signal[p] >= prominence_ratio * max_sig:
                    valid_subpeaks.append(p)

    all_peaks = np.unique(np.concatenate([primary_peaks, valid_subpeaks]))
    all_peaks.sort()
    return all_peaks
