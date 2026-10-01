"""
Peak-Preserving Downsampling Algorithms for Large Signal Visualization.
"""
import numpy as np
from typing import Tuple

def min_max_downsample(timestamps: np.ndarray, signal: np.ndarray, max_points: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Min-Max downsampling algorithm.
    Divides the signal into bins and extracts both min and max points per bin.
    This guarantees narrow peaks and troughs are preserved in visual plots.

    Parameters:
        timestamps: 1D array of time values
        signal: 1D array of signal amplitude values
        max_points: Maximum target points in downsampled output

    Returns:
        (ds_timestamps, ds_signal): Downsampled timestamps and signal arrays
    """
    n = len(signal)
    if n <= max_points:
        return timestamps, signal

    # Each bin produces 2 points (min and max)
    num_bins = max(1, max_points // 2)
    bin_size = n / num_bins

    ds_time = np.empty(num_bins * 2, dtype=np.float64)
    ds_signal = np.empty(num_bins * 2, dtype=np.float64)

    for i in range(num_bins):
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size) if i < num_bins - 1 else n
        if start_idx >= end_idx:
            continue

        bin_sig = signal[start_idx:end_idx]
        bin_time = timestamps[start_idx:end_idx]

        min_rel_idx = np.argmin(bin_sig)
        max_rel_idx = np.argmax(bin_sig)

        # Preserve chronological order within bin
        if min_rel_idx <= max_rel_idx:
            ds_time[2 * i] = bin_time[min_rel_idx]
            ds_signal[2 * i] = bin_sig[min_rel_idx]
            ds_time[2 * i + 1] = bin_time[max_rel_idx]
            ds_signal[2 * i + 1] = bin_sig[max_rel_idx]
        else:
            ds_time[2 * i] = bin_time[max_rel_idx]
            ds_signal[2 * i] = bin_sig[max_rel_idx]
            ds_time[2 * i + 1] = bin_time[min_rel_idx]
            ds_signal[2 * i + 1] = bin_sig[min_rel_idx]

    return ds_time, ds_signal

def envelope_downsample(timestamps: np.ndarray, signal: np.ndarray, max_points: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes signal envelope (upper & lower bounds) for multi-resolution rendering.

    Returns:
        (ds_timestamps, upper_envelope, lower_envelope)
    """
    n = len(signal)
    if n <= max_points:
        return timestamps, signal, signal

    num_bins = max_points
    bin_size = n / num_bins

    ds_time = np.empty(num_bins, dtype=np.float64)
    upper_env = np.empty(num_bins, dtype=np.float64)
    lower_env = np.empty(num_bins, dtype=np.float64)

    for i in range(num_bins):
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size) if i < num_bins - 1 else n
        if start_idx >= end_idx:
            continue

        bin_sig = signal[start_idx:end_idx]
        bin_time = timestamps[start_idx:end_idx]

        ds_time[i] = bin_time[len(bin_time) // 2]
        upper_env[i] = np.max(bin_sig)
        lower_env[i] = np.min(bin_sig)

    return ds_time, upper_env, lower_env
