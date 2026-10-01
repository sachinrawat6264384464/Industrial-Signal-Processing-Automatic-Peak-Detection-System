"""
Peak Feature Extraction Module.
Calculates geometric, amplitude, temporal, and SNR features for detected peaks.
"""
import numpy as np
import pandas as pd
from scipy.signal import peak_prominences, peak_widths
from typing import Dict, Any, Optional

def extract_peak_features(
    signal: np.ndarray,
    peak_indices: np.ndarray,
    sampling_rate: float = 50000.0,
    baseline: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Calculates detailed quantitative features for every detected peak.

    Parameters:
        signal: 1D signal array
        peak_indices: 1D array of sample indices where peaks occur
        sampling_rate: Sampling frequency in Hz
        baseline: Optional baseline signal array

    Returns:
        Pandas DataFrame containing extracted peak features.
    """
    if signal is None or len(signal) == 0 or len(peak_indices) == 0:
        return pd.DataFrame(columns=[
            "peak_index", "timestamp", "amplitude", "relative_amplitude",
            "prominence", "width", "width_height", "left_base", "right_base",
            "area", "rise_time", "fall_time", "duration",
            "local_baseline", "local_noise", "snr"
        ])

    n = len(signal)
    # Compute prominences and bases
    prominences, left_bases, right_bases = peak_prominences(signal, peak_indices)

    # Compute widths at half prominence
    widths, width_heights, left_ips, right_ips = peak_widths(
        signal, peak_indices, rel_height=0.5, prominence_data=(prominences, left_bases, right_bases)
    )

    records = []
    max_amp = np.max(signal) if np.max(signal) > 0 else 1.0

    for i, idx in enumerate(peak_indices):
        amp = float(signal[idx])
        rel_amp = float(amp / max_amp)
        prom = float(prominences[i])
        width = float(widths[i])
        w_height = float(width_heights[i])
        l_base = int(left_bases[i])
        r_base = int(right_bases[i])

        # Trapezoidal area under peak between left and right bases
        peak_segment = signal[l_base:r_base + 1]
        area = float(np.trapezoid(peak_segment)) if len(peak_segment) > 1 else amp

        rise_samples = max(0, idx - l_base)
        fall_samples = max(0, r_base - idx)
        duration_samples = rise_samples + fall_samples

        rise_time_s = rise_samples / sampling_rate
        fall_time_s = fall_samples / sampling_rate
        duration_s = duration_samples / sampling_rate
        timestamp_s = idx / sampling_rate

        # Local baseline and noise level around peak window
        win_l = max(0, idx - int(sampling_rate * 0.1))
        win_r = min(n, idx + int(sampling_rate * 0.1))
        local_sig = signal[win_l:win_r]

        loc_base = float(baseline[idx]) if baseline is not None and len(baseline) == n else float(np.median(local_sig))
        loc_noise = float(np.std(local_sig)) if len(local_sig) > 1 else 1e-5
        if loc_noise == 0:
            loc_noise = 1e-5

        snr = float(amp / loc_noise)

        records.append({
            "peak_index": int(idx),
            "timestamp": timestamp_s,
            "amplitude": amp,
            "relative_amplitude": rel_amp,
            "prominence": prom,
            "width": width,
            "width_height": w_height,
            "left_base": l_base,
            "right_base": r_base,
            "area": area,
            "rise_time": rise_time_s,
            "fall_time": fall_time_s,
            "duration": duration_s,
            "local_baseline": loc_base,
            "local_noise": loc_noise,
            "snr": snr
        })

    return pd.DataFrame(records)
