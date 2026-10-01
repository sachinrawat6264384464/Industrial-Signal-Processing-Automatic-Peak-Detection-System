"""
Peak Quantification and Statistical Summary Module.
Calculates total peaks, density rates (peaks/min, peaks/hr), threshold breakdown, and category statistics.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def quantify_peaks(
    peaks_df: pd.DataFrame,
    total_duration_s: float,
    threshold: float = 0.0,
    start_time_s: Optional[float] = None,
    end_time_s: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes statistical metrics over peak dataframe for a specified time window.

    Parameters:
        peaks_df: DataFrame of detected peak features
        total_duration_s: Total signal recording duration in seconds
        threshold: Amplitude threshold for filtering
        start_time_s: Optional window start time
        end_time_s: Optional window end time

    Returns:
        Dict of quantitative statistics
    """
    if peaks_df is None or peaks_df.empty:
        return {
            "total_peaks": 0,
            "above_threshold": 0,
            "below_threshold": 0,
            "peaks_per_sec": 0.0,
            "peaks_per_min": 0.0,
            "peaks_per_hour": 0.0,
            "category_counts": {},
            "mean_amplitude": 0.0,
            "max_amplitude": 0.0,
            "mean_prominence": 0.0,
            "mean_width": 0.0,
            "window_duration_s": total_duration_s
        }

    df = peaks_df.copy()

    # Apply time window filtering if provided
    if start_time_s is not None and end_time_s is not None:
        df = df[(df["timestamp"] >= start_time_s) & (df["timestamp"] <= end_time_s)]
        window_duration = max(0.001, end_time_s - start_time_s)
    else:
        window_duration = max(0.001, total_duration_s)

    total_peaks = len(df)
    above_thresh = len(df[df["amplitude"] >= threshold]) if "amplitude" in df.columns else total_peaks
    below_thresh = total_peaks - above_thresh

    peaks_per_sec = total_peaks / window_duration
    peaks_per_min = peaks_per_sec * 60.0
    peaks_per_hour = peaks_per_sec * 3600.0

    category_counts = {}
    if "category" in df.columns:
        category_counts = df["category"].value_counts().to_dict()

    mean_amp = float(df["amplitude"].mean()) if total_peaks > 0 and "amplitude" in df.columns else 0.0
    max_amp = float(df["amplitude"].max()) if total_peaks > 0 and "amplitude" in df.columns else 0.0
    mean_prom = float(df["prominence"].mean()) if total_peaks > 0 and "prominence" in df.columns else 0.0
    mean_w = float(df["width"].mean()) if total_peaks > 0 and "width" in df.columns else 0.0

    return {
        "total_peaks": total_peaks,
        "above_threshold": above_thresh,
        "below_threshold": below_thresh,
        "peaks_per_sec": peaks_per_sec,
        "peaks_per_min": peaks_per_min,
        "peaks_per_hour": peaks_per_hour,
        "category_counts": category_counts,
        "mean_amplitude": mean_amp,
        "max_amplitude": max_amp,
        "mean_prominence": mean_prom,
        "mean_width": mean_w,
        "window_duration_s": window_duration
    }
