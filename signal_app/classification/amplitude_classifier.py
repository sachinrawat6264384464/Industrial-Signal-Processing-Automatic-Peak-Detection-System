"""
Amplitude-Based Peak Classification Module.
Categorizes peaks into Low, Medium, and High based on configurable threshold boundaries.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple

def classify_amplitude(
    amplitude: float,
    low_threshold: float = 1.0,
    high_threshold: float = 3.0
) -> str:
    """
    Classifies a single peak amplitude into 'Low', 'Medium', or 'High'.

    Rules:
        amplitude < low_threshold -> Low
        low_threshold <= amplitude < high_threshold -> Medium
        amplitude >= high_threshold -> High
    """
    if amplitude < low_threshold:
        return "Low"
    elif amplitude < high_threshold:
        return "Medium"
    else:
        return "High"

def classify_peaks_by_amplitude(
    peaks_df: pd.DataFrame,
    low_threshold: float = 1.0,
    high_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Applies deterministic amplitude classification across all peaks in DataFrame.
    """
    if peaks_df is None or peaks_df.empty:
        return peaks_df

    df = peaks_df.copy()
    if "amplitude" in df.columns:
        df["category"] = df["amplitude"].apply(
            lambda amp: classify_amplitude(amp, low_threshold=low_threshold, high_threshold=high_threshold)
        )
    return df
