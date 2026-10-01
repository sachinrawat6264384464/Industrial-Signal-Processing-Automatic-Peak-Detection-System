"""
Peak and Signal Overlay Helper for Visualization.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

class PeakOverlay:
    """Formats detected peak metadata into plot-ready scatter markers and labels."""

    @staticmethod
    def prepare_peak_scatter(peaks_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts peak timestamps, amplitudes, categories, and colors for plotting.
        """
        if peaks_df is None or peaks_df.empty:
            return {
                "x": np.array([]),
                "y": np.array([]),
                "prominence": np.array([]),
                "category": [],
                "colors": []
            }

        x = peaks_df["timestamp"].to_numpy(dtype=np.float64)
        y = peaks_df["amplitude"].to_numpy(dtype=np.float64)
        prom = peaks_df["prominence"].to_numpy(dtype=np.float64) if "prominence" in peaks_df.columns else y
        cats = peaks_df["category"].tolist() if "category" in peaks_df.columns else ["High"] * len(x)

        category_colors = {
            "Low": "#3498db",      # Blue
            "Medium": "#f39c12",   # Orange
            "High": "#e74c3c",     # Red
            "Unclassified": "#95a5a6"
        }

        colors = [category_colors.get(c, "#e74c3c") for c in cats]

        return {
            "x": x,
            "y": y,
            "prominence": prom,
            "category": cats,
            "colors": colors
        }
