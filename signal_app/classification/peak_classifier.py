"""
Multi-Feature Peak Classifier Engine.
Combines Amplitude, Prominence, Width, and SNR for robust peak categorization.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .amplitude_classifier import classify_peaks_by_amplitude

class PeakClassifier:
    """Classifies peak signals into severity/amplitude tiers and assigns confidence metrics."""

    def __init__(self, low_threshold: float = 1.0, high_threshold: float = 3.0, min_snr: float = 3.0):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.min_snr = min_snr

    def classify(self, peaks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies DataFrame of peaks and adds 'category' and 'confidence' columns.
        """
        if peaks_df is None or peaks_df.empty:
            return peaks_df

        df = classify_peaks_by_amplitude(peaks_df, low_threshold=self.low_threshold, high_threshold=self.high_threshold)

        # Compute heuristic confidence score based on SNR and Prominence
        confidences = []
        for idx, row in df.iterrows():
            snr = row.get("snr", 5.0)
            prom = row.get("prominence", 1.0)
            amp = row.get("amplitude", 1.0)

            # High SNR & Prominence yields higher confidence
            snr_score = min(1.0, max(0.2, snr / 10.0))
            prom_score = min(1.0, max(0.2, prom / max(0.001, amp)))

            conf = float(0.6 * snr_score + 0.4 * prom_score)
            conf = float(np.clip(conf, 0.1, 1.0))
            confidences.append(conf)

        df["confidence"] = confidences
        return df
