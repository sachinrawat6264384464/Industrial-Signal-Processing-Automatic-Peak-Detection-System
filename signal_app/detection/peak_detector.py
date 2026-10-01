"""
Primary Peak Detector Engine.
Integrates SciPy find_peaks, Adaptive Thresholding, Overlapping Peak Resolution, and Feature Extraction.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Dict, Any, Optional, Tuple
from .adaptive_threshold import compute_adaptive_threshold
from .overlapping_peaks import detect_overlapping_peaks_derivative
from .peak_features import extract_peak_features
from ..app_config import ProcessingConfig
from ..utils.logging_config import logger

class PeakDetector:
    """Detects, filters, and quantifies signal peaks."""

    def __init__(self, config: ProcessingConfig):
        self.config = config

    def detect(self, signal: np.ndarray, baseline: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Executes peak detection pipeline on 1D signal.

        Returns:
            (peaks_df, threshold_array)
        """
        if signal is None or len(signal) == 0:
            return pd.DataFrame(), np.array([])

        n = len(signal)

        # 1. Compute Threshold curve (Adaptive or Fixed)
        if self.config.use_adaptive_threshold:
            threshold_curve = compute_adaptive_threshold(
                signal,
                method=self.config.adaptive_method,
                factor=self.config.adaptive_factor,
                window_size=self.config.adaptive_window,
                base_threshold=self.config.peak_height
            )
        else:
            threshold_curve = np.full(n, self.config.peak_height, dtype=np.float64)

        # 2. Primary scipy.signal.find_peaks search
        min_dist = max(1, self.config.peak_distance)
        min_prom = self.config.peak_prominence if self.config.peak_prominence > 0 else None
        min_w = self.config.peak_width if self.config.peak_width > 0 else None
        min_height = self.config.peak_height if self.config.peak_height > 0 else None

        peaks, properties = find_peaks(
            signal,
            height=min_height,
            distance=min_dist,
            prominence=min_prom,
            width=min_w,
            wlen=self.config.peak_wlen if self.config.peak_wlen > 0 else None
        )

        # Filter peaks by adaptive threshold curve if enabled
        if self.config.use_adaptive_threshold and len(peaks) > 0:
            valid_mask = signal[peaks] >= threshold_curve[peaks]
            peaks = peaks[valid_mask]

        # 3. Resolve overlapping fused peaks if explicitly enabled
        if self.config.resolve_overlapping_peaks and len(peaks) > 0:
            peaks = detect_overlapping_peaks_derivative(
                signal,
                peaks,
                min_distance=max(5, min_dist // 4),
                prominence_ratio=self.config.peak_height / max(1.0, np.max(signal))
            )

        # 4. Extract structured peak features DataFrame
        peaks_df = extract_peak_features(
            signal,
            peaks,
            sampling_rate=self.config.sampling_rate,
            baseline=baseline
        )

        logger.info(f"Detected {len(peaks_df)} peaks (Adaptive: {self.config.use_adaptive_threshold})")
        return peaks_df, threshold_curve
