"""
Complete Preprocessing Pipeline.
Sequentially applies Baseline Removal -> Digital Filtering -> Noise Smoothing -> Normalization.
"""
import numpy as np
from typing import Dict, Any, Tuple
from .baseline_removal import remove_baseline
from .filtering import apply_digital_filter
from .smoothing import smooth_signal
from .normalization import normalize_signal
from ..app_config import ProcessingConfig

class PreprocessingPipeline:
    """Manages full signal preprocessing execution."""

    def __init__(self, config: ProcessingConfig):
        self.config = config

    def process(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Executes full preprocessing pipeline.

        Returns dict containing:
            'raw': original signal
            'baseline': estimated baseline
            'baseline_removed': signal after baseline removal
            'filtered': signal after digital filtering
            'smoothed': final processed signal after noise reduction
        """
        if signal is None or len(signal) == 0:
            empty = np.array([], dtype=np.float64)
            return {
                'raw': empty,
                'baseline': empty,
                'baseline_removed': empty,
                'filtered': empty,
                'smoothed': empty
            }

        # Step 1: Baseline removal
        baseline_removed, baseline = remove_baseline(
            signal,
            method=self.config.baseline_method,
            window_size=int(self.config.baseline_window * self.config.sampling_rate),
            poly_order=self.config.baseline_poly_order,
            lam=self.config.baseline_als_lam,
            p=self.config.baseline_als_p
        )

        # Step 2: Digital filtering
        filtered = apply_digital_filter(
            baseline_removed,
            sampling_rate=self.config.sampling_rate,
            filter_type=self.config.filter_type,
            cutoff_low=self.config.cutoff_low,
            cutoff_high=self.config.cutoff_high,
            order=self.config.filter_order
        )

        # Step 3: Noise smoothing
        smoothed = smooth_signal(
            filtered,
            method=self.config.smoothing_method,
            window_length=self.config.smoothing_window,
            polyorder=self.config.savgol_polyorder,
            sigma=self.config.gaussian_sigma
        )

        return {
            'raw': signal,
            'baseline': baseline,
            'baseline_removed': baseline_removed,
            'filtered': filtered,
            'smoothed': smoothed
        }
