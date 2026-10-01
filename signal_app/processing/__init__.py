"""
Signal Preprocessing package.
"""
from .baseline_removal import remove_baseline, baseline_rolling_median, baseline_moving_average, baseline_polynomial, baseline_als
from .smoothing import smooth_signal, smooth_savgol, smooth_gaussian, smooth_moving_average
from .filtering import apply_digital_filter
from .normalization import normalize_signal, normalize_min_max, normalize_zscore, normalize_peak
from .preprocessing_pipeline import PreprocessingPipeline

__all__ = [
    "remove_baseline", "baseline_rolling_median", "baseline_moving_average", "baseline_polynomial", "baseline_als",
    "smooth_signal", "smooth_savgol", "smooth_gaussian", "smooth_moving_average",
    "apply_digital_filter",
    "normalize_signal", "normalize_min_max", "normalize_zscore", "normalize_peak",
    "PreprocessingPipeline"
]
