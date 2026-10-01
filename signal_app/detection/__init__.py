"""
Peak Detection Package.
"""
from .adaptive_threshold import compute_adaptive_threshold, compute_rolling_std, compute_rolling_mad
from .overlapping_peaks import detect_overlapping_peaks_derivative
from .peak_features import extract_peak_features
from .peak_detector import PeakDetector
from .peak_quantification import quantify_peaks

__all__ = [
    "compute_adaptive_threshold", "compute_rolling_std", "compute_rolling_mad",
    "detect_overlapping_peaks_derivative",
    "extract_peak_features",
    "PeakDetector",
    "quantify_peaks"
]
