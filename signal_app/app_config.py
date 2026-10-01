"""
Application Configuration and Default Hyperparameters.
"""
from dataclasses import dataclass, field
from typing import Dict, Any

DEFAULT_SAMPLING_RATE = 50000.0  # 50,000 Hz (samples/second)
DEFAULT_CHUNK_SIZE = 500000     # 500,000 rows per chunk (~10s of data at 50kHz)
DEFAULT_DOWNSAMPLE_POINTS = 5000 # Default downsampled points for display

@dataclass
class ProcessingConfig:
    """Configurable pipeline parameters."""
    sampling_rate: float = DEFAULT_SAMPLING_RATE
    chunk_size: int = DEFAULT_CHUNK_SIZE
    downsample_points: int = DEFAULT_DOWNSAMPLE_POINTS
    
    # Baseline
    baseline_method: str = "rolling_median"  # rolling_median, moving_average, polynomial, als
    baseline_window: float = 0.5            # seconds or window size factor
    baseline_poly_order: int = 2
    baseline_als_lam: float = 1e5
    baseline_als_p: float = 0.01
    
    # Smoothing / Filtering
    smoothing_method: str = "savgol"        # savgol, gaussian, moving_average, none
    smoothing_window: int = 51              # window length (odd integer for savgol)
    savgol_polyorder: int = 3
    gaussian_sigma: float = 5.0
    
    # Filtering
    filter_type: str = "none"               # none, lowpass, highpass, bandpass
    filter_order: int = 4
    cutoff_low: float = 10.0                # Hz
    cutoff_high: float = 2000.0             # Hz
    
    # Peak Detection
    peak_height: float = 0.1
    peak_prominence: float = 0.2
    peak_distance: int = 2500               # 50 ms at 50 kHz
    peak_width: float = 5.0                 # min width in samples
    peak_wlen: int = 5000
    resolve_overlapping_peaks: bool = False
    use_adaptive_threshold: bool = True
    adaptive_method: str = "rolling_std"     # rolling_std, mad, local_noise
    adaptive_factor: float = 3.0             # k * std
    adaptive_window: int = 50000             # window size (1 second)
    
    # Classification
    low_threshold: float = 1.0
    high_threshold: float = 3.0
    
    # Ground Truth Evaluation
    match_tolerance_ms: float = 20.0        # tolerance window for peak matching (ms)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()
