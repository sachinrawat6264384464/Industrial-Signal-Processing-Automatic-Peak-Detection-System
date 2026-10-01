"""
Unit tests for Peak Detection and Adaptive Thresholding.
"""
import pytest
import numpy as np
from signal_app.app_config import ProcessingConfig
from signal_app.detection.peak_detector import PeakDetector

def test_peak_detection_synthetic():
    # Synthetic signal with 5 known peaks
    fs = 50000.0
    t = np.linspace(0, 1.0, int(fs))
    sig = np.random.normal(0, 0.02, len(t))

    peak_times = [0.1, 0.3, 0.5, 0.7, 0.9]
    for pt in peak_times:
        sig += 2.0 * np.exp(-0.5 * ((t - pt) / 0.002) ** 2)

    config = ProcessingConfig(
        sampling_rate=fs,
        peak_height=0.5,
        peak_prominence=0.5,
        peak_distance=1000,
        use_adaptive_threshold=True
    )

    detector = PeakDetector(config)
    peaks_df, threshold = detector.detect(sig)

    assert len(peaks_df) == 5
    detected_times = peaks_df["timestamp"].to_numpy()
    for pt in peak_times:
        diffs = np.abs(detected_times - pt)
        assert np.min(diffs) < 0.005  # Within 5ms
