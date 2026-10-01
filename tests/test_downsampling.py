"""
Unit tests for Min-Max and Envelope Downsampling algorithms.
"""
import pytest
import numpy as np
from signal_app.visualization.downsampling import min_max_downsample, envelope_downsample

def test_min_max_downsample_peak_preservation():
    # Signal with 1,000,000 baseline zeros and a sharp peak of 100.0 at index 500,000
    n = 100000
    t = np.linspace(0, 10, n)
    sig = np.zeros(n)
    sig[50000] = 100.0  # Sharp spike

    ds_t, ds_sig = min_max_downsample(t, sig, max_points=1000)

    # Downsampled size should be at most max_points
    assert len(ds_sig) <= 1000
    # Peak amplitude 100.0 MUST be preserved in downsampled output
    assert np.max(ds_sig) == 100.0

def test_envelope_downsample():
    t = np.linspace(0, 1, 10000)
    sig = np.sin(2 * np.pi * 5 * t)
    ds_t, upper, lower = envelope_downsample(t, sig, max_points=100)

    assert len(ds_t) == 100
    assert np.max(upper) == pytest.approx(1.0, abs=0.05)
    assert np.min(lower) == pytest.approx(-1.0, abs=0.05)
