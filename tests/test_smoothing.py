"""
Unit tests for Smoothing and Noise Reduction algorithms.
"""
import pytest
import numpy as np
from signal_app.processing.smoothing import smooth_signal

def test_savgol_smoothing():
    t = np.linspace(0, 1, 500)
    clean = np.sin(2 * np.pi * 5 * t)
    noisy = clean + np.random.normal(0, 0.2, 500)

    smoothed = smooth_signal(noisy, method="savgol", window_length=31, polyorder=3)
    # Smoothed signal should be closer to clean signal than noisy
    err_noisy = np.mean((noisy - clean) ** 2)
    err_smoothed = np.mean((smoothed - clean) ** 2)
    assert err_smoothed < err_noisy

def test_gaussian_smoothing():
    signal = np.array([0, 0, 10.0, 0, 0], dtype=np.float64)
    smoothed = smooth_signal(signal, method="gaussian", sigma=1.0)
    assert smoothed[2] < 10.0  # Gaussian spreads out peak area
    assert np.sum(smoothed) == pytest.approx(10.0, abs=0.1)
