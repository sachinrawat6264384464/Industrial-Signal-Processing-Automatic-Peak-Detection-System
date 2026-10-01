"""
Unit tests for Baseline Removal algorithms.
"""
import pytest
import numpy as np
from signal_app.processing.baseline_removal import remove_baseline

def test_rolling_median_baseline():
    x = np.linspace(0, 10, 1000)
    baseline_true = 0.5 * x**2
    signal = baseline_true + np.random.normal(0, 0.1, 1000)

    corrected, estimated_base = remove_baseline(signal, method="rolling_median", window_size=101)
    assert len(corrected) == 1000
    assert len(estimated_base) == 1000

def test_polynomial_baseline():
    x = np.linspace(0, 10, 1000)
    baseline_true = 2.0 * x + 1.5
    signal = baseline_true + np.random.normal(0, 0.05, 1000)

    corrected, estimated_base = remove_baseline(signal, method="polynomial", poly_order=1)
    assert np.allclose(estimated_base, baseline_true, atol=0.2)
