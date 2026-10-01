"""
Unit tests for Tumour Size vs Peak Count Regression and Concentration Estimation.
"""
import pytest
import numpy as np
from signal_app.concentration.regression import fit_linear_regression, fit_polynomial_regression
from signal_app.concentration.concentration_estimator import ConcentrationEstimator

def test_linear_regression():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    res = fit_linear_regression(x, y)
    assert res["r2"] == pytest.approx(1.0)
    assert res["slope"] == pytest.approx(10.0)
    assert res["intercept"] == pytest.approx(0.0)

def test_concentration_estimator():
    estimator = ConcentrationEstimator()
    est = estimator.estimate_from_peak_count(250)
    assert est["detected_peak_count"] == 250
    assert est["estimated_concentration"] == "Medium"
    assert est["estimated_tumour_size_ml"] > 0
