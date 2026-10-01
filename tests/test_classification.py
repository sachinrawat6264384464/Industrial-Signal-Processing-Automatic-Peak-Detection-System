"""
Unit tests for Peak Classification algorithms.
"""
import pytest
import pandas as pd
from signal_app.classification.amplitude_classifier import classify_amplitude, classify_peaks_by_amplitude
from signal_app.classification.peak_classifier import PeakClassifier

def test_classify_amplitude():
    assert classify_amplitude(0.5, low_threshold=1.0, high_threshold=3.0) == "Low"
    assert classify_amplitude(1.5, low_threshold=1.0, high_threshold=3.0) == "Medium"
    assert classify_amplitude(3.5, low_threshold=1.0, high_threshold=3.0) == "High"

def test_peak_classifier_dataframe():
    df = pd.DataFrame({
        "amplitude": [0.4, 2.0, 5.0],
        "prominence": [0.3, 1.8, 4.5],
        "snr": [4.0, 12.0, 25.0]
    })
    classifier = PeakClassifier(low_threshold=1.0, high_threshold=3.0)
    classified_df = classifier.classify(df)

    assert classified_df["category"].tolist() == ["Low", "Medium", "High"]
    assert "confidence" in classified_df.columns
    assert all(0.0 <= c <= 1.0 for c in classified_df["confidence"])
