"""
Unit tests for Precision, Recall, and F1 Evaluation Metrics.
"""
import pytest
import numpy as np
from signal_app.evaluation.metrics import evaluate_peak_detection

def test_evaluate_peak_detection_exact_match():
    gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    det = np.array([1.001, 2.002, 3.001, 4.003, 5.001])

    metrics = evaluate_peak_detection(det, gt, match_tolerance_ms=20.0)
    assert metrics["true_positives"] == 5
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0

def test_evaluate_peak_detection_misses_and_false_alarms():
    gt = np.array([1.0, 2.0, 3.0, 4.0])
    det = np.array([1.0, 2.0, 10.0])  # 10.0 is FP, 3.0 and 4.0 missed (FN)

    metrics = evaluate_peak_detection(det, gt, match_tolerance_ms=20.0)
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 2
    assert metrics["precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["recall"] == pytest.approx(2.0 / 4.0)
