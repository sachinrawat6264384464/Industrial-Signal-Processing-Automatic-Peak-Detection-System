"""
Quantitative Evaluation Metrics for Ground-Truth vs Algorithm Comparison.
Calculates Precision, Recall, F1-Score, True Positives, False Positives, and False Negatives.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

def evaluate_peak_detection(
    detected_timestamps: np.ndarray,
    ground_truth_timestamps: np.ndarray,
    match_tolerance_ms: float = 20.0
) -> Dict[str, Any]:
    """
    Evaluates detected peak timestamps against ground truth peak timestamps.

    Parameters:
        detected_timestamps: 1D array of detected peak timestamps (seconds)
        ground_truth_timestamps: 1D array of true peak timestamps (seconds)
        match_tolerance_ms: Maximum temporal deviation (ms) allowed for matching

    Returns:
        Dict containing TP, FP, FN, Precision, Recall, F1, Detection Rate, False Detection Rate.
    """
    det = np.asarray(detected_timestamps, dtype=np.float64)
    gt = np.asarray(ground_truth_timestamps, dtype=np.float64)

    if len(gt) == 0:
        tp = 0
        fp = len(det)
        fn = 0
        precision = 0.0 if fp > 0 else 1.0
        recall = 1.0
        f1 = 0.0
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "detection_rate": recall,
            "false_detection_rate": 1.0 if fp > 0 else 0.0,
            "matched_pairs": []
        }

    if len(det) == 0:
        return {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(gt),
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "detection_rate": 0.0,
            "false_detection_rate": 0.0,
            "matched_pairs": []
        }

    tol_s = match_tolerance_ms / 1000.0

    gt_matched = np.zeros(len(gt), dtype=bool)
    det_matched = np.zeros(len(det), dtype=bool)
    matched_pairs: List[Tuple[float, float]] = []

    # Greedy match by smallest timestamp error
    for i, d_time in enumerate(det):
        diffs = np.abs(gt - d_time)
        best_idx = np.argmin(diffs)
        if diffs[best_idx] <= tol_s and not gt_matched[best_idx]:
            gt_matched[best_idx] = True
            det_matched[i] = True
            matched_pairs.append((d_time, gt[best_idx]))

    tp = int(np.sum(gt_matched))
    fp = int(len(det) - tp)
    fn = int(len(gt) - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "detection_rate": recall,
        "false_detection_rate": fdr,
        "matched_pairs": matched_pairs
    }
