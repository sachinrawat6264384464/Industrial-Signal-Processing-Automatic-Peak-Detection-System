"""
Evaluation and scientific metrics package.
"""
from .ground_truth import GroundTruth
from .metrics import evaluate_peak_detection
from .algorithm_comparison import AlgorithmComparison

__all__ = ["GroundTruth", "evaluate_peak_detection", "AlgorithmComparison"]
