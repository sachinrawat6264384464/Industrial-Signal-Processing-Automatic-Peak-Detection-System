"""
Classification package.
"""
from .amplitude_classifier import classify_amplitude, classify_peaks_by_amplitude
from .peak_classifier import PeakClassifier

__all__ = ["classify_amplitude", "classify_peaks_by_amplitude", "PeakClassifier"]
