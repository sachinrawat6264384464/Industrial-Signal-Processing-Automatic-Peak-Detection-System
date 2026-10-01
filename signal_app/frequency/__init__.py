"""
Frequency domain analysis package.
"""
from .fft_analysis import compute_fft, find_dominant_frequencies
from .frequency_features import extract_frequency_features

__all__ = ["compute_fft", "find_dominant_frequencies", "extract_frequency_features"]
