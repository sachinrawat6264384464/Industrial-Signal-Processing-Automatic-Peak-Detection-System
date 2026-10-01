"""
Compatibility wrapper module for legacy signal_analyzer imports.
"""
from signal_app.detection.peak_detector import PeakDetector
from signal_app.processing.preprocessing_pipeline import PreprocessingPipeline
from signal_app.app_config import ProcessingConfig

__all__ = ["PeakDetector", "PreprocessingPipeline", "ProcessingConfig"]
