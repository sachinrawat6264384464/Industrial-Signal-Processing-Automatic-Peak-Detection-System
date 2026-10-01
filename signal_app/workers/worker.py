"""
Worker Threads for Asynchronous Processing without Freezing PySide6 GUI.
"""
from PySide6.QtCore import QRunnable, QObject, Slot
import numpy as np
import pandas as pd
import traceback
from typing import Callable, Any, Optional, Dict
from .worker_signals import WorkerSignals
from ..app_config import ProcessingConfig
from ..processing.preprocessing_pipeline import PreprocessingPipeline
from ..detection.peak_detector import PeakDetector
from ..frequency.fft_analysis import compute_fft, find_dominant_frequencies
from ..frequency.frequency_features import extract_frequency_features
from ..utils.logging_config import logger

class ProcessingWorker(QRunnable):
    """Executes heavy signal processing and peak detection in background thread."""

    def __init__(self, timestamps: np.ndarray, signal: np.ndarray, config: ProcessingConfig):
        super().__init__()
        self.timestamps = timestamps
        self.signal = signal
        self.config = config
        self.signals = WorkerSignals()
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    @Slot()
    def run(self):
        """Executes preprocessing and peak detection asynchronously."""
        try:
            self.signals.started.emit()
            self.signals.progress.emit(10, "Initializing Preprocessing Pipeline...")

            if self.is_cancelled:
                return

            # Preprocessing
            pipeline = PreprocessingPipeline(self.config)
            proc_dict = pipeline.process(self.signal)
            self.signals.progress.emit(50, "Executing Peak Detection & Feature Extraction...")

            if self.is_cancelled:
                return

            # Peak Detection
            detector = PeakDetector(self.config)
            peaks_df, threshold_curve = detector.detect(
                proc_dict['smoothed'],
                baseline=proc_dict['baseline']
            )

            self.signals.progress.emit(80, "Performing Frequency Analysis...")

            # Frequency analysis
            freqs, mag, psd = compute_fft(proc_dict['smoothed'], sampling_rate=self.config.sampling_rate)
            freq_features = extract_frequency_features(proc_dict['smoothed'], sampling_rate=self.config.sampling_rate)
            dom_freqs = find_dominant_frequencies(freqs, mag)

            self.signals.progress.emit(100, "Processing Complete!")

            payload = {
                "timestamps": self.timestamps,
                "raw": proc_dict["raw"],
                "baseline": proc_dict["baseline"],
                "baseline_removed": proc_dict["baseline_removed"],
                "filtered": proc_dict["filtered"],
                "smoothed": proc_dict["smoothed"],
                "threshold_curve": threshold_curve,
                "peaks_df": peaks_df,
                "fft_freqs": freqs,
                "fft_mag": mag,
                "fft_psd": psd,
                "freq_features": freq_features,
                "dominant_freqs": dom_freqs
            }

            self.signals.result.emit(payload)
            self.signals.finished.emit()
        except Exception as e:
            err_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"Worker execution failed: {err_msg}")
            self.signals.error.emit(str(e))
