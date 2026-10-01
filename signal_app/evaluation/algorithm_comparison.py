"""
Algorithm Comparison Framework for Rigorous Scientific Validation.
Executes multiple peak detection pipeline configurations on identical signals and compares data-driven performance metrics.
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from ..app_config import ProcessingConfig
from ..processing.preprocessing_pipeline import PreprocessingPipeline
from ..detection.peak_detector import PeakDetector
from .metrics import evaluate_peak_detection

class AlgorithmComparison:
    """Executes benchmark comparisons across multiple peak detection algorithm configurations."""

    @staticmethod
    def compare_methods(
        signal: np.ndarray,
        ground_truth_timestamps: np.ndarray,
        sampling_rate: float = 50000.0,
        match_tolerance_ms: float = 20.0
    ) -> pd.DataFrame:
        """
        Benchmarks 4 candidate algorithm configurations against ground truth.

        Returns DataFrame of comparative metrics.
        """
        configs = [
            ("Method A: Baseline + Smoothing + Amplitude", ProcessingConfig(
                sampling_rate=sampling_rate,
                baseline_method="rolling_median",
                smoothing_method="moving_average",
                use_adaptive_threshold=False,
                peak_height=0.5,
                peak_prominence=0.0
            )),
            ("Method B: Baseline + Smoothing + Prominence", ProcessingConfig(
                sampling_rate=sampling_rate,
                baseline_method="rolling_median",
                smoothing_method="savgol",
                use_adaptive_threshold=False,
                peak_height=0.2,
                peak_prominence=0.3
            )),
            ("Method C: Baseline + Adaptive Threshold + Prominence", ProcessingConfig(
                sampling_rate=sampling_rate,
                baseline_method="rolling_median",
                smoothing_method="savgol",
                use_adaptive_threshold=True,
                adaptive_method="rolling_std",
                adaptive_factor=3.0,
                peak_prominence=0.2
            )),
            ("Method D: ALS Baseline + Gaussian + Adaptive MAD", ProcessingConfig(
                sampling_rate=sampling_rate,
                baseline_method="als",
                smoothing_method="gaussian",
                use_adaptive_threshold=True,
                adaptive_method="mad",
                adaptive_factor=3.5,
                peak_prominence=0.2
            )),
        ]

        results = []
        for name, config in configs:
            t0 = time.perf_counter()
            pipeline = PreprocessingPipeline(config)
            proc_dict = pipeline.process(signal)

            detector = PeakDetector(config)
            peaks_df, _ = detector.detect(proc_dict['smoothed'], baseline=proc_dict['baseline'])
            t_proc = (time.perf_counter() - t0) * 1000.0  # ms

            det_ts = peaks_df["timestamp"].to_numpy() if not peaks_df.empty else np.array([])
            metrics = evaluate_peak_detection(det_ts, ground_truth_timestamps, match_tolerance_ms=match_tolerance_ms)

            results.append({
                "Algorithm": name,
                "Detected Peaks": len(det_ts),
                "True Positives": metrics["true_positives"],
                "False Positives": metrics["false_positives"],
                "False Negatives": metrics["false_negatives"],
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1 Score": round(metrics["f1_score"], 4),
                "Runtime (ms)": round(t_proc, 2)
            })

        return pd.DataFrame(results)
