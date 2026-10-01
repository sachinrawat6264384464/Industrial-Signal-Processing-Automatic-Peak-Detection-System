"""
Academic & Industrial Scientific Report Generator.
Generates comprehensive Markdown report detailing signal metadata, preprocessing configuration, peak detection, classification, ground-truth evaluation, and concentration analysis.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

class ReportGenerator:
    """Generates structured Markdown report for UEF & Marginum project."""

    @staticmethod
    def generate_report(
        metadata: Dict[str, Any],
        config: Dict[str, Any],
        stats: Dict[str, Any],
        evaluation_metrics: Optional[Dict[str, Any]] = None,
        concentration_results: Optional[Dict[str, Any]] = None,
        comparison_df: Optional[pd.DataFrame] = None,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generates full academic markdown report text and writes to file if output_path is specified.
        """
        lines = [
            "# Industrial Signal Processing & Peak Analysis Report",
            "**Project**: UEF & Marginum 2026",
            f"**File**: `{metadata.get('file_path', 'N/A')}`",
            f"**Sampling Rate**: {metadata.get('sampling_rate', 50000)} Hz",
            f"**Duration**: {metadata.get('estimated_duration_s', 0):.2f} seconds",
            f"**Total Samples**: {metadata.get('total_samples', 0):,}",
            "",
            "---",
            "## 1. Signal Preprocessing & Detection Configuration",
            f"- **Baseline Method**: {config.get('baseline_method', 'rolling_median')}",
            f"- **Smoothing Method**: {config.get('smoothing_method', 'savgol')}",
            f"- **Filter Type**: {config.get('filter_type', 'none')}",
            f"- **Adaptive Threshold**: {'Enabled' if config.get('use_adaptive_threshold') else 'Disabled'}",
            f"- **Peak Height**: {config.get('peak_height', 0.1)}",
            f"- **Peak Prominence**: {config.get('peak_prominence', 0.2)}",
            f"- **Peak Distance**: {config.get('peak_distance', 2500)} samples",
            "",
            "---",
            "## 2. Peak Quantification & Statistical Results",
            f"- **Total Peaks Detected**: {stats.get('total_peaks', 0)}",
            f"- **Peaks Above Threshold**: {stats.get('above_threshold', 0)}",
            f"- **Peaks Per Minute**: {stats.get('peaks_per_min', 0):.2f}",
            f"- **Peaks Per Hour**: {stats.get('peaks_per_hour', 0):.2f}",
            f"- **Mean Amplitude**: {stats.get('mean_amplitude', 0):.4f}",
            f"- **Max Amplitude**: {stats.get('max_amplitude', 0):.4f}",
            f"- **Mean Prominence**: {stats.get('mean_prominence', 0):.4f}",
            f"- **Mean Width**: {stats.get('mean_width', 0):.2f} samples",
            "",
            "### Category Breakdown",
        ]

        cats = stats.get("category_counts", {})
        if cats:
            for cat, cnt in cats.items():
                lines.append(f"- **{cat}**: {cnt} peaks")
        else:
            lines.append("- No category breakdown available.")

        lines.append("")
        lines.append("---")
        lines.append("## 3. Ground Truth Scientific Evaluation")

        if evaluation_metrics:
            lines.extend([
                f"- **True Positives (TP)**: {evaluation_metrics.get('true_positives', 0)}",
                f"- **False Positives (FP)**: {evaluation_metrics.get('false_positives', 0)}",
                f"- **False Negatives (FN)**: {evaluation_metrics.get('false_negatives', 0)}",
                f"- **Precision**: {evaluation_metrics.get('precision', 0.0):.4f}",
                f"- **Recall (Detection Rate)**: {evaluation_metrics.get('recall', 0.0):.4f}",
                f"- **F1 Score**: {evaluation_metrics.get('f1_score', 0.0):.4f}",
                f"- **False Detection Rate**: {evaluation_metrics.get('false_detection_rate', 0.0):.4f}"
            ])
        else:
            lines.append("*Ground truth annotation file was not loaded for this run.*")

        lines.append("")
        lines.append("---")
        lines.append("## 4. Fluorophore Concentration & Tumour Size Analysis")

        if concentration_results:
            lines.extend([
                f"- **Detected Peak Count**: {concentration_results.get('detected_peak_count', 0)}",
                f"- **Estimated Tumour Size**: {concentration_results.get('estimated_tumour_size_ml', 0)} mL",
                f"- **Estimated Concentration**: **{concentration_results.get('estimated_concentration', 'N/A')}**",
                f"- **Regression Model**: {concentration_results.get('regression_model', 'Linear')} (R² = {concentration_results.get('regression_r2', 0):.4f})",
                f"> **Note**: {concentration_results.get('note', '')}"
            ])
        else:
            lines.append("*Concentration estimation was not performed.*")

        if comparison_df is not None and not comparison_df.empty:
            lines.append("")
            lines.append("---")
            lines.append("## 5. Algorithm Comparison Benchmark Table")
            lines.append(comparison_df.to_markdown(index=False))

        report_content = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)

        return report_content
