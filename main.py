"""
Industrial Signal Analyzer Main Application Entry Point.
"""
import sys
import os
from pathlib import Path
from PySide6.QtWidgets import QApplication
from signal_app.main_window import SignalAnalyzerMainWindow
from signal_app.utils.logging_config import logger

def main():
    """Main application initialization and launcher."""
    logger.info("Launching Industrial Signal Analyzer 2026...")

    app = QApplication(sys.argv)
    app.setApplicationName("Industrial Signal Analyzer")
    app.setOrganizationName("UEF & Marginum")

    window = SignalAnalyzerMainWindow()

    # Automatically load synthetic sample file if available
    sample_file = Path("data/sample/industrial_sensor_signal_120s.csv")
    gt_file = Path("data/ground_truth/ground_truth_peaks_120s.csv")

    if sample_file.exists():
        logger.info(f"Auto-loading default synthetic sample: {sample_file}")
        window.dashboard.load_file(str(sample_file))
        if gt_file.exists():
            from signal_app.evaluation.ground_truth import GroundTruth
            window.dashboard.ground_truth = GroundTruth(str(gt_file))
        
        # Trigger initial processing run automatically
        window.dashboard.run_processing()

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
