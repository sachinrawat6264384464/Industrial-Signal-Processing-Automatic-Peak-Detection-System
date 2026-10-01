"""
Dashboard Widget containing controls, interactive signal plots, results tables, and evaluation metrics.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QFileDialog, QProgressBar, QMessageBox, QSplitter, QHeaderView
)
from PySide6.QtCore import Qt, QThreadPool, Slot
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from .app_config import ProcessingConfig
from .io.csv_reader import SignalCSVReader
from .visualization.signal_plot import SignalPlotWidget
from .visualization.overview_plot import OverviewPlotWidget
from .workers.worker import ProcessingWorker
from .classification.peak_classifier import PeakClassifier
from .detection.peak_detector import PeakDetector
from .processing.preprocessing_pipeline import PreprocessingPipeline
from .detection.peak_quantification import quantify_peaks
from .evaluation.ground_truth import GroundTruth
from .evaluation.metrics import evaluate_peak_detection
from .evaluation.algorithm_comparison import AlgorithmComparison
from .concentration.concentration_estimator import ConcentrationEstimator
from .export.results_exporter import ResultsExporter
from .export.report_generator import ReportGenerator
from .utils.logging_config import logger
from .utils.helpers import format_file_size, format_time_duration

class SignalAnalyzerDashboard(QWidget):
    """Main application dashboard widget."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = ProcessingConfig()
        self.thread_pool = QThreadPool.globalInstance()

        self.reader: Optional[SignalCSVReader] = None
        self.ground_truth: Optional[GroundTruth] = None
        self.current_sensor: str = "Sensor 1"
        self.raw_timestamps: np.ndarray = np.array([])
        self.raw_signal: np.ndarray = np.array([])
        
        # Result cache
        self.processed_payload: Optional[Dict[str, Any]] = None

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # 1. Top File & Control Bar
        top_bar = self._create_top_bar()
        main_layout.addWidget(top_bar)

        # Main horizontal splitter (Controls Left | Plots & Tables Right)
        splitter = QSplitter(Qt.Horizontal)

        # 2. Left Control Panel
        left_controls = self._create_left_control_panel()
        left_controls.setFixedWidth(320)
        splitter.addWidget(left_controls)

        # Right Panel (Plot views + Results Tabs)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 3. Overview & Detailed Signal Plots
        self.overview_plot = OverviewPlotWidget()
        self.signal_plot = SignalPlotWidget()

        self.overview_plot.range_changed.connect(self._on_overview_range_changed)

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.overview_plot)
        plot_layout.addWidget(self.signal_plot)

        # 4. Results Tabs Widget
        self.results_tabs = self._create_results_tabs()

        # Vertical splitter for plots top vs results bottom
        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.addWidget(plot_container)
        v_splitter.addWidget(self.results_tabs)
        v_splitter.setSizes([450, 300])

        right_layout.addWidget(v_splitter)
        splitter.addWidget(right_panel)
        splitter.setSizes([320, 960])

        main_layout.addWidget(splitter)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

    def _create_top_bar(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        btn_open = QPushButton("📁 Open Signal CSV")
        btn_open.setStyleSheet("font-weight: bold; background-color: #313244; padding: 6px 12px;")
        btn_open.clicked.connect(self._on_open_file_clicked)

        btn_gt = QPushButton("🎯 Load Ground Truth")
        btn_gt.clicked.connect(self._on_load_gt_clicked)

        btn_synth = QPushButton("⚡ Generate Sample Data")
        btn_synth.clicked.connect(self._on_generate_sample_clicked)

        self.lbl_file_info = QLabel("No signal file loaded")
        self.lbl_file_info.setStyleSheet("color: #a6adc8; font-weight: 500;")

        layout.addWidget(btn_open)
        layout.addWidget(btn_gt)
        layout.addWidget(btn_synth)
        layout.addWidget(self.lbl_file_info)
        layout.addStretch()

        # Channel Selector
        layout.addWidget(QLabel("Sensor Channel:"))
        self.combo_sensor = QComboBox()
        self.combo_sensor.addItems(["Sensor 1", "Sensor 2"])
        self.combo_sensor.currentIndexChanged.connect(self._on_sensor_changed)
        layout.addWidget(self.combo_sensor)

        # Sample Rate Field
        layout.addWidget(QLabel("Sampling Rate (Hz):"))
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(1.0, 1000000.0)
        self.spin_fs.setValue(50000.0)
        self.spin_fs.valueChanged.connect(self._on_fs_changed)
        layout.addWidget(self.spin_fs)

        return widget

    def _create_left_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        # Group 1: Preprocessing Controls
        grp_proc = QGroupBox("Signal Preprocessing")
        g1_layout = QGridLayout(grp_proc)

        g1_layout.addWidget(QLabel("Baseline:"), 0, 0)
        self.combo_baseline = QComboBox()
        self.combo_baseline.addItems(["rolling_median", "moving_average", "polynomial", "als", "none"])
        g1_layout.addWidget(self.combo_baseline, 0, 1)

        g1_layout.addWidget(QLabel("Smoothing:"), 1, 0)
        self.combo_smoothing = QComboBox()
        self.combo_smoothing.addItems(["savgol", "gaussian", "moving_average", "none"])
        g1_layout.addWidget(self.combo_smoothing, 1, 1)

        g1_layout.addWidget(QLabel("Filter Type:"), 2, 0)
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["none", "lowpass", "highpass", "bandpass"])
        g1_layout.addWidget(self.combo_filter, 2, 1)

        layout.addWidget(grp_proc)

        # Group 2: Peak Detection Controls
        grp_detect = QGroupBox("Peak Detection Engine")
        g2_layout = QGridLayout(grp_detect)

        g2_layout.addWidget(QLabel("Min Height:"), 0, 0)
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.0, 100.0)
        self.spin_height.setValue(0.1)
        self.spin_height.setSingleStep(0.05)
        g2_layout.addWidget(self.spin_height, 0, 1)

        g2_layout.addWidget(QLabel("Prominence:"), 1, 0)
        self.spin_prominence = QDoubleSpinBox()
        self.spin_prominence.setRange(0.0, 100.0)
        self.spin_prominence.setValue(0.2)
        self.spin_prominence.setSingleStep(0.05)
        g2_layout.addWidget(self.spin_prominence, 1, 1)

        g2_layout.addWidget(QLabel("Distance (samples):"), 2, 0)
        self.spin_distance = QSpinBox()
        self.spin_distance.setRange(1, 500000)
        self.spin_distance.setValue(2500)
        g2_layout.addWidget(self.spin_distance, 2, 1)

        self.chk_adaptive = QCheckBox("Enable Adaptive Threshold")
        self.chk_adaptive.setChecked(True)
        g2_layout.addWidget(self.chk_adaptive, 3, 0, 1, 2)

        self.chk_overlap = QCheckBox("Deconvolve Overlapping Peaks")
        self.chk_overlap.setChecked(False)
        g2_layout.addWidget(self.chk_overlap, 4, 0, 1, 2)

        layout.addWidget(grp_detect)

        # Action Buttons (Fixing Qt mnemonics by using &&)
        btn_run = QPushButton("🚀 Run Processing && Peak Detection")
        btn_run.setStyleSheet("font-weight: bold; background-color: #a6e3a1; color: #11111b; padding: 10px;")
        btn_run.clicked.connect(self.run_processing)
        layout.addWidget(btn_run)

        btn_eval = QPushButton("🔬 Ground Truth Evaluation")
        btn_eval.clicked.connect(self.run_evaluation)
        layout.addWidget(btn_eval)

        btn_compare = QPushButton("📊 Compare Algorithms (A vs B vs C vs D)")
        btn_compare.clicked.connect(self.run_algorithm_comparison)
        layout.addWidget(btn_compare)

        btn_conc = QPushButton("🧪 Fluorophore Concentration Analysis")
        btn_conc.clicked.connect(self.run_concentration_analysis)
        layout.addWidget(btn_conc)

        # Export Controls
        grp_export = QGroupBox("Export && Reports")
        ex_layout = QVBoxLayout(grp_export)

        btn_exp_peaks = QPushButton("📄 Export Peaks CSV")
        btn_exp_peaks.clicked.connect(self._export_peaks_csv)
        ex_layout.addWidget(btn_exp_peaks)

        btn_exp_report = QPushButton("📝 Generate Scientific Report")
        btn_exp_report.clicked.connect(self._generate_report)
        ex_layout.addWidget(btn_exp_report)

        layout.addWidget(grp_export)
        layout.addStretch()

        return panel

    def _create_results_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        # Tab 1: Peaks Table
        self.tbl_peaks = QTableWidget()
        self.tbl_peaks.setColumnCount(9)
        self.tbl_peaks.setHorizontalHeaderLabels([
            "Peak #", "Timestamp (s)", "Amplitude", "Prominence", "Width", "Area", "Category", "SNR", "Confidence"
        ])
        self.tbl_peaks.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tabs.addTab(self.tbl_peaks, "Detected Peaks")

        # Tab 2: Quantification Summary
        self.lbl_quant_summary = QLabel("Run peak detection to generate quantification statistics.")
        self.lbl_quant_summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_quant_summary.setWordWrap(True)
        self.lbl_quant_summary.setStyleSheet("font-size: 13px; padding: 12px;")
        tabs.addTab(self.lbl_quant_summary, "Quantification Stats")

        # Tab 3: Frequency Analysis
        self.tbl_freq = QTableWidget()
        self.tbl_freq.setColumnCount(2)
        self.tbl_freq.setHorizontalHeaderLabels(["Dominant Frequency (Hz)", "Spectral Magnitude"])
        self.tbl_freq.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tabs.addTab(self.tbl_freq, "FFT Frequency Analysis")

        # Tab 4: Evaluation Metrics
        self.lbl_eval_metrics = QLabel("Load ground truth file and click 'Ground Truth Evaluation'.")
        self.lbl_eval_metrics.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_eval_metrics.setWordWrap(True)
        self.lbl_eval_metrics.setStyleSheet("font-size: 13px; padding: 12px;")
        tabs.addTab(self.lbl_eval_metrics, "Ground Truth Evaluation")

        # Tab 5: Concentration Analysis
        self.lbl_concentration = QLabel("Click 'Fluorophore Concentration Analysis' to fit regression model.")
        self.lbl_concentration.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_concentration.setWordWrap(True)
        self.lbl_concentration.setStyleSheet("font-size: 13px; padding: 12px;")
        tabs.addTab(self.lbl_concentration, "Fluorophore Concentration")

        # Tab 6: Algorithm Comparison
        self.tbl_comparison = QTableWidget()
        self.tbl_comparison.setColumnCount(9)
        self.tbl_comparison.setHorizontalHeaderLabels([
            "Algorithm", "Detected", "TP", "FP", "FN", "Precision", "Recall", "F1 Score", "Runtime (ms)"
        ])
        self.tbl_comparison.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tabs.addTab(self.tbl_comparison, "Algorithm Benchmark")

        return tabs

    def load_file(self, file_path: str):
        """Loads a signal CSV file."""
        try:
            self.reader = SignalCSVReader(file_path, sampling_rate=self.spin_fs.value())
            meta = self.reader.get_metadata()

            if not meta.is_valid:
                QMessageBox.critical(self, "Error Loading File", f"Failed to parse CSV: {meta.error_message}")
                return

            channels = meta.channels
            self.combo_sensor.clear()
            self.combo_sensor.addItems(channels if channels else ["Sensor 1", "Sensor 2"])

            file_size_str = format_file_size(meta.file_size_bytes)
            dur_str = format_time_duration(meta.estimated_duration_s)
            self.lbl_file_info.setText(
                f"📄 {meta.file_path.name} | {meta.total_samples:,} samples | ~{dur_str} | {file_size_str}"
            )

            # Load initial range for visualization
            self.current_sensor = channels[0] if channels else "Sensor 1"
            self._load_signal_data()
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            QMessageBox.critical(self, "Error", f"Could not load signal CSV:\n{e}")

    def _load_signal_data(self):
        if not self.reader:
            return

        meta = self.reader.get_metadata()
        max_samples = min(1000000, meta.total_samples)
        timestamps, data = self.reader.read_range(0, max_samples, self.current_sensor)

        self.raw_timestamps = timestamps
        self.raw_signal = data

        self.overview_plot.set_overview_data(timestamps, data)
        self.signal_plot.plot_signals(timestamps, raw_signal=data)

    def _on_open_file_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Industrial Signal CSV", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.load_file(file_path)

    def _on_load_gt_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Ground Truth Annotations", "", "CSV/JSON Files (*.csv *.json)")
        if file_path:
            self.ground_truth = GroundTruth(file_path)
            QMessageBox.information(self, "Ground Truth Loaded", f"Successfully loaded {len(self.ground_truth.get_timestamps())} ground truth peak annotations.")

    def _on_generate_sample_clicked(self):
        sample_path = Path("data/sample/industrial_sensor_signal_120s.csv")
        gt_path = Path("data/ground_truth/ground_truth_peaks_120s.csv")
        if sample_path.exists():
            self.load_file(str(sample_path))
            if gt_path.exists():
                self.ground_truth = GroundTruth(str(gt_path))
            QMessageBox.information(self, "Sample Data Loaded", "Loaded 120-second synthetic dataset with ground-truth peak annotations.")

    def _on_sensor_changed(self, index: int):
        if self.combo_sensor.currentText():
            self.current_sensor = self.combo_sensor.currentText()
            self._load_signal_data()

    def _on_fs_changed(self, val: float):
        self.config.sampling_rate = val

    def _on_overview_range_changed(self, start_s: float, end_s: float):
        if self.reader and len(self.raw_timestamps) > 0:
            start_idx = int(start_s * self.config.sampling_rate)
            end_idx = int(end_s * self.config.sampling_rate)
            ts, sig = self.reader.read_range(start_idx, end_idx, self.current_sensor)
            if self.processed_payload:
                self.signal_plot.plot_signals(ts, raw_signal=sig, processed_signal=sig)
            else:
                self.signal_plot.plot_signals(ts, raw_signal=sig)

    def _ensure_processed(self):
        """Helper that ensures peak detection has been run on current signal data."""
        if self.processed_payload is not None:
            return True
        if len(self.raw_signal) == 0:
            QMessageBox.warning(self, "No Signal Loaded", "Please load a signal CSV file first.")
            return False

        # Execute synchronous detection fallback for immediate results
        self._update_config_from_ui()
        pipeline = PreprocessingPipeline(self.config)
        proc_dict = pipeline.process(self.raw_signal)

        detector = PeakDetector(self.config)
        peaks_df, threshold_curve = detector.detect(proc_dict['smoothed'], baseline=proc_dict['baseline'])

        self.processed_payload = {
            "timestamps": self.raw_timestamps,
            "raw": proc_dict["raw"],
            "baseline": proc_dict["baseline"],
            "baseline_removed": proc_dict["baseline_removed"],
            "filtered": proc_dict["filtered"],
            "smoothed": proc_dict["smoothed"],
            "threshold_curve": threshold_curve,
            "peaks_df": peaks_df,
            "fft_freqs": np.array([]),
            "fft_mag": np.array([]),
            "fft_psd": np.array([]),
            "freq_features": {},
            "dominant_freqs": {}
        }
        self.signal_plot.plot_signals(
            time_arr=self.raw_timestamps,
            raw_signal=proc_dict["raw"],
            processed_signal=proc_dict["smoothed"],
            baseline=proc_dict["baseline"],
            threshold=threshold_curve,
            peaks_df=peaks_df
        )
        self._populate_peaks_table(peaks_df)
        dur = self.raw_timestamps[-1] - self.raw_timestamps[0] if len(self.raw_timestamps) > 0 else 1.0
        stats = quantify_peaks(peaks_df, total_duration_s=dur, threshold=self.config.peak_height)
        self._update_quant_stats_tab(stats)
        return True

    def _update_config_from_ui(self):
        self.config.sampling_rate = self.spin_fs.value()
        self.config.baseline_method = self.combo_baseline.currentText()
        self.config.smoothing_method = self.combo_smoothing.currentText()
        self.config.filter_type = self.combo_filter.currentText()
        self.config.peak_height = self.spin_height.value()
        self.config.peak_prominence = self.spin_prominence.value()
        self.config.peak_distance = self.spin_distance.value()
        self.config.use_adaptive_threshold = self.chk_adaptive.isChecked()
        self.config.resolve_overlapping_peaks = self.chk_overlap.isChecked()

    def run_processing(self):
        """Executes signal preprocessing and peak detection asynchronously in background thread."""
        if len(self.raw_signal) == 0:
            QMessageBox.warning(self, "No Signal Loaded", "Please load a signal CSV file first.")
            return

        self._update_config_from_ui()

        worker = ProcessingWorker(self.raw_timestamps, self.raw_signal, self.config)
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.result.connect(self._on_worker_result)
        worker.signals.error.connect(self._on_worker_error)

        self.progress_bar.setValue(0)
        self.thread_pool.start(worker)

    @Slot(int, str)
    def _on_worker_progress(self, percent: int, message: str):
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"%p% - {message}")

    @Slot(object)
    def _on_worker_result(self, payload: Dict[str, Any]):
        self.processed_payload = payload
        peaks_df = payload["peaks_df"]

        # 1. Update Signal Plot
        self.signal_plot.plot_signals(
            time_arr=payload["timestamps"],
            raw_signal=payload["raw"],
            processed_signal=payload["smoothed"],
            baseline=payload["baseline"],
            threshold=payload["threshold_curve"],
            peaks_df=peaks_df
        )

        # 2. Update Peaks Table
        self._populate_peaks_table(peaks_df)

        # 3. Update Quantification Stats Tab
        dur = payload["timestamps"][-1] - payload["timestamps"][0] if len(payload["timestamps"]) > 0 else 1.0
        stats = quantify_peaks(peaks_df, total_duration_s=dur, threshold=self.config.peak_height)
        self._update_quant_stats_tab(stats)

        # 4. Update Frequency Analysis Tab
        self._populate_freq_table(payload.get("dominant_freqs", {}), payload.get("freq_features", {}))

        QMessageBox.information(self, "Processing Finished", f"Peak Detection Complete!\nTotal Peaks Detected: {len(peaks_df)}")

    @Slot(str)
    def _on_worker_error(self, err_msg: str):
        QMessageBox.critical(self, "Processing Error", f"An error occurred during signal processing:\n{err_msg}")

    def _populate_peaks_table(self, peaks_df: pd.DataFrame):
        self.tbl_peaks.setRowCount(0)
        if peaks_df is None or peaks_df.empty:
            return

        classifier = PeakClassifier(low_threshold=1.0, high_threshold=3.0)
        peaks_df = classifier.classify(peaks_df)

        self.tbl_peaks.setRowCount(len(peaks_df))
        for i, row in peaks_df.iterrows():
            self.tbl_peaks.setItem(i, 0, QTableWidgetItem(str(int(row.get("peak_index", i)))))
            self.tbl_peaks.setItem(i, 1, QTableWidgetItem(f"{row.get('timestamp', 0.0):.4f}"))
            self.tbl_peaks.setItem(i, 2, QTableWidgetItem(f"{row.get('amplitude', 0.0):.4f}"))
            self.tbl_peaks.setItem(i, 3, QTableWidgetItem(f"{row.get('prominence', 0.0):.4f}"))
            self.tbl_peaks.setItem(i, 4, QTableWidgetItem(f"{row.get('width', 0.0):.2f}"))
            self.tbl_peaks.setItem(i, 5, QTableWidgetItem(f"{row.get('area', 0.0):.4f}"))
            self.tbl_peaks.setItem(i, 6, QTableWidgetItem(str(row.get('category', 'N/A'))))
            self.tbl_peaks.setItem(i, 7, QTableWidgetItem(f"{row.get('snr', 0.0):.2f}"))
            self.tbl_peaks.setItem(i, 8, QTableWidgetItem(f"{row.get('confidence', 0.0):.2f}"))

    def _update_quant_stats_tab(self, stats: Dict[str, Any]):
        html = f"""
        <h2>Peak Quantification & Statistical Summary</h2>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%;">
            <tr><td><b>Total Peaks Detected</b></td><td>{stats.get('total_peaks', 0)}</td></tr>
            <tr><td><b>Peaks Above Height Threshold</b></td><td>{stats.get('above_threshold', 0)}</td></tr>
            <tr><td><b>Peaks Below Threshold</b></td><td>{stats.get('below_threshold', 0)}</td></tr>
            <tr><td><b>Peak Density (Peaks / Minute)</b></td><td>{stats.get('peaks_per_min', 0.0):.2f}</td></tr>
            <tr><td><b>Peak Density (Peaks / Hour)</b></td><td>{stats.get('peaks_per_hour', 0.0):.2f}</td></tr>
            <tr><td><b>Mean Amplitude</b></td><td>{stats.get('mean_amplitude', 0.0):.4f} V</td></tr>
            <tr><td><b>Max Amplitude</b></td><td>{stats.get('max_amplitude', 0.0):.4f} V</td></tr>
            <tr><td><b>Mean Prominence</b></td><td>{stats.get('mean_prominence', 0.0):.4f}</td></tr>
            <tr><td><b>Mean Width</b></td><td>{stats.get('mean_width', 0.0):.2f} samples</td></tr>
        </table>
        """
        self.lbl_quant_summary.setText(html)

    def _populate_freq_table(self, dom_freqs: Dict[str, Any], freq_features: Dict[str, Any]):
        freq_list = dom_freqs.get("dominant_freqs", [])
        mag_list = dom_freqs.get("magnitudes", [])

        self.tbl_freq.setRowCount(len(freq_list))
        for i, (f, m) in enumerate(zip(freq_list, mag_list)):
            self.tbl_freq.setItem(i, 0, QTableWidgetItem(f"{f:.2f} Hz"))
            self.tbl_freq.setItem(i, 1, QTableWidgetItem(f"{m:.6f}"))

    def run_evaluation(self):
        """Runs Precision, Recall, F1 evaluation against ground truth annotations."""
        if not self.ground_truth:
            QMessageBox.warning(self, "No Ground Truth Loaded", "Please click 'Load Ground Truth' first.")
            return
        if not self._ensure_processed():
            return

        peaks_df = self.processed_payload["peaks_df"]
        det_ts = peaks_df["timestamp"].to_numpy() if not peaks_df.empty else np.array([])
        gt_ts = self.ground_truth.get_timestamps()

        metrics = evaluate_peak_detection(det_ts, gt_ts, match_tolerance_ms=20.0)

        html = f"""
        <h2>Ground Truth Scientific Evaluation</h2>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%;">
            <tr><td><b>Ground Truth Peaks</b></td><td>{len(gt_ts)}</td></tr>
            <tr><td><b>Algorithm Detected Peaks</b></td><td>{len(det_ts)}</td></tr>
            <tr><td><b>True Positives (TP)</b></td><td><b style="color: green;">{metrics['true_positives']}</b></td></tr>
            <tr><td><b>False Positives (FP)</b></td><td><b style="color: red;">{metrics['false_positives']}</b></td></tr>
            <tr><td><b>False Negatives (FN)</b></td><td><b style="color: orange;">{metrics['false_negatives']}</b></td></tr>
            <tr><td><b>Precision</b></td><td><b>{metrics['precision']:.4f}</b></td></tr>
            <tr><td><b>Recall (Detection Rate)</b></td><td><b>{metrics['recall']:.4f}</b></td></tr>
            <tr><td><b>F1 Score</b></td><td><b style="color: #89b4fa;">{metrics['f1_score']:.4f}</b></td></tr>
            <tr><td><b>False Detection Rate</b></td><td>{metrics['false_detection_rate']:.4f}</td></tr>
        </table>
        """
        self.lbl_eval_metrics.setText(html)
        self.results_tabs.setCurrentIndex(3)

    def run_algorithm_comparison(self):
        """Runs benchmark comparison across Algorithm Methods A, B, C, D."""
        if len(self.raw_signal) == 0:
            QMessageBox.warning(self, "No Signal Loaded", "Please load a signal CSV file first.")
            return

        gt_ts = self.ground_truth.get_timestamps() if self.ground_truth else np.array([])
        comp_df = AlgorithmComparison.compare_methods(
            self.raw_signal,
            gt_ts,
            sampling_rate=self.spin_fs.value()
        )

        self.tbl_comparison.setRowCount(len(comp_df))
        for i, row in comp_df.iterrows():
            self.tbl_comparison.setItem(i, 0, QTableWidgetItem(str(row["Algorithm"])))
            self.tbl_comparison.setItem(i, 1, QTableWidgetItem(str(row["Detected Peaks"])))
            self.tbl_comparison.setItem(i, 2, QTableWidgetItem(str(row["True Positives"])))
            self.tbl_comparison.setItem(i, 3, QTableWidgetItem(str(row["False Positives"])))
            self.tbl_comparison.setItem(i, 4, QTableWidgetItem(str(row["False Negatives"])))
            self.tbl_comparison.setItem(i, 5, QTableWidgetItem(f"{row['Precision']:.4f}"))
            self.tbl_comparison.setItem(i, 6, QTableWidgetItem(f"{row['Recall']:.4f}"))
            self.tbl_comparison.setItem(i, 7, QTableWidgetItem(f"{row['F1 Score']:.4f}"))
            self.tbl_comparison.setItem(i, 8, QTableWidgetItem(f"{row['Runtime (ms)']:.2f}"))

        self.results_tabs.setCurrentIndex(5)

    def run_concentration_analysis(self):
        """Fits regression model and estimates fluorophore concentration from detected peak count (Objective 3)."""
        if not self._ensure_processed():
            return

        peak_cnt = len(self.processed_payload["peaks_df"])
        estimator = ConcentrationEstimator()
        result = estimator.estimate_from_peak_count(peak_cnt)

        cal_df = estimator.cal.df

        cal_table_rows = ""
        if not cal_df.empty:
            for idx, r in cal_df.iterrows():
                cal_table_rows += f"<tr><td>{r['tumour_size_ml']}</td><td>{r['peak_count']}</td><td><b>{r['concentration']}</b></td></tr>"

        html = f"""
        <h2>Objective 3 — Fluorophore Concentration & Tumour Size Analysis</h2>
        <div style="display: flex; gap: 20px;">
            <div style="flex: 1;">
                <h3>1. Calibration Model Dataset (Measurements)</h3>
                <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%; background-color: #181825;">
                    <tr style="background-color: #313244; color: #89b4fa;">
                        <th>Tumour Size (mL)</th>
                        <th>Peak Count</th>
                        <th>Concentration Category</th>
                    </tr>
                    {cal_table_rows}
                </table>
                <p><b>Linear Dependence Model</b>: Peak Count $\\approx$ 123 $\\times$ Tumour Size (mL) - 10 (<i>R² = {result['regression_r2']:.4f}</i>)</p>
            </div>
            <div style="flex: 1; background-color: #181825; padding: 15px; border-radius: 8px; border: 1px solid #45475a;">
                <h3>2. Current Signal Prediction Results</h3>
                <table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%;">
                    <tr><td><b>Detected Peak Count</b></td><td><b style="color: #a6e3a1; font-size: 16px;">{result['detected_peak_count']}</b></td></tr>
                    <tr><td><b>Estimated Tumour Size</b></td><td><b style="color: #89b4fa; font-size: 16px;">{result['estimated_tumour_size_ml']} mL</b></td></tr>
                    <tr><td><b>Estimated Concentration Tier</b></td><td><b style="color: #f39c12; font-size: 18px;">{result['estimated_concentration']}</b></td></tr>
                    <tr><td><b>Regression Model Type</b></td><td>{result['regression_model']}</td></tr>
                </table>
                <br>
                <p style="color: #a6adc8;"><i>{result['note']}</i></p>
            </div>
        </div>
        """
        self.lbl_concentration.setText(html)
        self.results_tabs.setCurrentIndex(4)

    def _export_peaks_csv(self):
        if not self._ensure_processed():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Detected Peaks CSV", "output/exports/detected_peaks.csv", "CSV Files (*.csv)")
        if path:
            ResultsExporter.export_peaks_csv(self.processed_payload["peaks_df"], Path(path))
            QMessageBox.information(self, "Export Complete", f"Peaks exported successfully to:\n{path}")

    def _generate_report(self):
        if not self._ensure_processed():
            return

        path, _ = QFileDialog.getSaveFileName(self, "Generate Scientific Report", "output/reports/signal_analysis_report.md", "Markdown Files (*.md)")
        if path:
            meta = self.reader.get_metadata().__dict__ if self.reader else {}
            dur = self.raw_timestamps[-1] - self.raw_timestamps[0] if len(self.raw_timestamps) > 0 else 1.0
            stats = quantify_peaks(self.processed_payload["peaks_df"], total_duration_s=dur, threshold=self.config.peak_height)
            gt_ts = self.ground_truth.get_timestamps() if self.ground_truth else np.array([])
            det_ts = self.processed_payload["peaks_df"]["timestamp"].to_numpy() if not self.processed_payload["peaks_df"].empty else np.array([])

            eval_metrics = evaluate_peak_detection(det_ts, gt_ts) if len(gt_ts) > 0 else None
            estimator = ConcentrationEstimator()
            conc_res = estimator.estimate_from_peak_count(len(det_ts))

            ReportGenerator.generate_report(
                metadata=meta,
                config=self.config.to_dict(),
                stats=stats,
                evaluation_metrics=eval_metrics,
                concentration_results=conc_res,
                output_path=Path(path)
            )
            QMessageBox.information(self, "Report Generated", f"Scientific report successfully generated:\n{path}")
