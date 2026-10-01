"""
Main Window Component for Industrial Signal Analyzer Application.
"""
from PySide6.QtWidgets import QMainWindow, QStatusBar, QMenuBar, QMessageBox
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt
from .dashboard import SignalAnalyzerDashboard
import sys

class SignalAnalyzerMainWindow(QMainWindow):
    """Main desktop application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Industrial Signal Analyzer — UEF & Marginum 2026")
        self.resize(1400, 900)

        # Apply dark aesthetic stylesheet
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', Roboto, sans-serif;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 12px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #45475a;
                border-color: #89b4fa;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #11111b;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 4px;
            }
            QTableWidget {
                background-color: #11111b;
                gridline-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
            }
            QHeaderView::section {
                background-color: #181825;
                color: #89b4fa;
                padding: 4px;
                border: 1px solid #313244;
                font-weight: bold;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
            }
            QTabBar::tab {
                background-color: #181825;
                color: #a6adc8;
                padding: 8px 16px;
                border: 1px solid #313244;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #313244;
                color: #89b4fa;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #45475a;
                border-radius: 4px;
                text-align: center;
                background-color: #11111b;
                color: #cdd6f4;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
            }
        """)

        # Main Central Dashboard
        self.dashboard = SignalAnalyzerDashboard(self)
        self.setCentralWidget(self.dashboard)

        # Status bar
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

        # Menu bar
        self._create_menu_bar()

    def _create_menu_bar(self):
        menubar = self.menuBar()

        # File Menu
        menu_file = menubar.addMenu("&File")

        action_open = QAction("Open Signal CSV...", self)
        action_open.setShortcut("Ctrl+O")
        action_open.triggered.connect(self.dashboard._on_open_file_clicked)
        menu_file.addAction(action_open)

        action_gt = QAction("Load Ground Truth...", self)
        action_gt.triggered.connect(self.dashboard._on_load_gt_clicked)
        menu_file.addAction(action_gt)

        menu_file.addSeparator()

        action_exit = QAction("Exit", self)
        action_exit.triggered.connect(self.close)
        menu_file.addAction(action_exit)

        # Actions Menu
        menu_actions = menubar.addMenu("&Actions")

        action_run = QAction("Run Processing & Peak Detection", self)
        action_run.setShortcut("Ctrl+R")
        action_run.triggered.connect(self.dashboard.run_processing)
        menu_actions.addAction(action_run)

        action_eval = QAction("Run Ground Truth Evaluation", self)
        action_eval.triggered.connect(self.dashboard.run_evaluation)
        menu_actions.addAction(action_eval)

        action_compare = QAction("Benchmark Algorithm Comparison", self)
        action_compare.triggered.connect(self.dashboard.run_algorithm_comparison)
        menu_actions.addAction(action_compare)

        action_conc = QAction("Fluorophore Concentration Analysis", self)
        action_conc.triggered.connect(self.dashboard.run_concentration_analysis)
        menu_actions.addAction(action_conc)

        # Help Menu
        menu_help = menubar.addMenu("&Help")
        action_about = QAction("About Industrial Signal Analyzer", self)
        action_about.triggered.connect(self._show_about)
        menu_help.addAction(action_about)

    def _show_about(self):
        QMessageBox.about(
            self,
            "About Industrial Signal Analyzer",
            "<h3>Industrial Signal Processing & Peak Analyzer</h3>"
            "<p><b>Version</b>: 1.0.0 (UEF & Marginum 2026)</p>"
            "<p>High-performance desktop application for processing multi-gigabyte sensor signals, "
            "peak detection, adaptive thresholding, FFT spectral analysis, ground truth evaluation, "
            "and fluorophore concentration estimation.</p>"
        )
