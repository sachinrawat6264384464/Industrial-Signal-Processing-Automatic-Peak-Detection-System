"""
Interactive PyQtGraph Signal Plot Widget.
Displays original signal, processed signal, baseline, adaptive threshold, and peak markers.
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from .downsampling import min_max_downsample
from .peak_overlay import PeakOverlay

# Modern dark aesthetic styling
pg.setConfigOption('background', '#1e1e2e')
pg.setConfigOption('foreground', '#cdd6f4')
pg.setConfigOptions(antialias=True)

class SignalPlotWidget(QWidget):
    """Main interactive plot for displaying raw signal, baseline, processed signal, and peaks."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Amplitude', units='V')
        self.plot_widget.addLegend(offset=(10, 10))

        self.layout.addWidget(self.plot_widget)

        # Plot curves
        self.raw_curve = self.plot_widget.plot(pen=pg.mkPen('#89b4fa', width=1.5), name='Raw Signal')
        self.processed_curve = self.plot_widget.plot(pen=pg.mkPen('#a6e3a1', width=1.5), name='Processed Signal')
        self.baseline_curve = self.plot_widget.plot(pen=pg.mkPen('#f9e2af', width=1.5, style=pg.QtCore.Qt.DashLine), name='Baseline')
        self.threshold_curve = self.plot_widget.plot(pen=pg.mkPen('#f38ba8', width=1.5, style=pg.QtCore.Qt.DotLine), name='Threshold')
        
        # Peak Scatter plot item
        self.peak_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen('w', width=1), name='Detected Peaks')
        self.plot_widget.addItem(self.peak_scatter)

        # Cache data
        self._raw_time: Optional[np.ndarray] = None
        self._raw_signal: Optional[np.ndarray] = None

    def plot_signals(
        self,
        time_arr: np.ndarray,
        raw_signal: Optional[np.ndarray] = None,
        processed_signal: Optional[np.ndarray] = None,
        baseline: Optional[np.ndarray] = None,
        threshold: Optional[np.ndarray] = None,
        peaks_df: Optional[pd.DataFrame] = None,
        downsample: bool = True,
        max_points: int = 5000
    ):
        """Updates all plot curves and peak markers."""
        if time_arr is None or len(time_arr) == 0:
            self.clear_plot()
            return

        self._raw_time = time_arr
        self._raw_signal = raw_signal

        # Downsample for fast responsive rendering
        if downsample and len(time_arr) > max_points:
            t_ds, s_ds = min_max_downsample(time_arr, raw_signal if raw_signal is not None else np.zeros_like(time_arr), max_points)
        else:
            t_ds, s_ds = time_arr, raw_signal

        # Raw signal
        if raw_signal is not None:
            self.raw_curve.setData(t_ds, s_ds)
            self.raw_curve.setVisible(True)
        else:
            self.raw_curve.setVisible(False)

        # Processed signal
        if processed_signal is not None:
            if downsample and len(time_arr) > max_points:
                t_p, s_p = min_max_downsample(time_arr, processed_signal, max_points)
            else:
                t_p, s_p = time_arr, processed_signal
            self.processed_curve.setData(t_p, s_p)
            self.processed_curve.setVisible(True)
        else:
            self.processed_curve.setVisible(False)

        # Baseline
        if baseline is not None:
            if downsample and len(time_arr) > max_points:
                t_b, s_b = min_max_downsample(time_arr, baseline, max_points)
            else:
                t_b, s_b = time_arr, baseline
            self.baseline_curve.setData(t_b, s_b)
            self.baseline_curve.setVisible(True)
        else:
            self.baseline_curve.setVisible(False)

        # Threshold
        if threshold is not None:
            if len(threshold) == 1 or np.isscalar(threshold):
                # Fixed threshold
                thresh_arr = np.full_like(time_arr, threshold)
                if downsample and len(time_arr) > max_points:
                    t_th, s_th = min_max_downsample(time_arr, thresh_arr, max_points)
                else:
                    t_th, s_th = time_arr, thresh_arr
            else:
                if downsample and len(time_arr) > max_points:
                    t_th, s_th = min_max_downsample(time_arr, threshold, max_points)
                else:
                    t_th, s_th = time_arr, threshold
            self.threshold_curve.setData(t_th, s_th)
            self.threshold_curve.setVisible(True)
        else:
            self.threshold_curve.setVisible(False)

        # Peaks
        if peaks_df is not None and not peaks_df.empty:
            scatter_data = PeakOverlay.prepare_peak_scatter(peaks_df)
            spots = []
            for x, y, col in zip(scatter_data["x"], scatter_data["y"], scatter_data["colors"]):
                spots.append({'pos': (x, y), 'brush': pg.mkBrush(col), 'symbol': 'o', 'size': 12})
            self.peak_scatter.setData(spots)
            self.peak_scatter.setVisible(True)
        else:
            self.peak_scatter.clear()
            self.peak_scatter.setVisible(False)

    def clear_plot(self):
        """Clears all curves and peak markers."""
        self.raw_curve.clear()
        self.processed_curve.clear()
        self.baseline_curve.clear()
        self.threshold_curve.clear()
        self.peak_scatter.clear()
