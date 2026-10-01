"""
Overview Plot Widget with Interactive Time Range Selector (LinearRegionItem).
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal
import pyqtgraph as pg
import numpy as np
from typing import Optional
from .downsampling import min_max_downsample

class OverviewPlotWidget(QWidget):
    """Provides a compressed full-signal overview plot with interactive range selector."""

    range_changed = Signal(float, float)  # Emits (start_time_s, end_time_s)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget(title="Full Signal Overview (Drag region to zoom)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Amp')
        self.plot_widget.setFixedHeight(120)

        self.layout.addWidget(self.plot_widget)

        self.overview_curve = self.plot_widget.plot(pen=pg.mkPen('#89b4fa', width=1))

        # Interactive Linear Region ROI
        self.region = pg.LinearRegionItem(values=[0, 10], brush=pg.mkBrush(137, 180, 250, 50))
        self.region.setZValue(10)
        self.plot_widget.addItem(self.region)

        self.region.sigRegionChanged.connect(self._on_region_changed)

    def set_overview_data(self, timestamps: np.ndarray, signal: np.ndarray, max_points: int = 2000):
        """Sets the overview dataset (downsampled for fast overview)."""
        if timestamps is None or len(timestamps) == 0:
            self.overview_curve.clear()
            return

        t_ds, s_ds = min_max_downsample(timestamps, signal, max_points)
        self.overview_curve.setData(t_ds, s_ds)

        # Set default region to first 10% or full signal if short
        t_min, t_max = timestamps[0], timestamps[-1]
        default_span = min(t_max - t_min, max(1.0, (t_max - t_min) * 0.2))
        self.region.setRegion([t_min, t_min + default_span])
        self.plot_widget.setXRange(t_min, t_max, padding=0.01)

    def set_range(self, start_s: float, end_s: float):
        """Sets the selected region range programmatically."""
        self.region.setRegion([start_s, end_s])

    def _on_region_changed(self):
        min_val, max_val = self.region.getRegion()
        self.range_changed.emit(min_val, max_val)
