"""
Visualization package for signal plots and downsampling.
"""
from .downsampling import min_max_downsample, envelope_downsample
from .signal_plot import SignalPlotWidget
from .overview_plot import OverviewPlotWidget
from .peak_overlay import PeakOverlay

__all__ = ["min_max_downsample", "envelope_downsample", "SignalPlotWidget", "OverviewPlotWidget", "PeakOverlay"]
