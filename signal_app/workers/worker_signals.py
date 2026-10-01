"""
PySide6 Custom Worker Signals Module.
"""
from PySide6.QtCore import QObject, Signal
from typing import Any, Dict

class WorkerSignals(QObject):
    """Defines signals available for worker threads."""
    started = Signal()
    progress = Signal(int, str)  # (percent: 0-100, message: str)
    result = Signal(object)      # result payload
    finished = Signal()
    error = Signal(str)         # error message
