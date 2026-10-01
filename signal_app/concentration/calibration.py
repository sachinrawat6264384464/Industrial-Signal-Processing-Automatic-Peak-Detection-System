"""
Calibration Dataset Module for Fluorophore/Tumour Analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

class CalibrationData:
    """Manages calibration dataset mapping Tumour Size (mL) to Peak Count and Concentration Category."""

    def __init__(self, file_path: Optional[str] = None):
        self.df: pd.DataFrame = pd.DataFrame()
        if file_path and Path(file_path).exists():
            self.load(Path(file_path))
        else:
            self.load_default_sample()

    def load(self, file_path: Path) -> None:
        """Loads calibration CSV file."""
        self.df = pd.read_csv(file_path)

    def load_default_sample(self) -> None:
        """Loads default synthetic calibration curve data for initial demonstration."""
        data = {
            "tumour_size_ml": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            "peak_count": [45, 110, 175, 240, 295, 360, 415, 480, 530, 600],
            "concentration": ["Low", "Low", "Low", "Medium", "Medium", "Medium", "High", "High", "High", "High"]
        }
        self.df = pd.DataFrame(data)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (tumour_size_ml, peak_count) arrays."""
        if self.df.empty:
            return np.array([]), np.array([])
        return self.df["tumour_size_ml"].to_numpy(), self.df["peak_count"].to_numpy()
