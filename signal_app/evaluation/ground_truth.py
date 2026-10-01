"""
Ground Truth Loader and Representation Module.
Supports CSV/JSON ground truth files containing ground truth peak timestamps or indices.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..utils.logging_config import logger

class GroundTruth:
    """Manages ground-truth peak annotations for scientific validation."""

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = Path(file_path) if file_path else None
        self.gt_timestamps: np.ndarray = np.array([])
        self.gt_amplitudes: np.ndarray = np.array([])
        if self.file_path and self.file_path.exists():
            self.load(self.file_path)

    def load(self, file_path: Path) -> None:
        """Loads ground truth peak locations from CSV or JSON file."""
        self.file_path = Path(file_path)
        try:
            if self.file_path.suffix.lower() == ".json":
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.gt_timestamps = np.array([item.get("timestamp", 0.0) for item in data], dtype=np.float64)
                    self.gt_amplitudes = np.array([item.get("amplitude", 1.0) for item in data], dtype=np.float64)
                elif isinstance(data, dict):
                    self.gt_timestamps = np.array(data.get("timestamps", []), dtype=np.float64)
                    self.gt_amplitudes = np.array(data.get("amplitudes", [1.0]*len(self.gt_timestamps)), dtype=np.float64)
            else: # CSV format
                df = pd.read_csv(self.file_path)
                if "timestamp" in df.columns:
                    self.gt_timestamps = df["timestamp"].to_numpy(dtype=np.float64)
                elif "sample_index" in df.columns:
                    fs = 50000.0
                    self.gt_timestamps = df["sample_index"].to_numpy(dtype=np.float64) / fs
                else:
                    self.gt_timestamps = df.iloc[:, 0].to_numpy(dtype=np.float64)

                self.gt_amplitudes = df["amplitude"].to_numpy(dtype=np.float64) if "amplitude" in df.columns else np.ones_like(self.gt_timestamps)

            self.gt_timestamps = np.sort(self.gt_timestamps)
            logger.info(f"Loaded ground truth file {self.file_path.name}: {len(self.gt_timestamps)} peaks")
        except Exception as e:
            logger.error(f"Failed to load ground truth file {file_path}: {e}")
            self.gt_timestamps = np.array([])
            self.gt_amplitudes = np.array([])

    def get_timestamps(self) -> np.ndarray:
        return self.gt_timestamps
