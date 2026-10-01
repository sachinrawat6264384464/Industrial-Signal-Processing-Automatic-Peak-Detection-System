"""
Signal CSV Reader for high-performance large-file processing.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Iterator
from .signal_metadata import SignalMetadata
from .chunk_reader import ChunkReader
from ..utils.logging_config import logger
from ..utils.validators import validate_file_path

class SignalCSVReader:
    """Reads large CSV files containing multi-channel sensor signals in chunks."""

    def __init__(self, file_path: str, sampling_rate: float = 50000.0, chunk_size: int = 500000):
        self.file_path = validate_file_path(file_path)
        self.sampling_rate = float(sampling_rate)
        self.chunk_size = int(chunk_size)
        self._metadata: Optional[SignalMetadata] = None
        self._inspect_file()

    def _inspect_file(self) -> None:
        """Inspects CSV headers, estimates line counts, and constructs metadata."""
        try:
            # Read first few rows to get header/column names
            sample_df = pd.read_csv(self.file_path, nrows=5)
            channels = [str(c).strip() for c in sample_df.columns]
            
            file_size = os.path.getsize(self.file_path)
            
            # Fast estimation of total samples using average line byte length
            with open(self.file_path, 'rb') as f:
                header = f.readline()
                sample_lines = [f.readline() for _ in range(100)]
                sample_lines = [l for l in sample_lines if l]
                
            if sample_lines:
                avg_line_len = sum(len(l) for l in sample_lines) / len(sample_lines)
                header_len = len(header)
                estimated_data_bytes = max(0, file_size - header_len)
                total_samples = int(estimated_data_bytes / max(1.0, avg_line_len))
            else:
                total_samples = 0

            self._metadata = SignalMetadata(
                file_path=self.file_path,
                file_size_bytes=file_size,
                total_samples=total_samples,
                sampling_rate=self.sampling_rate,
                channels=channels,
                estimated_duration_s=total_samples / self.sampling_rate if self.sampling_rate > 0 else 0.0
            )
            logger.info(f"Loaded metadata for {self.file_path.name}: {total_samples} samples (~{self._metadata.estimated_duration_s:.1f}s), channels: {channels}")
        except Exception as e:
            logger.error(f"Failed to inspect signal file {self.file_path}: {e}")
            self._metadata = SignalMetadata(
                file_path=self.file_path,
                file_size_bytes=os.path.getsize(self.file_path) if self.file_path.exists() else 0,
                total_samples=0,
                sampling_rate=self.sampling_rate,
                channels=[],
                estimated_duration_s=0.0,
                is_valid=False,
                error_message=str(e)
            )

    def get_metadata(self) -> SignalMetadata:
        """Returns metadata for the file."""
        return self._metadata

    def get_channel_names(self) -> List[str]:
        """Returns list of sensor channel names in the CSV."""
        return self._metadata.channels if self._metadata else []

    def get_sample_rate(self) -> float:
        """Returns current configured sampling rate."""
        return self.sampling_rate

    def estimate_duration(self) -> float:
        """Returns estimated duration in seconds."""
        return self._metadata.estimated_duration_s if self._metadata else 0.0

    def iter_chunks(self, channels: Optional[List[str]] = None) -> Iterator[pd.DataFrame]:
        """Iterates through file in chunks for specified channels."""
        reader = ChunkReader(self.file_path, chunk_size=self.chunk_size, target_columns=channels)
        return reader.iter_chunks()

    def read_range(self, start_idx: int, end_idx: int, channel: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads a specific sample index range [start_idx, end_idx) for a single channel.
        Returns tuple of (timestamps, signal_values).
        Memory efficient range reader without loading full file.
        """
        if start_idx < 0:
            start_idx = 0
        if end_idx <= start_idx:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        skip_rows = start_idx + 1  # 1 for header
        nrows = end_idx - start_idx

        try:
            df = pd.read_csv(
                self.file_path,
                skiprows=range(1, skip_rows) if skip_rows > 1 else None,
                nrows=nrows,
                usecols=[channel],
                dtype=np.float32,
                on_bad_lines='skip'
            )
            data = df[channel].to_numpy(dtype=np.float64)
            # handle NaNs
            nan_mask = np.isnan(data) | np.isinf(data)
            if np.any(nan_mask):
                data[nan_mask] = 0.0

            timestamps = (np.arange(start_idx, start_idx + len(data)) / self.sampling_rate).astype(np.float64)
            return timestamps, data
        except Exception as e:
            logger.error(f"Error reading range [{start_idx}:{end_idx}] for channel {channel}: {e}")
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
