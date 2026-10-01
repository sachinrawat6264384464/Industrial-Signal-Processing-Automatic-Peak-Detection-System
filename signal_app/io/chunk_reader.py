"""
Chunk Reader for streaming large CSV files efficiently.
"""
from typing import Iterator, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from ..utils.logging_config import logger

class ChunkReader:
    """Streams data from CSV files in memory-efficient chunks."""

    def __init__(self, file_path: Path, chunk_size: int = 500000, target_columns: Optional[List[str]] = None):
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.target_columns = target_columns

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """Yields chunks of DataFrame with optimized memory dtypes."""
        try:
            reader = pd.read_csv(
                self.file_path,
                chunksize=self.chunk_size,
                usecols=self.target_columns,
                dtype=np.float32,
                on_bad_lines='skip'
            )
            for chunk in reader:
                # Clean up missing or infinite values
                chunk = chunk.ffill().bfill().fillna(0.0)
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming chunks from {self.file_path}: {e}")
            raise
