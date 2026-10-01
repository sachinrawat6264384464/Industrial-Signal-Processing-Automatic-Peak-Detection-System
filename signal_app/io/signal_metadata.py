"""
Metadata structure for signal files.
"""
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class SignalMetadata:
    """Class holding file and signal metadata."""
    file_path: Path
    file_size_bytes: int
    total_samples: int
    sampling_rate: float
    channels: List[str]
    estimated_duration_s: float
    is_valid: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.sampling_rate > 0 and self.total_samples >= 0:
            self.estimated_duration_s = self.total_samples / self.sampling_rate
        else:
            self.estimated_duration_s = 0.0
