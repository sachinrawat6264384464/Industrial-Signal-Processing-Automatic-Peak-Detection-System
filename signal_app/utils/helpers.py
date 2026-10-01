"""
General Helper Utilities for Time Formatting, File Sizes, and Array Helpers.
"""
import time

def format_time_duration(seconds: float) -> str:
    """Formats time duration in seconds to HH:MM:SS format."""
    if seconds < 0:
        return "00:00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"

def format_file_size(size_bytes: int) -> str:
    """Formats bytes to human readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def sample_to_timestamp(sample_idx: int, sampling_rate: float) -> float:
    """Converts sample index to timestamp in seconds."""
    return sample_idx / sampling_rate

def timestamp_to_sample(timestamp_s: float, sampling_rate: float) -> int:
    """Converts timestamp in seconds to sample index."""
    return max(0, int(timestamp_s * sampling_rate))
