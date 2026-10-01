"""
IO package for reading large CSV files and signal metadata.
"""
from .signal_metadata import SignalMetadata
from .chunk_reader import ChunkReader
from .csv_reader import SignalCSVReader

__all__ = ["SignalMetadata", "ChunkReader", "SignalCSVReader"]
