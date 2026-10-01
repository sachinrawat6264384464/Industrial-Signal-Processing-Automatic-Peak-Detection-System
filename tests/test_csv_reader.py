"""
Unit tests for SignalCSVReader and ChunkReader.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from signal_app.io.csv_reader import SignalCSVReader

@pytest.fixture
def temp_csv_file(tmp_path):
    """Creates a temporary CSV signal file."""
    csv_file = tmp_path / "test_signal.csv"
    data = {
        "Sensor 1": np.random.randn(10000),
        "Sensor 2": np.random.randn(10000) * 2.0
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)
    return csv_file

def test_csv_reader_metadata(temp_csv_file):
    reader = SignalCSVReader(str(temp_csv_file), sampling_rate=50000.0, chunk_size=2000)
    metadata = reader.get_metadata()
    assert metadata.is_valid is True
    assert "Sensor 1" in metadata.channels
    assert "Sensor 2" in metadata.channels
    assert reader.get_sample_rate() == 50000.0

def test_csv_reader_iter_chunks(temp_csv_file):
    reader = SignalCSVReader(str(temp_csv_file), sampling_rate=50000.0, chunk_size=3000)
    chunks = list(reader.iter_chunks(channels=["Sensor 1"]))
    assert len(chunks) == 4
    total_rows = sum(len(c) for c in chunks)
    assert total_rows == 10000

def test_csv_reader_read_range(temp_csv_file):
    reader = SignalCSVReader(str(temp_csv_file), sampling_rate=50000.0)
    timestamps, data = reader.read_range(100, 500, "Sensor 1")
    assert len(data) == 400
    assert len(timestamps) == 400
    assert timestamps[0] == pytest.approx(100 / 50000.0)
