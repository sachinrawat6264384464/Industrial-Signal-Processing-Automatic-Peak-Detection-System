"""
Synthetic Signal Data Generator for Industrial Signal Analyzer.
Generates realistic multi-channel sensor signals with baseline drift, Gaussian noise, and known ground-truth peaks.
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any

def generate_gaussian_peak(x: np.ndarray, center: float, height: float, width: float) -> np.ndarray:
    """Generates a Gaussian peak profile."""
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)

def generate_synthetic_signal(
    duration_s: float = 120.0,
    sampling_rate: float = 50000.0,
    num_peaks: int = 30,
    noise_level: float = 0.05,
    baseline_amplitude: float = 0.5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Generates synthetic 2-channel signal and ground truth annotations.

    Returns:
        (timestamps, sensor_1, sensor_2, ground_truth_peaks)
    """
    np.random.seed(seed)
    total_samples = int(duration_s * sampling_rate)
    timestamps = np.linspace(0, duration_s, total_samples, endpoint=False)

    # 1. Baseline Drift (low frequency sine + polynomial trend)
    baseline_1 = baseline_amplitude * np.sin(2 * np.pi * 0.01 * timestamps) + 0.2 * (timestamps / duration_s) ** 2
    baseline_2 = baseline_amplitude * 1.2 * np.cos(2 * np.pi * 0.008 * timestamps) - 0.1 * (timestamps / duration_s)

    # 2. Gaussian Random Noise
    noise_1 = np.random.normal(0, noise_level, total_samples)
    noise_2 = np.random.normal(0, noise_level * 1.1, total_samples)

    # 3. Peak generation
    sensor_1 = baseline_1 + noise_1
    sensor_2 = baseline_2 + noise_2

    # Distribute peak centers across time
    peak_times = np.sort(np.random.uniform(5.0, duration_s - 5.0, num_peaks))
    
    # Introduce a couple of overlapping peak pairs
    if num_peaks >= 4:
        peak_times[1] = peak_times[0] + 0.08  # 80ms overlap
        peak_times[3] = peak_times[2] + 0.10  # 100ms overlap

    gt_peaks = []
    for i, p_time in enumerate(peak_times):
        height_1 = float(np.random.uniform(0.5, 4.0))
        height_2 = float(height_1 * np.random.uniform(0.7, 1.3))
        width = float(np.random.uniform(0.01, 0.04))  # 10ms to 40ms width

        idx = int(p_time * sampling_rate)
        sensor_1 += generate_gaussian_peak(timestamps, p_time, height_1, width)
        sensor_2 += generate_gaussian_peak(timestamps, p_time, height_2, width)

        gt_peaks.append({
            "peak_id": i + 1,
            "sample_index": idx,
            "timestamp": float(p_time),
            "amplitude_sensor1": height_1,
            "amplitude_sensor2": height_2,
            "width": width
        })

    return timestamps, sensor_1, sensor_2, gt_peaks

def main():
    """Generates synthetic CSV files into data/sample/ and data/ground_truth/."""
    data_dir = Path("data")
    sample_dir = data_dir / "sample"
    gt_dir = data_dir / "ground_truth"
    sample_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic 120-second signal (50,000 Hz)...")
    timestamps, sensor1, sensor2, gt_peaks = generate_synthetic_signal(
        duration_s=120.0,
        sampling_rate=50000.0,
        num_peaks=30
    )

    df_signal = pd.DataFrame({
        "Sensor 1": sensor1,
        "Sensor 2": sensor2
    })

    signal_path = sample_dir / "industrial_sensor_signal_120s.csv"
    df_signal.to_csv(signal_path, index=False)
    print(f"Saved signal CSV: {signal_path} ({len(df_signal):,} rows)")

    # Ground truth CSV & JSON
    df_gt = pd.DataFrame(gt_peaks)
    gt_csv_path = gt_dir / "ground_truth_peaks_120s.csv"
    gt_json_path = gt_dir / "ground_truth_peaks_120s.json"
    df_gt.to_csv(gt_csv_path, index=False)

    with open(gt_json_path, "w") as f:
        json.dump(gt_peaks, f, indent=2)

    print(f"Saved ground truth files: {gt_csv_path} and {gt_json_path}")

if __name__ == "__main__":
    main()
