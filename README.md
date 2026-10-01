# Industrial Signal Analyzer (UEF & Marginum 2026)

A production-quality Python signal processing application for high-throughput automated detection, quantification, classification, and scientific evaluation of industrial sensor signal peaks.

---

## 1. Project Overview & Problem Statement

Industrial signal processing datasets (such as UEF & Marginum recordings) often feature:
- Long durations (from 120 seconds up to ~4 hours, yielding up to **720 million samples** at 50,000 Hz per channel).
- Multi-gigabyte file sizes (up to ~6 GB per CSV).
- Multi-channel sensor recordings (`Sensor 1`, `Sensor 2`).
- Complex signal artifacts including baseline drift, high-frequency Gaussian noise, variable peak amplitudes, and overlapping/fused peaks.

The primary objective of this system is to **reliably detect, quantify, and classify genuine signal peaks while suppressing baseline drift and noise artifacts without loading multi-gigabyte CSV files into system RAM at once.**

---

## 2. System Architecture & Features

```
industrial_signal_analyzer/
│
├── main.py                          # Application entry point
├── pyproject.toml                   # Project metadata and dependencies
├── requirements.txt                 # Requirements list
├── README.md                        # Documentation
│
├── signal_app/
│   ├── app_config.py                # Processing hyperparameters & configuration
│   ├── dashboard.py                 # Core PySide6 Dashboard UI widget
│   ├── main_window.py               # QMainWindow application shell
│   │
│   ├── io/                          # Large CSV streaming & chunk processing
│   │   ├── csv_reader.py            # Memory-efficient chunked CSV reader
│   │   ├── chunk_reader.py          # Pandas chunk iterator with dtype optimization
│   │   └── signal_metadata.py       # Signal metadata dataclass
│   │
│   ├── visualization/               # Interactive high-performance plotting
│   │   ├── downsampling.py          # Min-Max & Envelope peak-preserving downsampling
│   │   ├── signal_plot.py           # PyQtGraph multi-curve plot widget
│   │   ├── overview_plot.py         # Compressed overview with LinearRegion ROI selector
│   │   └── peak_overlay.py          # Peak marker scatter plot formatting
│   │
│   ├── processing/                  # Preprocessing pipeline
│   │   ├── baseline_removal.py      # Rolling median, Moving Average, Polynomial, ALS
│   │   ├── smoothing.py             # Savitzky-Golay, Gaussian, Moving Average
│   │   ├── filtering.py             # Butterworth Lowpass, Highpass, Bandpass
│   │   ├── normalization.py         # Min-Max, Z-Score, Peak Max
│   │   └── preprocessing_pipeline.py # Sequential execution engine
│   │
│   ├── detection/                   # Peak detection & feature extraction
│   │   ├── peak_detector.py         # Primary detector engine
│   │   ├── adaptive_threshold.py    # Rolling std & MAD dynamic noise thresholding
│   │   ├── overlapping_peaks.py     # 2nd derivative inflection point deconvolution
│   │   ├── peak_features.py         # Peak amplitude, prominence, width, area, SNR
│   │   └── peak_quantification.py   # Statistical density summary (peaks/min, peaks/hr)
│   │
│   ├── classification/              # Peak classification
│   │   ├── amplitude_classifier.py  # Deterministic amplitude tiering (Low, Medium, High)
│   │   └── peak_classifier.py       # Multi-feature classifier & confidence scoring
│   │
│   ├── frequency/                   # Spectral analysis
│   │   ├── fft_analysis.py          # Real FFT (rFFT) & Power Spectral Density
│   │   └── frequency_features.py    # Spectral centroid, bandwidth, roll-off
│   │
│   ├── concentration/               # Fluorophore & Tumour size analysis
│   │   ├── calibration.py           # Calibration dataset manager
│   │   ├── regression.py            # Linear & Polynomial regression models (R², MAE, RMSE)
│   │   └── concentration_estimator.py # Concentration category prediction
│   │
│   ├── evaluation/                  # Ground-truth scientific validation
│   │   ├── ground_truth.py          # CSV/JSON annotation loader
│   │   ├── metrics.py               # Precision, Recall, F1 Score calculation
│   │   └── algorithm_comparison.py  # Data-driven benchmark comparing Methods A, B, C, D
│   │
│   ├── workers/                     # Multithreading / QThread execution
│   │   ├── worker.py                # Asynchronous processing background worker
│   │   └── worker_signals.py        # Custom PySide6 worker Qt Signals
│   │
│   ├── export/                      # Exporting results
│   │   ├── results_exporter.py      # CSV peak table & JSON manifest exporters
│   │   └── report_generator.py      # Academic markdown report generator
│   │
│   └── utils/                       # Logging & Validators
│       ├── logging_config.py
│       ├── validators.py
│       └── helpers.py
│
├── data/                            # Sample & ground truth datasets
│   ├── generate_sample_data.py      # Synthetic signal dataset generator
│   ├── sample/
│   └── ground_truth/
│
└── tests/                           # Pytest unit test suite
    ├── test_csv_reader.py
    ├── test_downsampling.py
    ├── test_baseline.py
    ├── test_smoothing.py
    ├── test_peak_detection.py
    ├── test_classification.py
    ├── test_regression.py
    └── test_metrics.py
```

---

## 3. Memory & Large-File Strategy

For a 4-hour recording at 50,000 Hz, loading 720,000,000 floats into memory would consume several gigabytes of RAM.

To handle large files safely:
1. **Chunk Processing**: CSV files are read in configurable chunk streams (e.g. 500,000 rows at a time) using `SignalCSVReader`.
2. **Peak-Preserving Min-Max Downsampling**: For global signal overview plots, the full dataset is split into bins and only the **local minimum and maximum values per bin** are extracted. This guarantees that narrow peak spikes are retained visually without plotting millions of raw points.
3. **Selective Window Processing**: High-resolution processing is performed on selected time windows specified by the user or ROI selector.

---

## 4. Scientific Rationale of Signal Algorithms

- **Baseline Removal (`rolling_median`, `als`)**: Thermal and instrumental drift causes low-frequency baseline fluctuations that can distort fixed amplitude thresholds. Removing baseline drift isolates true peak amplitudes.
- **Noise Smoothing (`savgol`, `gaussian`)**: Savitzky-Golay filtering uses local low-degree polynomial fitting, preserving peak heights and sharp edges far better than standard moving averages.
- **Adaptive Thresholding (`rolling_std`, `mad`)**: Noise levels often vary over long multi-hour industrial recordings. An adaptive threshold ($T(t) = \text{base} + k \cdot \sigma_\text{local}(t)$) dynamically adjusts detection sensitivity.
- **Peak Prominence**: Peak height alone can misidentify noise wiggles on high baselines. Prominence measures how much a peak stands out relative to its local trough background.
- **Precision, Recall & F1-Score**: Evaluates algorithm accuracy against known ground-truth peak locations within a configurable time matching tolerance window ($\pm 20$ ms).

---

## 5. Quick Start & Installation

### Option A: Using `uv` (Recommended)

```bash
# Sync environment
uv sync

# Run synthetic data generator
uv run python data/generate_sample_data.py

# Run main application
uv run python main.py

# Run unit tests
uv run python -m pytest -v
```

### Option B: Using Conda or Standard Pip

```bash
# Create Conda environment
conda create -n signal-analyzer python=3.14 -y
conda activate signal-analyzer

# Install dependencies
pip install -r requirements.txt

# Generate synthetic dataset
python data/generate_sample_data.py

# Run application
python main.py

# Run test suite
python -m pytest -v
```

---

## 6. Running Unit Tests

The project includes an automated unit test suite covering chunk reading, downsampling peak preservation, baseline correction, noise smoothing, peak detection, classification, regression, and ground truth metrics:

```bash
python -m pytest -v
```

Output:
```
tests/test_baseline.py PASSED
tests/test_classification.py PASSED
tests/test_csv_reader.py PASSED
tests/test_downsampling.py PASSED
tests/test_metrics.py PASSED
tests/test_peak_detection.py PASSED
tests/test_regression.py PASSED
tests/test_smoothing.py PASSED
16 passed in 4.61s
```

---

## 7. Acceptance Criteria Checklist

- [x] Repository inspected and clean modular structure established
- [x] Application launches successfully via PySide6 desktop GUI
- [x] Large CSV files processed in chunks without loading multi-GB files into RAM
- [x] Sensor 1 and Sensor 2 selectable
- [x] Min-Max peak-preserving downsampling implemented
- [x] Interactive Overview & Zoom/Pan region selection
- [x] Baseline correction (Rolling Median, Moving Average, Polynomial, ALS)
- [x] Noise smoothing (Savitzky-Golay, Gaussian, Moving Average)
- [x] Butterworth digital filtering (Lowpass, Highpass, Bandpass)
- [x] SciPy `find_peaks` with height, prominence, distance, width
- [x] Dynamic adaptive thresholding (Rolling Std & MAD)
- [x] Overlapping peak 2nd derivative inflection deconvolution
- [x] Peak feature extraction (Amplitude, Prominence, Width, Area, SNR, Rise/Fall time)
- [x] Peak classification (Low, Medium, High tiers & confidence scores)
- [x] Real FFT frequency spectral analysis (Dominant frequencies, Centroid, Bandwidth, Roll-off)
- [x] Ground truth scientific evaluation (Precision, Recall, F1 Score)
- [x] Data-driven algorithm comparison benchmark table (Methods A, B, C, D)
- [x] Fluorophore concentration & tumour size regression model
- [x] CSV peaks table & JSON manifest exporter
- [x] Scientific Markdown report generator
- [x] Automated unit test suite (100% passing)
- [x] Synthetic sample dataset generator included
#   I n d u s t r i a l - S i g n a l - P r o c e s s i n g - A u t o m a t i c - P e a k - D e t e c t i o n - S y s t e m  
 