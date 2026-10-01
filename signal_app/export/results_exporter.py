"""
Results Exporter Module.
Exports CSV peak tables, JSON analysis manifests, and PNG graphics.
"""
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from ..utils.logging_config import logger

class ResultsExporter:
    """Handles exporting peak detection and analysis results."""

    @staticmethod
    def export_peaks_csv(peaks_df: pd.DataFrame, output_path: Path) -> Path:
        """Exports detected peaks DataFrame to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if peaks_df is not None and not peaks_df.empty:
            peaks_df.to_csv(output_path, index=False)
        else:
            pd.DataFrame().to_csv(output_path, index=False)
        logger.info(f"Exported peaks CSV to {output_path}")
        return output_path

    @staticmethod
    def export_manifest_json(manifest: Dict[str, Any], output_path: Path) -> Path:
        """Exports analysis parameters and statistics to JSON manifest."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        def sanitize(obj):
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return str(obj)

        clean_manifest = sanitize(manifest)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_manifest, f, indent=2)
        logger.info(f"Exported JSON manifest to {output_path}")
        return output_path
