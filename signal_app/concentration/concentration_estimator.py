"""
Fluorophore Concentration Estimator Module.
Estimates fluorophore concentration category and predicted tumour volume from detected peak count.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .calibration import CalibrationData
from .regression import fit_linear_regression, fit_polynomial_regression

class ConcentrationEstimator:
    """Estimates concentration and tumour volume based on calibration models."""

    def __init__(self, calibration_data: Optional[CalibrationData] = None):
        self.cal = calibration_data if calibration_data else CalibrationData()
        self.fitted_model: Dict[str, Any] = {}
        self.fit_model()

    def fit_model(self, use_poly: bool = False, poly_degree: int = 2):
        """Fits regression model on calibration data."""
        x, y = self.cal.get_data()
        if len(x) > 0 and len(y) > 0:
            # We regress Tumour Size (x) from Peak Count (y)
            if use_poly:
                self.fitted_model = fit_polynomial_regression(y, x, degree=poly_degree)
            else:
                self.fitted_model = fit_linear_regression(y, x)

    def estimate_from_peak_count(self, detected_peak_count: int) -> Dict[str, Any]:
        """
        Predicts estimated tumour size and concentration category from detected peak count.
        """
        if not self.fitted_model or "predict_fn" not in self.fitted_model:
            self.fit_model()

        predict_fn = self.fitted_model.get("predict_fn", lambda v: 0.0)
        estimated_tumour_ml = max(0.0, float(predict_fn(detected_peak_count)))

        # Categorize concentration based on peak count range
        if detected_peak_count < 200:
            category = "Low"
        elif detected_peak_count < 400:
            category = "Medium"
        else:
            category = "High"

        return {
            "detected_peak_count": detected_peak_count,
            "estimated_tumour_size_ml": round(estimated_tumour_ml, 2),
            "estimated_concentration": category,
            "regression_r2": round(self.fitted_model.get("r2", 0.0), 4),
            "regression_model": self.fitted_model.get("model_type", "Linear"),
            "is_experimental_validation": False,
            "note": "Estimation derived from calibration curve model. Requires experimental validation."
        }
