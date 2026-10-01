"""
Regression Analysis Module for Tumour Size vs Peak Count.
Supports Linear and Polynomial Regression with metrics (R², MAE, RMSE).
"""
import numpy as np
from typing import Dict, Any, Tuple

def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Fits linear regression y = slope * x + intercept."""
    if len(x) < 2 or len(y) < 2:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0, "mae": 0.0, "rmse": 0.0, "predict_fn": lambda v: 0.0}

    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]
    y_pred = slope * x + intercept

    r2 = float(1.0 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))) if np.sum((y - np.mean(y)) ** 2) > 0 else 0.0
    mae = float(np.mean(np.abs(y - y_pred)))
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    return {
        "model_type": "Linear",
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "predict_fn": lambda val: float(slope * val + intercept)
    }

def fit_polynomial_regression(x: np.ndarray, y: np.ndarray, degree: int = 2) -> Dict[str, Any]:
    """Fits polynomial regression of specified degree."""
    if len(x) <= degree or len(y) <= degree:
        return fit_linear_regression(x, y)

    coeffs = np.polyfit(x, y, degree)
    poly_fn = np.poly1d(coeffs)
    y_pred = poly_fn(x)

    r2 = float(1.0 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))) if np.sum((y - np.mean(y)) ** 2) > 0 else 0.0
    mae = float(np.mean(np.abs(y - y_pred)))
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    return {
        "model_type": f"Polynomial (degree {degree})",
        "coefficients": coeffs.tolist(),
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "predict_fn": lambda val: float(poly_fn(val))
    }
