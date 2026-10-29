"""Metrics and evaluation helpers for commodity price forecasts."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(actual: Any, predicted: Any) -> dict[str, float]:
	"""Calculate requested metrics while excluding non-finite prediction pairs."""
	actual_array = np.asarray(actual, dtype=float)
	predicted_array = np.asarray(predicted, dtype=float)
	valid = np.isfinite(actual_array) & np.isfinite(predicted_array)
	actual_array = actual_array[valid]
	predicted_array = predicted_array[valid]
	if len(actual_array) == 0:
		return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
	nonzero = actual_array != 0
	mape = (
		float(np.mean(np.abs((actual_array[nonzero] - predicted_array[nonzero]) / actual_array[nonzero])) * 100)
		if nonzero.any()
		else np.nan
	)
	return {
		"mae": float(mean_absolute_error(actual_array, predicted_array)),
		"rmse": float(np.sqrt(mean_squared_error(actual_array, predicted_array))),
		"mape": mape,
		"r2": float(r2_score(actual_array, predicted_array)) if len(actual_array) > 1 else np.nan,
	}
