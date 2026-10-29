"""Stored-artifact prediction helpers for the API and Streamlit dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml


def load_model(path: str | Path) -> dict[str, Any]:
	"""Load a joblib artifact produced by ``src.train``."""
	artifact = joblib.load(path)
	if not isinstance(artifact, dict) or "model" not in artifact or "features" not in artifact:
		raise ValueError(f"Invalid model artifact: {path}")
	return artifact


def predict_from_features(artifact: dict[str, Any], features: pd.DataFrame) -> pd.Series:
	"""Predict from a feature frame, filling absent columns as missing for the pipeline imputer."""
	model_features = artifact["features"]
	prepared = features.reindex(columns=model_features)
	return pd.Series(artifact["model"].predict(prepared), index=features.index, name="predicted_price")


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, Any]:
	with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def _safe_model_name(commodity: str) -> str:
	return commodity.lower().replace("/", "_").replace(" ", "_")


def _artifact_path(commodity: str, model_name: str) -> Path:
	models = ROOT / "models"
	if model_name == "global_lightgbm":
		return models / "ceda_weekly_global_lightgbm_4w.joblib"
	if model_name == "commodity_lightgbm":
		return models / f"ceda_weekly_{_safe_model_name(commodity)}_lightgbm_4w.joblib"
	return models / "ceda_weekly_global_lightgbm_4w.joblib"


def _feature_row(history: pd.DataFrame, commodity: str, state: str, as_of: pd.Timestamp) -> pd.DataFrame:
	series = history[(history["commodity"] == commodity) & (history["state"] == state) & (history["week"] <= as_of)].sort_values("week")
	if series.empty:
		raise ValueError(f"No history found for {commodity}/{state} at {as_of.date()}")
	row = series.iloc[-1]
	values: dict[str, Any] = {
		"commodity": commodity, "state": state, "week_of_year": int(as_of.isocalendar().week),
		"month": int(as_of.month), "is_festival_season": int(as_of.month in _config().get("features", {}).get("festival_months", [10, 11])),
	}
	prices = series["modal_price"].astype(float).tolist()
	for lag in (1, 2, 4, 8, 52):
		values[f"modal_price_lag_{lag}w"] = prices[-lag] if len(prices) >= lag else np.nan
		values[f"is_filled_lag_{lag}w"] = bool(series["is_filled"].iloc[-lag]) if len(series) >= lag else False
		values[f"is_outlier_lag_{lag}w"] = bool(series["is_outlier"].iloc[-lag]) if len(series) >= lag else False
	for window in (4, 8, 13):
		past = np.asarray(prices[-window:], dtype=float)
		values[f"modal_price_rolling_mean_{window}w"] = float(past.mean()) if len(past) == window else np.nan
		values[f"modal_price_rolling_std_{window}w"] = float(past.std(ddof=1)) if len(past) == window else np.nan
	return pd.DataFrame([values]), float(row["modal_price"]), pd.Timestamp(row["week"])


def _validation_residuals(history: pd.DataFrame, commodity: str, state: str, model_name: str) -> np.ndarray:
	validation_path = ROOT / "data" / "processed" / "weekly_val.csv"
	if not validation_path.exists():
		return np.array([])
	frame = pd.read_csv(validation_path, parse_dates=["week"])
	frame = frame[(frame["commodity"] == commodity) & (frame["state"] == state) & (~frame["target_is_filled"])]
	if frame.empty:
		return np.array([])
	if model_name in {"naive", "seasonal_naive"}:
		predicted = frame["modal_price"] if model_name == "naive" else frame["modal_price_lag_52w"]
	else:
		artifact = load_model(_artifact_path(commodity, model_name))
		predicted = frame["modal_price"] * np.exp(artifact["model"].predict(frame.reindex(columns=artifact["features"])))
	return (frame["target_price"].to_numpy(dtype=float) - np.asarray(predicted, dtype=float))[np.isfinite(predicted)]


def forecast(
	commodity: str,
	state: str,
	horizon_weeks: int = 4,
	as_of_week: str | None = None,
) -> dict[str, Any]:
	"""Forecast from stored weekly data and a saved model registry entry."""
	if horizon_weeks != int(_config().get("model", {}).get("horizon_weeks", 4)):
		raise ValueError("Only the configured four-week horizon is supported by the saved artifacts")
	weekly_path = ROOT / "data" / "processed" / "ceda_weekly_clean.csv"
	history = pd.read_csv(weekly_path, parse_dates=["week"])
	if commodity not in set(history["commodity"]) or state not in set(history["state"]):
		raise KeyError(f"Unknown commodity/state: {commodity}/{state}")
	series = history[(history["commodity"] == commodity) & (history["state"] == state)].sort_values("week")
	as_of = pd.Timestamp(as_of_week) if as_of_week else pd.Timestamp(series["week"].max())
	row, last_price, observed_week = _feature_row(history, commodity, state, as_of)
	registry = json.loads((ROOT / "data" / "processed" / "model_registry.json").read_text(encoding="utf-8"))
	model_name = registry.get(commodity, "naive")
	if model_name == "naive":
		predicted = last_price
	elif model_name == "seasonal_naive":
		past = series[series["week"] <= observed_week - pd.Timedelta(weeks=52)]
		predicted = float(past.iloc[-1]["modal_price"]) if not past.empty else last_price
	else:
		artifact = load_model(_artifact_path(commodity, model_name))
		predicted = float(last_price * np.exp(artifact["model"].predict(row.reindex(columns=artifact["features"]))[0]))
	residuals = _validation_residuals(history, commodity, state, model_name)
	if len(residuals) >= 5:
		low_error, high_error = np.quantile(residuals, [0.05, 0.95])
	else:
		low_error, high_error = -0.10 * last_price, 0.10 * last_price
	return {
		"commodity": commodity, "state": state, "model": model_name, "horizon_weeks": horizon_weeks,
		"as_of_week": observed_week.date().isoformat(), "last_observed_price": last_price,
		"target_week": (observed_week + pd.to_timedelta(horizon_weeks, unit="W")).date().isoformat(),
		"forecast_price": predicted, "lower_90": max(0.0, predicted + low_error), "upper_90": max(0.0, predicted + high_error),
	}
