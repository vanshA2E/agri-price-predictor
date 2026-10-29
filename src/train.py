"""Train and compare time-aware commodity forecasting models."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import calculate_metrics

LOGGER = logging.getLogger(__name__)

try:
	from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - depends on optional environment installation
	XGBRegressor = None

try:
	from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - depends on optional environment installation
	LGBMRegressor = None


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
	with Path(path).open(encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def make_supervised(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
	"""Create a future target while retaining only present/past feature columns."""
	result = frame.sort_values(["commodity", "market", "date"]).copy()
	result["target"] = result.groupby(["commodity", "market"], dropna=False)["modal_price"].shift(-horizon)
	return result.dropna(subset=["target"])


def chronological_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	ordered = frame.sort_values("date").reset_index(drop=True)
	train_end = int(len(ordered) * 0.70)
	validation_end = int(len(ordered) * 0.85)
	return ordered.iloc[:train_end], ordered.iloc[train_end:validation_end], ordered.iloc[validation_end:]


def model_factories(seed: int) -> dict[str, Any]:
	factories: dict[str, Any] = {
		"ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
		"random_forest": lambda: make_pipeline(
			SimpleImputer(strategy="median"),
			RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=seed, n_jobs=-1),
		),
	}
	if XGBRegressor is not None:
		factories["xgboost"] = lambda: make_pipeline(
			SimpleImputer(strategy="median"),
			XGBRegressor(
				n_estimators=300,
				max_depth=6,
				learning_rate=0.05,
				subsample=0.8,
				colsample_bytree=0.8,
				objective="reg:squarederror",
				random_state=seed,
				n_jobs=2,
			),
		)
	if LGBMRegressor is not None:
		factories["lightgbm"] = lambda: make_pipeline(
			SimpleImputer(strategy="median"),
			LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=seed, verbosity=-1),
		)
	return factories


def feature_columns(frame: pd.DataFrame) -> list[str]:
	excluded = {"date", "modal_price", "target", "state", "market", "commodity", "variety", "unit"}
	return [column for column in frame.select_dtypes(include=[np.number]).columns if column not in excluded]


def train_models(config_path: str | Path = "config.yaml") -> pd.DataFrame:
	config = load_config(config_path)
	input_path = Path(config["paths"]["processed_data"]) / "mandi_features.csv"
	model_dir = Path(config["paths"]["models"])
	results_path = Path(config["paths"]["processed_data"]) / "model_comparison.csv"
	model_dir.mkdir(parents=True, exist_ok=True)
	if not input_path.exists() or input_path.stat().st_size == 0:
		LOGGER.warning("No feature data found at %s; no model scores can be computed.", input_path)
		empty = pd.DataFrame(columns=["commodity", "horizon", "model", "mae", "rmse", "mape", "r2"])
		empty.to_csv(results_path, index=False)
		return empty
	try:
		frame = pd.read_csv(input_path)
	except pd.errors.EmptyDataError:
		LOGGER.warning("Feature data at %s is empty; no model scores can be computed.", input_path)
		empty = pd.DataFrame(columns=["commodity", "horizon", "model", "mae", "rmse", "mape", "r2"])
		empty.to_csv(results_path, index=False)
		return empty
	frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
	factories = model_factories(int(config.get("project", {}).get("random_seed", 42)))
	rows: list[dict[str, Any]] = []
	for commodity in sorted(frame["commodity"].dropna().unique()):
		commodity_frame = frame[frame["commodity"] == commodity].copy()
		for horizon in (7, 30):
			supervised = make_supervised(commodity_frame, horizon)
			if len(supervised) < 20:
				LOGGER.warning("Skipping %s horizon=%s: only %s usable rows.", commodity, horizon, len(supervised))
				continue
			train, _, test = chronological_rows(supervised)
			columns = feature_columns(supervised)
			if not columns or train.empty or test.empty:
				continue
			x_train, y_train = train[columns], train["target"]
			x_test, y_test = test[columns], test["target"]
			baseline_predictions = {
				"naive": test.get("price_lag_1", pd.Series(np.nan, index=test.index)),
				"seasonal_naive": test.get("price_lag_7", pd.Series(np.nan, index=test.index)),
			}
			for name, predictions in baseline_predictions.items():
				metrics = calculate_metrics(y_test, predictions)
				rows.append({"commodity": commodity, "horizon": horizon, "model": name, **metrics})
			for name, factory in factories.items():
				model = factory()
				model.fit(x_train, y_train)
				predictions = model.predict(x_test)
				metrics = calculate_metrics(y_test, predictions)
				rows.append({"commodity": commodity, "horizon": horizon, "model": name, **metrics})
				artifact = model_dir / f"{commodity.lower().replace('/', '_')}_{horizon}d_{name}.joblib"
				joblib.dump(
					{
						"model": model,
						"features": columns,
						"commodity": commodity,
						"horizon": horizon,
						"metrics": metrics,
						"trained_at": datetime.now(timezone.utc).isoformat(),
					},
					artifact,
				)
	results = pd.DataFrame(rows)
	results.to_csv(results_path, index=False)
	if not results.empty:
		best = results.sort_values(["commodity", "horizon", "mae"]).groupby(
			["commodity", "horizon"], as_index=False
		).first()
		best.to_csv(Path(config["paths"]["models"]) / "best_models.csv", index=False)
	LOGGER.info("Model comparison completed with %s scored rows.", len(results))
	return results


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="config.yaml")
	train_models(parser.parse_args().config)
