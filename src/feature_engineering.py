"""Create calendar, lag, rolling, and categorical features without leakage."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
	with Path(path).open(encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def add_calendar_features(frame: pd.DataFrame, festival_months: list[int]) -> pd.DataFrame:
	result = frame.copy()
	result["date"] = pd.to_datetime(result["date"], errors="coerce")
	result["year"] = result["date"].dt.year
	result["month"] = result["date"].dt.month
	result["week_of_year"] = result["date"].dt.isocalendar().week.astype("Int64")
	result["day_of_week"] = result["date"].dt.dayofweek
	result["quarter"] = result["date"].dt.quarter
	result["is_festival_season"] = result["month"].isin(festival_months).astype(int)
	return result


def add_lag_and_rolling_features(
	frame: pd.DataFrame, lag_days: list[int], rolling_windows: list[int]
) -> pd.DataFrame:
	"""Build all price-derived values from observations strictly before each row."""
	result = frame.sort_values(["market", "commodity", "date"]).copy()
	groups = result.groupby(["market", "commodity"], dropna=False)["modal_price"]
	for lag in lag_days:
		result[f"price_lag_{lag}"] = groups.shift(lag)
	prior_price = groups.shift(1)
	result["price_change_pct"] = prior_price.groupby(
		[result["market"], result["commodity"]], dropna=False
	).pct_change(fill_method=None) * 100
	for window in rolling_windows:
		result[f"price_rolling_mean_{window}"] = groups.transform(
			lambda values: values.shift(1).rolling(window, min_periods=2).mean()
		)
		result[f"price_rolling_std_{window}"] = groups.transform(
			lambda values: values.shift(1).rolling(window, min_periods=2).std()
		)
		result[f"price_rolling_min_{window}"] = groups.transform(
			lambda values: values.shift(1).rolling(window, min_periods=2).min()
		)
		result[f"price_rolling_max_{window}"] = groups.transform(
			lambda values: values.shift(1).rolling(window, min_periods=2).max()
		)
	if "arrival_quantity" in result:
		arrival_groups = result.groupby(["market", "commodity"], dropna=False)["arrival_quantity"]
		result["arrival_quantity_lag_1"] = arrival_groups.shift(1)
	return result


def encode_categories(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
	"""Apply deterministic label encoding; mappings are saved with model metadata later."""
	result = frame.copy()
	mappings: dict[str, dict[str, int]] = {}
	for column in columns:
		if column not in result:
			continue
		values = result[column].fillna("Unknown").astype(str)
		labels = {value: index for index, value in enumerate(sorted(values.unique()))}
		result[f"{column}_encoded"] = values.map(labels).astype(int)
		mappings[column] = labels
	return result, mappings


def engineer_features(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
	"""Return a feature table where no feature reads the current target price."""
	if frame.empty:
		return frame.copy(), {}
	feature_config = config.get("features", {})
	result = add_calendar_features(frame, feature_config.get("festival_months", [10, 11]))
	result = add_lag_and_rolling_features(
		result,
		[int(value) for value in feature_config.get("lag_days", [1, 7, 14, 30])],
		[int(value) for value in feature_config.get("rolling_windows", [7, 14, 30])],
	)
	result, mappings = encode_categories(result, ["commodity", "market", "state"])
	result = result.replace([np.inf, -np.inf], np.nan)
	return result, mappings


def run_feature_engineering(config_path: str | Path = "config.yaml") -> dict[str, Any]:
	config = load_config(config_path)
	input_path = Path(config["paths"]["interim_data"]) / "mandi_daily_cleaned.csv"
	output_path = Path(config["paths"]["processed_data"]) / "mandi_features.csv"
	if not input_path.exists() or input_path.stat().st_size == 0:
		LOGGER.warning("No cleaned daily data found at %s; skipping feature generation.", input_path)
		pd.DataFrame().to_csv(output_path, index=False)
		return {"rows": 0, "features": 0}
	try:
		frame = pd.read_csv(input_path)
	except pd.errors.EmptyDataError:
		LOGGER.warning("Cleaned daily data at %s is empty; skipping feature generation.", input_path)
		pd.DataFrame().to_csv(output_path, index=False)
		return {"rows": 0, "features": 0}
	features, mappings = engineer_features(frame, config)
	features.to_csv(output_path, index=False)
	mapping_path = output_path.with_name("feature_category_mappings.yaml")
	mapping_path.write_text(yaml.safe_dump(mappings, sort_keys=True), encoding="utf-8")
	summary = {"rows": len(features), "features": len(features.columns), "category_mappings": len(mappings)}
	LOGGER.info("Feature engineering summary: %s", summary)
	return summary


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="config.yaml")
	run_feature_engineering(parser.parse_args().config)
