"""Create chronological train, validation, and test datasets."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
	with Path(path).open(encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def chronological_split(
	frame: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Split by sorted timestamp; the latest observations remain untouched for testing."""
	if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
		raise ValueError("Split fractions must be between zero and one.")
	if train_fraction + validation_fraction >= 1:
		raise ValueError("Training plus validation fractions must be below one.")
	ordered = frame.copy()
	ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
	ordered = ordered.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
	train_end = int(len(ordered) * train_fraction)
	validation_end = int(len(ordered) * (train_fraction + validation_fraction))
	return ordered.iloc[:train_end], ordered.iloc[train_end:validation_end], ordered.iloc[validation_end:]


def run_split(config_path: str | Path = "config.yaml") -> dict[str, int]:
	config = load_config(config_path)
	input_path = Path(config["paths"]["processed_data"]) / "mandi_features.csv"
	output_dir = Path(config["paths"]["processed_data"])
	if not input_path.exists() or input_path.stat().st_size == 0:
		LOGGER.warning("No feature data found at %s; writing empty split files.", input_path)
		sizes = {}
		for name in ("train", "val", "test"):
			pd.DataFrame().to_csv(output_dir / f"{name}.csv", index=False)
			sizes[name] = 0
		return sizes
	try:
		frame = pd.read_csv(input_path)
	except pd.errors.EmptyDataError:
		LOGGER.warning("Feature data at %s is empty; writing empty split files.", input_path)
		sizes = {}
		for name in ("train", "val", "test"):
			pd.DataFrame().to_csv(output_dir / f"{name}.csv", index=False)
			sizes[name] = 0
		return sizes
	split_config = config.get("split", {})
	train, validation, test = chronological_split(
		frame,
		float(split_config.get("train_fraction", 0.70)),
		float(split_config.get("validation_fraction", 0.15)),
	)
	for name, subset in (("train", train), ("val", validation), ("test", test)):
		subset.to_csv(output_dir / f"{name}.csv", index=False)
	sizes = {"train": len(train), "val": len(validation), "test": len(test)}
	LOGGER.info("Chronological split summary: %s", sizes)
	return sizes


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="config.yaml")
	run_split(parser.parse_args().config)
