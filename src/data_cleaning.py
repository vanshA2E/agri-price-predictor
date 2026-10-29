"""Clean raw mandi records into leakage-safe daily and weekly price tables."""
from __future__ import annotations

import argparse
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)

ALIASES = {
	"state": "state",
	"statename": "state",
	"market": "market",
	"marketname": "market",
	"mandi": "market",
	"market_id": "market_id",
	"marketid": "market_id",
	"state_id": "state_id",
	"stateid": "state_id",
	"district_id": "district_id",
	"districtid": "district_id",
	"commodity_id": "commodity_id",
	"commodityid": "commodity_id",
	"district": "district",
	"commodity": "commodity",
	"commodityname": "commodity",
	"variety": "variety",
	"arrivaldate": "date",
	"arrival_date": "date",
	"date": "date",
	"pricedate": "date",
	"modalprice": "modal_price",
	"modal_price": "modal_price",
	"minprice": "min_price",
	"min_price": "min_price",
	"maxprice": "max_price",
	"max_price": "max_price",
	"arrivalquantity": "arrival_quantity",
	"arrival_qty": "arrival_quantity",
	"arrivals": "arrival_quantity",
	"unit": "unit",
}

COMMODITY_NAMES = {
	"onion": "Onion",
	"potato": "Potato",
	"tomato": "Tomato",
	"tur/arhar": "Tur/Arhar",
	"tur arhar": "Tur/Arhar",
	"arhar": "Tur/Arhar",
	"moong": "Moong",
	"green gram": "Moong",
	"urad": "Urad",
	"black gram": "Urad",
	"masoor": "Masoor",
	"lentil": "Masoor",
	"gram/chana": "Gram/Chana",
	"gram chana": "Gram/Chana",
	"chana": "Gram/Chana",
	"gram": "Gram/Chana",
}


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
	with Path(path).open(encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def configure_logging(config: dict[str, Any]) -> None:
	logging.basicConfig(
		level=getattr(logging, config.get("logging", {}).get("level", "INFO").upper()),
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		handlers=[logging.StreamHandler()],
		force=True,
	)


def canonical_column(value: str) -> str:
	"""Normalize headers while retaining only meaningful separators."""
	key = re.sub(r"[^a-z0-9_]", "", str(value).strip().lower().replace(" ", "_"))
	return ALIASES.get(key, key)


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
	"""Map common Agmarknet/data.gov.in headers to a stable internal schema."""
	result = frame.copy()
	result.columns = [canonical_column(column) for column in result.columns]
	result = result.loc[:, ~result.columns.duplicated()]
	for required in (
		"state",
		"market",
		"commodity",
		"date",
		"modal_price",
		"district",
		"variety",
		"unit",
	):
		if required not in result:
			result[required] = pd.NA
	return result


def normalize_text(value: Any) -> str | None:
	if pd.isna(value):
		return None
	text = re.sub(r"\s+", " ", str(value).strip())
	return text.title() if text else None


def normalize_commodity(value: Any) -> str | None:
	text = normalize_text(value)
	return COMMODITY_NAMES.get(text.lower(), text) if text else None


def parse_dates(values: pd.Series) -> pd.Series:
	"""Parse ISO dates first, then recover common day-first mandi formats."""
	parsed = pd.to_datetime(values, errors="coerce")
	missing = parsed.isna()
	if missing.any():
		parsed.loc[missing] = pd.to_datetime(values.loc[missing], errors="coerce", dayfirst=True)
	return parsed


def convert_to_rs_per_quintal(values: pd.Series, units: pd.Series) -> pd.Series:
	"""Convert supported price units; unknown units remain unchanged and are logged."""
	numeric = pd.to_numeric(values.astype(str).str.replace(",", "", regex=False), errors="coerce")
	unit_text = units.fillna("rs/quintal").astype(str).str.lower().str.replace(" ", "", regex=False)
	factors = pd.Series(1.0, index=values.index)
	factors[unit_text.str.contains(r"/kg|perkg|kilogram", regex=True)] = 100.0
	factors[unit_text.str.contains(r"/ton|/tonne|perton|pertonne", regex=True)] = 10.0
	known = unit_text.str.contains(r"quintal|qtl|rs/q", regex=True) | (unit_text == "rs/quintal")
	unknown = ~(known | unit_text.str.contains(r"/kg|perkg|kilogram|/ton|/tonne|perton|pertonne", regex=True))
	if unknown.any():
		LOGGER.warning("Leaving %s prices unchanged because their unit is unknown.", int(unknown.sum()))
	return numeric * factors


def _series_key(frame: pd.DataFrame) -> pd.Series:
	return frame["market"].fillna("Unknown") + "|" + frame["commodity"].fillna("Unknown")


def _add_outlier_flags(frame: pd.DataFrame, window: int) -> pd.DataFrame:
	result = frame.copy()
	min_periods = min(7, window)
	grouped = result.groupby(["market", "commodity"], dropna=False)["modal_price"]
	rolling_median = grouped.transform(lambda values: values.rolling(window, min_periods=min_periods).median())
	rolling_std = grouped.transform(lambda values: values.rolling(window, min_periods=min_periods).std())
	rolling_z = (result["modal_price"] - rolling_median).abs() / rolling_std.replace(0, np.nan)
	iqr = result.groupby(["market", "commodity"], dropna=False)["modal_price"].transform(
		lambda values: values.quantile(0.75) - values.quantile(0.25)
	)
	q1 = result.groupby(["market", "commodity"], dropna=False)["modal_price"].transform("quantile", 0.25)
	q3 = result.groupby(["market", "commodity"], dropna=False)["modal_price"].transform("quantile", 0.75)
	result["price_outlier"] = (rolling_z > 4) | (result["modal_price"] < q1 - 1.5 * iqr) | (
		result["modal_price"] > q3 + 1.5 * iqr
	)
	return result


def detect_stuck_prices(frame: pd.DataFrame) -> pd.Series:
	"""Flag modal prices stuck for more than five consecutive observations."""
	if frame.empty:
		return pd.Series(False, index=frame.index, dtype=bool)
	result = pd.Series(False, index=frame.index, dtype=bool)
	keys = ["commodity", "state", "market_id"]
	for _, group in frame.sort_values("date").groupby(keys, dropna=False):
		values = group["modal_price"].astype(float)
		run = values.eq(values.shift()).groupby((values.ne(values.shift())).cumsum()).transform("sum")
		triplet = group["min_price"].eq(group["max_price"]) & group["max_price"].eq(group["modal_price"])
		result.loc[group.index] = (run > 5) | (triplet & (run > 5))
	return result


def add_log_iqr_outliers(frame: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
	"""Flag, rather than remove, extreme values using a per-series log-IQR fence."""
	result = frame.copy()
	result["price_outlier"] = False
	for _, indexes in result.groupby(["commodity", "state", "market_id"], dropna=False).groups.items():
		values = result.loc[indexes, "modal_price"].astype(float)
		logged = np.log(values[values > 0])
		if len(logged) < 4:
			continue
		spread = logged.quantile(0.75) - logged.quantile(0.25)
		lower = logged.quantile(0.25) - factor * spread
		upper = logged.quantile(0.75) + factor * spread
		result.loc[indexes, "price_outlier"] = np.log(values).lt(lower) | np.log(values).gt(upper)
	return result


def _week_end(values: pd.Series) -> pd.Series:
	return values.dt.to_period("W-SUN").dt.end_time.dt.normalize()


def build_ceda_weekly(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
	"""Clean CEDA rows and build fixed-market weekly series and coverage files."""
	processed = Path(config["paths"]["processed_data"])
	processed.mkdir(parents=True, exist_ok=True)
	if frame.empty:
		return pd.DataFrame(), {"rows_before": 0, "rows_after": 0, "empty_input": True}
	result = standardize_columns(frame)
	for column in ("commodity", "state"):
		result[column] = result[column].map(normalize_text)
	result["date"] = parse_dates(result["date"])
	for column in ("market_id", "min_price", "max_price", "modal_price"):
		result[column] = pd.to_numeric(result[column], errors="coerce")
	result = result.dropna(subset=["date", "commodity", "state", "market_id", "modal_price"])
	result = result[result["modal_price"] > 0]
	result = result.sort_values(["commodity", "state", "market_id", "date"])
	rows_after_positive = len(result)
	group_keys = ["commodity", "state", "market_id", "date"]
	for column in ("min_price", "max_price", "modal_price"):
		result[column] = pd.to_numeric(result[column], errors="coerce")
	result = result.groupby(group_keys, as_index=False).agg(
		min_price=("min_price", "median"), max_price=("max_price", "median"), modal_price=("modal_price", "median")
	)
	result["stale_price"] = detect_stuck_prices(result)
	result = add_log_iqr_outliers(result)
	interim = Path(config["paths"]["interim_data"])
	interim.mkdir(parents=True, exist_ok=True)
	result.to_csv(interim / "ceda_daily_cleaned.csv", index=False)
	usable = result[~result["stale_price"]].copy()
	usable["week"] = _week_end(usable["date"])

	coverage_rows: list[dict[str, Any]] = []
	for keys, group in usable.groupby(["commodity", "state", "market_id"], dropna=False):
		weeks = sorted(group["week"].dropna().unique())
		gaps = [max(0, int((right - left).days / 7) - 1) for left, right in zip(weeks, weeks[1:])]
		coverage_rows.append({
			"commodity": keys[0], "state": keys[1], "market_id": keys[2], "weeks_with_price": len(weeks),
			"first_date": group["date"].min().date().isoformat(), "last_date": group["date"].max().date().isoformat(),
			"longest_gap_weeks": max(gaps, default=0),
		})
	coverage = pd.DataFrame(coverage_rows)
	coverage.to_csv(processed / "market_coverage.csv", index=False)
	if coverage.empty:
		selected = coverage.copy()
	else:
		selected = coverage.sort_values(["commodity", "state", "weeks_with_price", "last_date"], ascending=[True, True, False, False])
		selected = selected.groupby(["commodity", "state"], group_keys=False).head(3).reset_index(drop=True)
	selected.to_csv(processed / "selected_markets.csv", index=False)

	min_history = int(config.get("cleaning", {}).get("min_history_weeks", 52))
	dropped_series = 0
	weekly_parts: list[pd.DataFrame] = []
	for row in selected.itertuples(index=False):
		group = usable[(usable["commodity"] == row.commodity) & (usable["state"] == row.state) & (usable["market_id"] == row.market_id)]
		weeks = pd.date_range(group["week"].min(), group["week"].max(), freq="W-SUN")
		weekly = group.groupby("week", as_index=True)[["min_price", "max_price", "modal_price"]].median().reindex(weeks)
		weekly = weekly.ffill(limit=2).dropna(subset=["modal_price"])
		if len(weekly) < min_history:
			dropped_series += 1
			continue
		weekly = weekly.reset_index(names="week")
		weekly["commodity"], weekly["state"], weekly["market_id"], weekly["series_type"] = row.commodity, row.state, row.market_id, "market"
		weekly_parts.append(weekly)

	if weekly_parts:
		market_weekly = pd.concat(weekly_parts, ignore_index=True)
		state_weekly = market_weekly.groupby(["commodity", "state", "week"], as_index=False)[["min_price", "max_price", "modal_price"]].median()
		state_weekly["market_id"], state_weekly["series_type"] = "STATE_MEDIAN", "state_median"
		weekly_output = pd.concat([market_weekly, state_weekly], ignore_index=True)
	else:
		weekly_output = pd.DataFrame(columns=["week", "commodity", "state", "market_id", "series_type", "min_price", "max_price", "modal_price"])
	weekly_output.to_csv(processed / "mandi_weekly.csv", index=False)
	report = {
		"rows_before": len(frame), "rows_after_positive": rows_after_positive, "rows_after_deduplicated": len(result),
		"zero_or_negative_removed": len(frame) - rows_after_positive, "stale_rows_flagged": int(result["stale_price"].sum()),
		"stale_rows_removed_for_modeling": int(result["stale_price"].sum()), "outliers_flagged": int(result["price_outlier"].sum()),
		"selected_markets": len(selected), "series_dropped_for_too_little_history": dropped_series,
		"weekly_rows": len(weekly_output), "missing_weeks_after_fill": int(weekly_output["modal_price"].isna().sum()) if not weekly_output.empty else 0,
	}
	return weekly_output, report


def _ceda_safe_name(value: str) -> str:
	return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def ceda_file_metadata(path: Path, config: dict[str, Any]) -> tuple[str, str]:
	"""Read commodity and state only from the documented CEDA filename format."""
	parts = path.stem.split("__")
	if len(parts) != 4:
		raise ValueError(f"Unexpected CEDA filename format: {path.name}")
	commodity_token, state_token, start_token, end_token = parts
	try:
		date.fromisoformat(start_token)
		date.fromisoformat(end_token)
	except ValueError as exc:
		raise ValueError(f"Unexpected CEDA date window in filename: {path.name}") from exc
	commodity = next(
		(name for name in config["ceda"]["commodities"] if _ceda_safe_name(name) == commodity_token),
		normalize_text(commodity_token) or commodity_token,
	)
	state = next(
		(name for name in config["ceda"]["states"] if _ceda_safe_name(name) == state_token),
		normalize_text(state_token) or state_token,
	)
	return commodity, state


def ceda_validation_report(frame: pd.DataFrame) -> pd.DataFrame:
	"""Return one coverage/quality row per commodity and state."""
	rows: list[dict[str, Any]] = []
	for (commodity, state), group in frame.groupby(["commodity", "state"], dropna=False):
		dates = sorted(group["date"].dt.normalize().dropna().unique())
		first = dates[0].date().isoformat() if dates else None
		last = dates[-1].date().isoformat() if dates else None
		gaps = [max(0, int((right - left).days) - 1) for left, right in zip(dates, dates[1:])]
		rows.append({
			"commodity": commodity,
			"state": state,
			"first_date": first,
			"last_date": last,
			"row_count": len(group),
			"missing_days": max(0, (dates[-1] - dates[0]).days + 1 - len(dates)) if dates else 0,
			"longest_gap_days": max(gaps, default=0),
			"min_modal_price": group["modal_price"].min(),
			"max_modal_price": group["modal_price"].max(),
		})
	return pd.DataFrame(rows, columns=[
		"commodity", "state", "first_date", "last_date", "row_count", "missing_days",
		"longest_gap_days", "min_modal_price", "max_modal_price",
	])


def _fill_short_log_gaps(weekly: pd.DataFrame) -> tuple[pd.DataFrame, int]:
	"""Fill bounded missing runs of at most six weeks using log-linear interpolation."""
	result = weekly.copy()
	result["is_filled"] = False
	missing = result["modal_price"].isna().to_numpy()
	position = 0
	while position < len(result):
		if not missing[position]:
			position += 1
			continue
		start = position
		while position < len(result) and missing[position]:
			position += 1
		end = position - 1
		length = end - start + 1
		if length <= 6 and start > 0 and end < len(result) - 1:
			left, right = start - 1, end + 1
			for column in ("modal_price", "min_price", "max_price"):
				if pd.isna(result.iloc[left][column]) or pd.isna(result.iloc[right][column]):
					continue
				log_values = np.log(result[column].astype(float))
				interpolated = np.interp(
					np.arange(start, end + 1), [left, right], [log_values.iloc[left], log_values.iloc[right]]
				)
				result.iloc[start:end + 1, result.columns.get_loc(column)] = np.exp(interpolated)
			result.loc[result.index[start:end + 1], "is_filled"] = True
	return result, int(result["is_filled"].sum())


def _ceda_series_report(
	weekly: pd.DataFrame,
	daily: pd.DataFrame,
	max_missing: float,
	min_history: int,
	trimmed_weeks: dict[tuple[str, str], int],
) -> pd.DataFrame:
	rows: list[dict[str, Any]] = []
	for (commodity, state), group in weekly.groupby(["commodity", "state"], dropna=False):
		first_week, last_week = group["week"].min(), group["week"].max()
		full_weeks = pd.date_range(first_week, last_week, freq="W-SUN")
		missing_fraction = float(group["modal_price"].isna().sum() / len(group)) if len(group) else 1.0
		daily_group = daily[(daily["commodity"] == commodity) & (daily["state"] == state)]
		weeks = len(full_weeks)
		reason = []
		if missing_fraction > max_missing:
			reason.append(f"weekly missing fraction {missing_fraction:.3f} > {max_missing:.2f}")
		if weeks < min_history:
			reason.append(f"only {weeks} weeks < {min_history}")
		rows.append({
			"commodity": commodity, "state": state, "weeks": weeks,
			"weekly_missing_fraction": missing_fraction, "filled_weeks": int(group["is_filled"].sum()),
			"outlier_count": int(daily_group["is_outlier"].sum()),
			"trimmed_weeks": trimmed_weeks.get((commodity, state), 0),
			"first_week": first_week.date().isoformat(), "last_week": last_week.date().isoformat(),
			"status": "DROPPED" if reason else "PASSED", "reason": "; ".join(reason),
		})
	return pd.DataFrame(rows)


def build_ceda_state_weekly(paths: list[Path], config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
	"""Load direct CEDA CSVs and produce cleaned state-level weekly series."""
	processed = Path(config["paths"]["processed_data"])
	interim = Path(config["paths"]["interim_data"])
	processed.mkdir(parents=True, exist_ok=True)
	interim.mkdir(parents=True, exist_ok=True)
	exclusions = {
		(item["commodity"], item["state"]): item.get("reason", "excluded by configuration")
		for item in config.get("cleaning", {}).get("exclude_series", [])
	}
	frames: list[pd.DataFrame] = []
	rows_before = 0
	excluded_rows: list[dict[str, Any]] = []
	excluded_seen: set[tuple[str, str]] = set()
	trimmed_weeks: dict[tuple[str, str], int] = {}
	for path in paths:
		commodity, state = ceda_file_metadata(path, config)
		series_key = (_ceda_safe_name(commodity), _ceda_safe_name(state))
		if series_key in exclusions:
			reason = exclusions[series_key]
			if series_key not in excluded_seen:
				LOGGER.warning("Excluding %s / %s: %s", commodity, state, reason)
				excluded_rows.append({
					"commodity": commodity, "state": state, "weeks": 0, "weekly_missing_fraction": 1.0,
					"filled_weeks": 0, "outlier_count": 0, "trimmed_weeks": 0, "first_week": None, "last_week": None,
					"status": "DROPPED", "reason": reason,
				})
				excluded_seen.add(series_key)
			continue
		try:
			raw = pd.read_csv(path)
		except pd.errors.EmptyDataError:
			continue
		rows_before += len(raw)
		missing = {"date", "min_price", "max_price", "modal_price"}.difference(raw.columns)
		if missing:
			raise ValueError(f"CEDA file {path.name} is missing columns: {sorted(missing)}")
		frame = raw[["date", "min_price", "max_price", "modal_price"]].copy()
		frame["commodity"], frame["state"] = commodity, state
		frames.append(frame)

	columns = ["date", "commodity", "state", "min_price", "max_price", "modal_price", "is_outlier"]
	if not frames:
		empty = pd.DataFrame(columns=columns)
		empty.to_csv(interim / "ceda_daily_cleaned.csv", index=False)
		report = pd.DataFrame(columns=["commodity", "state", "weeks", "weekly_missing_fraction", "filled_weeks", "outlier_count", "trimmed_weeks", "first_week", "last_week", "status", "reason"])
		if excluded_rows:
			report = pd.concat([report, pd.DataFrame(excluded_rows)], ignore_index=True)
		report.to_csv(processed / "ceda_validation_report.csv", index=False)
		return pd.DataFrame(columns=["commodity", "state", "week", "modal_price", "min_price", "max_price", "is_filled", "is_outlier"]), {"rows_before": 0, "rows_after": 0}

	daily = pd.concat(frames, ignore_index=True)
	daily["date"] = parse_dates(daily["date"])
	for column in ("min_price", "max_price", "modal_price"):
		daily[column] = pd.to_numeric(daily[column], errors="coerce")
	daily = daily.dropna(subset=["date", "modal_price"])
	daily = daily[daily["modal_price"] > 0]
	daily = daily.sort_values(["commodity", "state", "date"]).drop_duplicates(["commodity", "state", "date"], keep="last")
	daily["is_outlier"] = False
	for keys, indexes in daily.groupby(["commodity", "state"], dropna=False).groups.items():
		group_daily = daily.loc[indexes].sort_values("date")
		series = group_daily.set_index("date")["modal_price"]
		median = series.rolling("30D", center=True, min_periods=10).median()
		flags = series.lt(0.4 * median).fillna(False)
		daily.loc[group_daily.index, "is_outlier"] = flags.to_numpy()
	daily.to_csv(interim / "ceda_daily_cleaned.csv", index=False)

	daily["week"] = _week_end(daily["date"])
	weekly_parts: list[pd.DataFrame] = []
	for (commodity, state), group in daily.groupby(["commodity", "state"], dropna=False):
		weekly = group.groupby("week")[["min_price", "max_price", "modal_price"]].median()
		weekly["is_outlier"] = group.groupby("week")["is_outlier"].max().astype(bool)
		min_days = int(config.get("cleaning", {}).get("min_days_per_week", 4))
		daily_counts = group.groupby("week").size()
		trimmed = 0
		while not weekly.empty and int(daily_counts.get(weekly.index.max(), 0)) < min_days:
			weekly = weekly.iloc[:-1]
			trimmed += 1
		trimmed_weeks[(commodity, state)] = trimmed
		if weekly.empty:
			continue
		full_weeks = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-SUN")
		weekly = weekly.reindex(full_weeks)
		weekly, _ = _fill_short_log_gaps(weekly)
		weekly["commodity"], weekly["state"] = commodity, state
		weekly_parts.append(weekly.reset_index(names="week"))
	weekly_all = pd.concat(weekly_parts, ignore_index=True)
	min_history = int(config.get("cleaning", {}).get("min_history_weeks", 52))
	max_missing = float(config.get("cleaning", {}).get("max_series_missing_fraction", 0.40))
	report = _ceda_series_report(weekly_all, daily, max_missing, min_history, trimmed_weeks)
	if excluded_rows:
		report = pd.concat([report, pd.DataFrame(excluded_rows)], ignore_index=True)
	passed_report = report[
		(report["status"] == "PASSED")
		& (report["weeks"] >= min_history)
		& (report["weekly_missing_fraction"] <= max_missing)
	]
	passed_keys = set(passed_report.set_index(["commodity", "state"]).index)
	weekly_all["key"] = list(zip(weekly_all["commodity"], weekly_all["state"]))
	weekly_clean = weekly_all[weekly_all["key"].isin(passed_keys)].drop(columns="key")
	weekly_clean = weekly_clean[["commodity", "state", "week", "modal_price", "min_price", "max_price", "is_filled", "is_outlier"]]
	report.to_csv(processed / "ceda_validation_report.csv", index=False)
	weekly_clean.to_csv(processed / "ceda_weekly_clean.csv", index=False)
	weekly_clean.to_csv(processed / "mandi_weekly.csv", index=False)
	print("\nCEDA weekly validation report")
	print(report.to_string(index=False))
	return weekly_clean, {
		"rows_before": rows_before, "rows_after": len(daily), "validation_rows": len(report),
		"passed_series": len(passed_keys), "weekly_rows": len(weekly_clean),
		"trimmed_weeks": sum(trimmed_weeks.values()),
	}


def clean_mandi_data(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
	"""Clean records and return daily data plus auditable quality statistics.

	Missing prices are interpolated only inside each market/commodity series,
	then edge gaps are carried forward/backward. Series with excessive missing
	values are removed instead of being filled with an unreliable global value.
	Outliers are flagged, not deleted, because seasonal and festival spikes can
	be genuine market behavior.
	"""
	before = len(frame)
	if frame.empty:
		return pd.DataFrame(), {"rows_before": 0, "rows_after": 0, "empty_input": True}

	result = standardize_columns(frame)
	for column in ("state", "market", "district", "variety"):
		result[column] = result[column].map(normalize_text)
	result["commodity"] = result["commodity"].map(normalize_commodity)
	result["date"] = parse_dates(result["date"])
	result["modal_price"] = convert_to_rs_per_quintal(result["modal_price"], result["unit"])
	for column in ("min_price", "max_price", "arrival_quantity"):
		if column in result:
			result[column] = pd.to_numeric(
				result[column].astype(str).str.replace(",", "", regex=False), errors="coerce"
			)
	result = result.dropna(subset=["date", "market", "commodity"])
	result.loc[result["modal_price"] <= 0, "modal_price"] = np.nan
	result = result.sort_values(["market", "commodity", "date"])
	result = result.drop_duplicates(subset=["date", "market", "commodity", "variety"], keep="last")

	missing_before_fill = int(result["modal_price"].isna().sum())
	result["missing_price"] = result["modal_price"].isna()
	result["modal_price"] = result.groupby(["market", "commodity"], dropna=False)["modal_price"].transform(
		lambda values: values.interpolate(limit_area="inside")
	)
	result["modal_price"] = result.groupby(["market", "commodity"], dropna=False)["modal_price"].transform(
		lambda values: values.ffill().bfill()
	)
	result = _add_outlier_flags(result, int(config.get("cleaning", {}).get("rolling_window_days", 30)))
	report = {
		"rows_before": before,
		"rows_after": len(result),
		"duplicates_removed": max(0, before - len(result)),
		"missing_prices_before_fill": missing_before_fill,
		"missing_prices_after_fill": int(result["modal_price"].isna().sum()),
		"outliers_flagged": int(result["price_outlier"].sum()),
		"series": int(_series_key(result).nunique()),
		"date_min": result["date"].min().date().isoformat() if not result.empty else None,
		"date_max": result["date"].max().date().isoformat() if not result.empty else None,
	}
	return result, report


def aggregate_weekly(daily: pd.DataFrame, frequency: str = "W-SUN") -> pd.DataFrame:
	"""Aggregate daily prices by market and commodity without using future rows."""
	if daily.empty:
		return daily.copy()
	group_columns = ["market", "commodity", "state"]
	aggregations: dict[str, str] = {"modal_price": "mean"}
	for optional_price in ("min_price", "max_price"):
		if optional_price in daily:
			aggregations[optional_price] = "mean"
	if "arrival_quantity" in daily:
		aggregations["arrival_quantity"] = "sum"
	aggregations["price_outlier"] = "max"
	weekly = daily.set_index("date").groupby(group_columns, dropna=False).resample(frequency).agg(aggregations).reset_index()
	return weekly.dropna(subset=["modal_price"])


def run_cleaning(config_path: str | Path = "config.yaml") -> dict[str, Any]:
	config = load_config(config_path)
	configure_logging(config)
	raw_dir = Path(config["paths"]["raw_data"])
	ceda_sources = sorted((raw_dir / "ceda").glob("*.csv"))
	if ceda_sources:
		_, report = build_ceda_state_weekly(ceda_sources, config)
		pd.DataFrame([report]).to_json(
			Path(config["paths"]["processed_data"]) / "data_quality_report.json",
			orient="records",
			indent=2,
		)
		LOGGER.info("CEDA state-level cleaning report: %s", report)
		return report

	sources = sorted(raw_dir.glob("*.csv"))
	frames: list[pd.DataFrame] = []
	for path in sources:
		try:
			frame = pd.read_csv(path)
		except pd.errors.EmptyDataError:
			continue
		LOGGER.info("Reading raw source %s with columns %s", path, list(frame.columns))
		frames.append(frame)
	raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
	interim = Path(config["paths"]["interim_data"])
	processed = Path(config["paths"]["processed_data"])
	interim.mkdir(parents=True, exist_ok=True)
	processed.mkdir(parents=True, exist_ok=True)
	canonical_names = {column: canonical_column(column) for column in raw.columns}
	LOGGER.info("Mapped columns: %s", canonical_names)
	canonical_columns = set(canonical_names.values())
	if {"market_id", "commodity_id"}.issubset(canonical_columns):
		weekly, report = build_ceda_weekly(raw, config)
	else:
		daily, report = clean_mandi_data(raw, config)
		daily.to_csv(interim / "mandi_daily_cleaned.csv", index=False)
		weekly = aggregate_weekly(daily, config.get("cleaning", {}).get("weekly_aggregation", "W-SUN"))
		weekly.to_csv(processed / "mandi_weekly.csv", index=False)
	pd.DataFrame([report]).to_json(processed / "data_quality_report.json", orient="records", indent=2)
	LOGGER.info("Cleaning report: %s", report)
	return report


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="config.yaml")
	run_cleaning(parser.parse_args().config)
