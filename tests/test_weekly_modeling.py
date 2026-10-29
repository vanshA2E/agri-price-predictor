from pathlib import Path

import numpy as np
import pandas as pd

from src.data_cleaning import _fill_short_log_gaps, build_ceda_state_weekly
from src.weekly_modeling import build_weekly_features, split_by_global_calendar


def weekly_frame(rows: int = 120) -> pd.DataFrame:
    return pd.DataFrame({
        "commodity": ["Onion"] * rows,
        "state": ["Rajasthan"] * rows,
        "week": pd.date_range("2020-01-05", periods=rows, freq="W-SUN"),
        "modal_price": np.arange(100, 100 + rows, dtype=float),
        "min_price": np.arange(90, 90 + rows, dtype=float),
        "max_price": np.arange(110, 110 + rows, dtype=float),
        "is_filled": [False] * rows,
        "is_outlier": [False] * rows,
    })


def test_weekly_features_are_shifted_and_target_is_future() -> None:
    features = build_weekly_features(weekly_frame(120), {"model": {"horizon_weeks": 4}})
    first = features.iloc[0]
    assert first["modal_price_lag_1w"] == 151
    assert first["modal_price_rolling_mean_4w"] == 149.5
    assert first["target"] == 156


def test_global_split_has_common_dates_and_horizon_gaps() -> None:
    frame = weekly_frame()
    train, validation, test, cuts = split_by_global_calendar(frame, 0.70, 0.15, 4)
    assert train["week"].max() < pd.Timestamp(cuts["train_end_exclusive"])
    assert validation["week"].min() == pd.Timestamp(cuts["validation_start_inclusive"])
    assert test["week"].min() == pd.Timestamp(cuts["test_start_inclusive"])
    assert set(train["week"]).isdisjoint(validation["week"])
    assert set(validation["week"]).isdisjoint(test["week"])
    assert len(set(frame["week"]) - set(train["week"]) - set(validation["week"]) - set(test["week"])) >= 8


def test_short_gap_fill_does_not_fill_long_gap() -> None:
    weekly = pd.DataFrame(
        {"modal_price": [100, np.nan, np.nan, 110, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 130],
         "min_price": [90, np.nan, np.nan, 100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 120],
         "max_price": [110, np.nan, np.nan, 120, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 140]},
        index=pd.date_range("2024-01-07", periods=13, freq="W-SUN"),
    )
    filled, count = _fill_short_log_gaps(weekly)
    assert count == 2
    assert filled.iloc[1:3]["is_filled"].all()
    assert filled.iloc[4:11]["modal_price"].isna().all()


def test_trailing_week_is_trimmed(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "ceda"
    raw_dir.mkdir(parents=True)
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    pd.DataFrame({"date": dates, "min_price": 90, "max_price": 110, "modal_price": range(100, 106)}).to_csv(
        raw_dir / "onion__rajasthan__2015-01-01__2019-12-30.csv", index=False
    )
    weekly, report = build_ceda_state_weekly(
        [raw_dir / "onion__rajasthan__2015-01-01__2019-12-30.csv"],
        {"paths": {"interim_data": str(tmp_path / "interim"), "processed_data": str(tmp_path / "processed")},
         "ceda": {"commodities": {"Onion": 23}, "states": {"Rajasthan": 8}},
         "cleaning": {"min_days_per_week": 4, "min_history_weeks": 1, "max_series_missing_fraction": 0.40}},
    )
    assert report["weekly_rows"] == 1
    assert report["trimmed_weeks"] == 1
    assert weekly["week"].max() == pd.Timestamp("2024-01-07")
