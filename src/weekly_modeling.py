"""Weekly log-return modeling and local evaluation for CEDA state series."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluate import calculate_metrics

LOGGER = logging.getLogger(__name__)
try:
    from lightgbm import LGBMRegressor, early_stopping
except ImportError:  # pragma: no cover
    LGBMRegressor = None
    early_stopping = None
try:
    from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing as StateSpaceETS
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:  # pragma: no cover
    StateSpaceETS = None
    SARIMAX = None

LAGS = (1, 2, 4, 8, 52)
ROLLING_WINDOWS = (4, 8, 13)
CATEGORICAL_COLUMNS = ["commodity", "state"]
METRIC_COLUMNS = [
    "split", "scope", "commodity", "state", "model", "horizon_weeks", "mae", "rmse", "mape", "mase",
    "directional_accuracy", "n_scored", "skipped_filled_target", "naive_mae", "improvement_vs_naive_pct", "beats_naive",
]


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_weekly_features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build past-only features and a future log-price-change target."""
    if frame.empty:
        return frame.copy()
    horizon = int(config.get("model", {}).get("horizon_weeks", 4))
    result = frame.copy()
    result["week"] = pd.to_datetime(result["week"], errors="coerce")
    result = result.dropna(subset=["week", "commodity", "state", "modal_price"])
    result = result.sort_values(["commodity", "state", "week"]).reset_index(drop=True)
    groups = result.groupby(["commodity", "state"], sort=False, dropna=False)
    for lag in LAGS:
        result[f"modal_price_lag_{lag}w"] = groups["modal_price"].shift(lag)
        for flag in ("is_filled", "is_outlier"):
            if flag in result:
                result[f"{flag}_lag_{lag}w"] = groups[flag].shift(lag).astype("boolean").fillna(False)
    for window in ROLLING_WINDOWS:
        result[f"modal_price_rolling_mean_{window}w"] = groups["modal_price"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).mean()
        )
        result[f"modal_price_rolling_std_{window}w"] = groups["modal_price"].transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).std()
        )
    result["week_of_year"] = result["week"].dt.isocalendar().week.astype(int)
    result["month"] = result["week"].dt.month
    result["is_festival_season"] = result["month"].isin(config.get("features", {}).get("festival_months", [10, 11])).astype(int)
    result["target_price"] = groups["modal_price"].shift(-horizon)
    result["target"] = result["target_price"]
    result["target_is_filled"] = groups["is_filled"].shift(-horizon).astype("boolean").fillna(False)
    result["target_log_change"] = np.log(result["target_price"]) - np.log(result["modal_price"])
    result = result.replace([np.inf, -np.inf], np.nan)
    required = [
        *(f"modal_price_lag_{lag}w" for lag in LAGS),
        *(f"modal_price_rolling_{stat}_{window}w" for window in ROLLING_WINDOWS for stat in ("mean", "std")),
        "week_of_year", "month", "is_festival_season", "target_price", "target_log_change",
    ]
    return result.dropna(subset=required).reset_index(drop=True)


def split_by_global_calendar(frame: pd.DataFrame, train_fraction: float, validation_fraction: float, horizon_weeks: int = 0):
    """Split on common calendar dates and leave horizon-sized boundary gaps."""
    weeks = sorted(pd.to_datetime(frame["week"], errors="coerce").dropna().unique())
    if len(weeks) < 3 or train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Invalid split configuration or insufficient weeks")
    train_index = min(len(weeks) - 2, max(1, int(len(weeks) * train_fraction)))
    validation_index = min(len(weeks) - 1, max(train_index + 1, int(len(weeks) * (train_fraction + validation_fraction))))
    train_cut, validation_cut = pd.Timestamp(weeks[train_index]), pd.Timestamp(weeks[validation_index])
    delta = pd.to_timedelta(int(horizon_weeks), unit="W")
    validation_start, test_start = train_cut + delta, validation_cut + delta
    train = frame[frame["week"] < train_cut].copy()
    validation = frame[(frame["week"] >= validation_start) & (frame["week"] < validation_cut)].copy()
    test = frame[frame["week"] >= test_start].copy()
    cuts = {
        "train_end_exclusive": train_cut.date().isoformat(),
        "validation_start_inclusive": validation_start.date().isoformat(),
        "validation_end_exclusive": validation_cut.date().isoformat(),
        "test_start_inclusive": test_start.date().isoformat(),
    }
    return train, validation, test, cuts


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = {"week", "modal_price", "target", "target_price", "target_log_change", "is_filled", "is_outlier", "target_is_filled"}
    categorical = [column for column in CATEGORICAL_COLUMNS if column in frame.columns]
    numeric = [column for column in frame.select_dtypes(include=[np.number, "bool"]).columns if column not in excluded]
    return numeric, categorical


def make_model(seed: int, numeric: list[str], categorical: list[str], kind: str = "lightgbm") -> tuple[Pipeline, str]:
    if kind == "ridge":
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        model: Any = Ridge(alpha=10.0)
        name = "ridge"
    else:
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        if kind == "lightgbm" and LGBMRegressor is not None:
            model = LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=31, random_state=seed, verbosity=-1)
            name = "lightgbm"
        else:
            model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=31, random_state=seed)
            name = "hist_gradient_boosting"
    transformers = [
        ("numeric", numeric_pipe, numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ]
    return Pipeline([("preprocessor", ColumnTransformer(transformers)), ("model", model)]), name


def fit_model(model: Pipeline, name: str, train: pd.DataFrame, validation: pd.DataFrame, features: list[str]) -> Pipeline:
    if name != "lightgbm":
        return model.fit(train[features], train["target_log_change"])
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    x_train = preprocessor.fit_transform(train[features])
    if validation.empty or early_stopping is None:
        estimator.fit(x_train, train["target_log_change"])
    else:
        estimator.fit(
            x_train, train["target_log_change"],
            eval_X=preprocessor.transform(validation[features]), eval_y=validation["target_log_change"],
            callbacks=[early_stopping(50, verbose=False)],
        )
    return model


def _price_from_return(frame: pd.DataFrame, returns: Any) -> pd.Series:
    return frame["modal_price"].to_numpy(dtype=float) * np.exp(np.asarray(returns, dtype=float))


def _directional_accuracy(actual: pd.Series, predicted: pd.Series, current: pd.Series) -> float:
    actual_change = actual.to_numpy() - current.to_numpy()
    predicted_change = predicted.to_numpy() - current.to_numpy()
    mask = np.abs(actual_change / current.to_numpy()) >= 0.01
    return float((np.sign(actual_change[mask]) == np.sign(predicted_change[mask])).mean() * 100) if mask.any() else np.nan


def _naive_scales(train: pd.DataFrame) -> dict[tuple[str, str], float]:
    result = {}
    for key, group in train.groupby(["commodity", "state"], dropna=False):
        values = group.sort_values("week")["modal_price"].to_numpy(dtype=float)
        result[key] = float(np.mean(np.abs(values[4:] - values[:-4]))) if len(values) > 4 else np.nan
    return result


def _metrics(actual: pd.DataFrame, predicted: pd.Series, split: str, model: str, horizon: int, skipped: int, scales: dict[tuple[str, str], float], naive_lookup: dict[tuple[str, str, str], float]) -> list[dict[str, Any]]:
    valid = ~actual["target_is_filled"].astype(bool) & actual["target_price"].notna() & predicted.notna()
    scored, values = actual.loc[valid], predicted.loc[valid]
    rows = []

    def add(scope: str, commodity: str, state: str, group: pd.DataFrame, group_values: pd.Series) -> None:
        base = calculate_metrics(group["target_price"], group_values)
        scales_for_rows = np.asarray([scales.get((c, s), np.nan) for c, s in zip(group["commodity"], group["state"])])
        errors = np.abs(group["target_price"].to_numpy() - group_values.to_numpy())
        mase = float(np.nanmean(errors / scales_for_rows)) if np.isfinite(scales_for_rows).any() else np.nan
        lookup_key = (split, scope, commodity, state) if scope == "series" else (split, scope, commodity)
        naive_mae = naive_lookup.get(lookup_key, np.nan)
        rows.append({
            "split": split, "scope": scope, "commodity": commodity, "state": state, "model": model, "horizon_weeks": horizon,
            **base, "mase": mase, "directional_accuracy": _directional_accuracy(group["target_price"], group_values, group["modal_price"]),
            "n_scored": len(group), "skipped_filled_target": skipped, "naive_mae": naive_mae,
            "improvement_vs_naive_pct": (naive_mae - base["mae"]) / naive_mae * 100 if np.isfinite(naive_mae) and naive_mae else np.nan,
            "beats_naive": bool(base["mae"] < naive_mae) if np.isfinite(naive_mae) else False,
        })

    add("overall", "ALL", "ALL", scored, values)
    for commodity, group in scored.groupby("commodity", dropna=False):
        add("commodity", str(commodity), "ALL", group, values.loc[group.index])
    for state, group in scored.groupby("state", dropna=False):
        add("state", "ALL", str(state), group, values.loc[group.index])
    for (commodity, state), group in scored.groupby(["commodity", "state"], dropna=False):
        add("series", str(commodity), str(state), group, values.loc[group.index])
    return rows


def _state_space_predictions(frame: pd.DataFrame, initial_end: pd.Timestamp, kind: str) -> pd.Series:
    """Forecast test origins after one train+validation fit, updating without refitting."""
    if (kind == "ets" and StateSpaceETS is None) or (kind == "sarima" and SARIMAX is None):
        LOGGER.warning("%s skipped: statsmodels is not installed.", kind.upper())
        return pd.Series(dtype=float)
    predictions: dict[Any, float] = {}
    selected_states: set[str] = set()
    for (commodity_value, _state), group in frame.sort_values(["commodity", "state"]).groupby(["commodity", "state"], dropna=False):
        commodity = str(commodity_value)
        if commodity in selected_states:
            continue
        selected_states.add(commodity)
        group = group.sort_values("week")
        history = group[group["week"] <= initial_end]
        if len(history) < 20:
            continue
        try:
            series = history.set_index("week")["modal_price"].asfreq("W-SUN")
            if kind == "ets":
                fitted = StateSpaceETS(series, trend=True, seasonal=None, initialization_method="estimated").fit(disp=False)
            else:
                fitted = SARIMAX(series, order=(1, 1, 1), trend="c", enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            for index, row in group[group["week"] > initial_end].iterrows():
                predictions[index] = float(np.asarray(fitted.forecast(4))[3])
                fitted = fitted.append([row["modal_price"]], refit=False)
        except Exception as exc:
            LOGGER.warning("%s skipped for %s/%s: %s", kind, commodity, group["state"].iloc[0], exc)
    return pd.Series(predictions, dtype=float)


def _rolling_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["model", "folds", "mean_mae", "std_mae"])
    weeks = sorted(predictions["week"].unique())
    fold_map = {week: min(3, index * 4 // max(1, len(weeks))) for index, week in enumerate(weeks)}
    data = predictions.copy()
    data["fold"] = data["week"].map(fold_map)
    rows = []
    for (model, fold), group in data.groupby(["model", "fold"]):
        group = group[group["predicted_price"].notna()]
        if not group.empty:
            rows.append({"model": model, "fold": fold, "mae": np.mean(np.abs(group["target_price"] - group["predicted_price"]))})
    return pd.DataFrame(rows).groupby("model")["mae"].agg(folds="count", mean_mae="mean", std_mae="std").reset_index()


def _save_plots(predictions: pd.DataFrame, artifact_path: Path) -> None:
    import matplotlib.pyplot as plt
    output = Path("outputs/figures")
    output.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = pd.DataFrame({"feature": model.named_steps["preprocessor"].get_feature_names_out(), "importance": estimator.feature_importances_}).sort_values("importance", ascending=False)
        importance.to_csv(output / "feature_importance.csv", index=False)
        importance.head(30).sort_values("importance").plot.barh(x="feature", y="importance", legend=False, figsize=(9, 8))
        plt.tight_layout(); plt.savefig(output / "feature_importance.png", dpi=140); plt.close()
    test = predictions[predictions["split"] == "test"].sort_values("week")
    if not test.empty:
        plt.figure(figsize=(10, 5)); plt.plot(test["week"], test["target_price"], label="actual"); plt.plot(test["week"], test["global_lightgbm"], label="global_lightgbm"); plt.plot(test["week"], test["naive"], label="naive"); plt.legend(); plt.tight_layout(); plt.savefig(output / "actual_vs_predicted_test.png", dpi=140); plt.close()


def train_weekly(config_path: str | Path = "config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    processed = Path(config["paths"]["processed_data"])
    model_dir = Path(config["paths"]["models"]); model_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(processed / "ceda_weekly_clean.csv")
    features = build_weekly_features(frame, config)
    features.to_csv(processed / "ceda_weekly_features.csv", index=False)
    horizon = int(config.get("model", {}).get("horizon_weeks", 4))
    train, validation, test, cuts = split_by_global_calendar(features, float(config["split"]["train_fraction"]), float(config["split"]["validation_fraction"]), horizon)
    train.to_csv(processed / "weekly_train.csv", index=False); validation.to_csv(processed / "weekly_val.csv", index=False); test.to_csv(processed / "weekly_test.csv", index=False)
    print(f"Global split cuts: {json.dumps(cuts)}")
    print(f"Split row counts: train={len(train)}, validation={len(validation)}, test={len(test)}")
    numeric, categorical = feature_columns(features); feature_names = numeric + categorical
    seed = int(config.get("model", {}).get("random_seed", 42))
    train_fit = train[~train["target_is_filled"]]; validation_fit = validation[~validation["target_is_filled"]]
    scales = _naive_scales(train_fit)
    global_model, global_name = make_model(seed, numeric, categorical, "lightgbm")
    global_model = fit_model(global_model, global_name, train_fit, validation_fit, feature_names)
    ridge, ridge_name = make_model(seed, numeric, categorical, "ridge"); ridge.fit(train_fit[feature_names], train_fit["target_log_change"])
    commodity_models = {}
    for commodity, group in train_fit.groupby("commodity"):
        model, name = make_model(seed, numeric, categorical, "lightgbm")
        commodity_models[commodity] = fit_model(model, name, group, validation_fit[validation_fit["commodity"] == commodity], feature_names)
        joblib.dump({"model": commodity_models[commodity], "features": feature_names, "commodity": commodity, "horizon_weeks": horizon}, model_dir / f"ceda_weekly_{commodity.lower().replace('/', '_').replace(' ', '_')}_lightgbm_4w.joblib")

    state_space = {
        "ets": _state_space_predictions(features, train["week"].max(), "ets"),
        "sarima": _state_space_predictions(features, train["week"].max(), "sarima"),
    }

    records = []
    for split_name, subset in (("validation", validation), ("test", test)):
        for index, row in subset.iterrows():
            one = pd.DataFrame([row]); record = {"index": index, "split": split_name, "week": row["week"], "commodity": row["commodity"], "state": row["state"], "target_price": row["target_price"], "modal_price": row["modal_price"], "target_is_filled": row["target_is_filled"]}
            record["naive"] = row["modal_price"]
            record["seasonal_naive"] = row["modal_price"] if pd.isna(row["modal_price_lag_52w"]) else row["modal_price_lag_52w"]
            record["global_lightgbm"] = _price_from_return(one, global_model.predict(one[feature_names]))[0]
            record["ridge"] = _price_from_return(one, ridge.predict(one[feature_names]))[0]
            commodity_model = commodity_models.get(row["commodity"])
            record["commodity_lightgbm"] = _price_from_return(one, commodity_model.predict(one[feature_names]))[0] if commodity_model else np.nan
            record["ets"] = state_space["ets"].get(index, np.nan)
            record["sarima"] = state_space["sarima"].get(index, np.nan)
            records.append(record)
    predictions = pd.DataFrame(records)
    validation_predictions = predictions[predictions["split"] == "validation"]
    choices = {}
    candidate_models = ("naive", "seasonal_naive", "ridge", "global_lightgbm", "commodity_lightgbm", "ets", "sarima")
    for commodity in sorted(predictions["commodity"].unique()):
        subset = validation_predictions[validation_predictions["commodity"] == commodity]
        def score(name: str) -> float:
            valid = ~subset["target_is_filled"] & subset[name].notna()
            return float(np.mean(np.abs(subset.loc[valid, "target_price"] - subset.loc[valid, name]))) if valid.any() else np.inf
        choices[commodity] = "naive" if subset.empty else min(candidate_models, key=score)
    predictions["selected_lightgbm"] = predictions.apply(
        lambda row: row["commodity_lightgbm"] if row["commodity"] in choices and choices[row["commodity"]] == "commodity_lightgbm" else row["global_lightgbm"],
        axis=1,
    )
    predictions["blend"] = 0.5 * predictions["naive"] + 0.5 * predictions["selected_lightgbm"]
    naive_lookup = {}
    for split_name in ("validation", "test"):
        subset = predictions[predictions["split"] == split_name]
        valid = ~subset["target_is_filled"]
        naive_lookup[(split_name, "overall", "ALL")] = np.mean(np.abs(subset.loc[valid, "target_price"] - subset.loc[valid, "naive"]))
        for commodity, group in subset.groupby("commodity"):
            good = ~group["target_is_filled"]; naive_lookup[(split_name, "commodity", commodity)] = np.mean(np.abs(group.loc[good, "target_price"] - group.loc[good, "naive"]))
        for state, group in subset.groupby("state"):
            good = ~group["target_is_filled"]; naive_lookup[(split_name, "state", state)] = np.mean(np.abs(group.loc[good, "target_price"] - group.loc[good, "naive"]))
        for (commodity, state), group in subset.groupby(["commodity", "state"]):
            good = ~group["target_is_filled"]; naive_lookup[(split_name, "series", commodity, state)] = np.mean(np.abs(group.loc[good, "target_price"] - group.loc[good, "naive"]))
    rows = []
    for split_name in ("validation", "test"):
        subset = predictions[predictions["split"] == split_name].set_index("index"); skipped = int(subset["target_is_filled"].sum())
        for name in ("naive", "seasonal_naive", "ridge", "global_lightgbm", "commodity_lightgbm", "ets", "sarima", "blend"):
            rows.extend(_metrics(subset, subset[name], split_name, name, horizon, skipped, scales, naive_lookup))
    metrics = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    metrics.to_csv(processed / "metrics.csv", index=False); metrics[metrics["scope"].isin(["commodity", "state", "series"])].to_csv(processed / "metrics_by_series.csv", index=False)
    (processed / "model_registry.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")
    artifact_path = model_dir / "ceda_weekly_global_lightgbm_4w.joblib"
    joblib.dump({"model": global_model, "features": feature_names, "horizon_weeks": horizon, "split_cuts": cuts, "trained_at": datetime.now(timezone.utc).isoformat()}, artifact_path)
    (model_dir / "ceda_weekly_global_lightgbm_4w.features.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    rolling = predictions.melt(id_vars=["split", "week", "target_price", "target_is_filled"], value_vars=["naive", "seasonal_naive", "ridge", "global_lightgbm", "commodity_lightgbm", "ets", "sarima", "blend"], var_name="model", value_name="predicted_price")
    _rolling_summary(rolling[~rolling["target_is_filled"]]).to_csv(processed / "rolling_origin_metrics.csv", index=False)
    _save_plots(predictions, artifact_path)
    arhar = metrics[(metrics["commodity"] == "Arhar (Tur/Red Gram) (Whole)") & (metrics["state"] == "Rajasthan")]
    if not arhar.empty: print("Rajasthan Arhar metrics:\n", arhar.to_string(index=False))
    LOGGER.info("Weekly modeling complete: %s", artifact_path)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", default="config.yaml")
    train_weekly(parser.parse_args().config)