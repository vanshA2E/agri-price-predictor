"""Local Streamlit dashboard for stored CEDA forecasts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import ROOT, forecast

st.set_page_config(page_title="Agri Price Predictor", layout="wide")
DATA = ROOT / "data" / "processed"
weekly = pd.read_csv(DATA / "ceda_weekly_clean.csv", parse_dates=["week"])
metrics = pd.read_csv(DATA / "metrics.csv")
last_date = weekly["week"].max().date().isoformat()

st.title("Weekly Agri Price Predictor")
st.info(f"Data available up to {last_date}")

commodities = sorted(weekly["commodity"].unique())
states = sorted(weekly["state"].unique())
commodity = st.selectbox("Commodity", commodities)
state = st.selectbox("State", states)
mode = st.radio("Mode", ["Forecast", "REPLAY"], horizontal=True)

series = weekly[(weekly["commodity"] == commodity) & (weekly["state"] == state)].sort_values("week")
if mode == "REPLAY":
	choices = series["week"].iloc[:-4] if len(series) > 4 else series["week"]
	as_of = st.selectbox("Historical as-of week", choices, format_func=lambda value: pd.Timestamp(value).date().isoformat())
	as_of_string = pd.Timestamp(as_of).date().isoformat()
else:
	as_of_string = None

if st.button("Generate forecast", type="primary"):
	try:
		result = forecast(commodity, state, 4, as_of_string)
		st.metric("Forecast price (Rs/quintal)", f"{result['forecast_price']:,.0f}")
		st.caption(f"Model: {result['model']} | Target week: {result['target_week']}")
		st.write(f"90% interval: Rs {result['lower_90']:,.0f} to Rs {result['upper_90']:,.0f}")
		chart = series.tail(104).copy()
		forecast_point = pd.DataFrame({"week": [pd.Timestamp(result["target_week"])], "modal_price": [result["forecast_price"]]})
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=chart["week"], y=chart["modal_price"], mode="lines", name="History"))
		fig.add_trace(go.Scatter(x=forecast_point["week"], y=forecast_point["modal_price"], mode="markers", name="Forecast", marker={"size": 11}))
		fig.add_trace(go.Scatter(x=[forecast_point["week"].iloc[0], forecast_point["week"].iloc[0]], y=[result["lower_90"], result["upper_90"]], mode="lines", name="90% interval"))
		if mode == "REPLAY":
			actual = series[series["week"] == pd.Timestamp(result["target_week"])]
			if not actual.empty:
				st.success(f"Replay actual: Rs {actual.iloc[0]['modal_price']:,.0f} | Forecast error: Rs {actual.iloc[0]['modal_price'] - result['forecast_price']:,.0f}")
				fig.add_trace(go.Scatter(x=actual["week"], y=actual["modal_price"], mode="markers", name="Replay actual", marker={"symbol": "x", "size": 12}))
		st.plotly_chart(fig, width="stretch")
	except (KeyError, ValueError, FileNotFoundError) as exc:
		st.error(str(exc))

st.subheader("Model comparison")
comparison = metrics[(metrics["scope"] == "commodity") & (metrics["commodity"] == commodity) & (metrics["split"] == "test")]
st.dataframe(comparison[["model", "mae", "rmse", "mape", "mase", "directional_accuracy", "improvement_vs_naive_pct"]], width="stretch", hide_index=True)
