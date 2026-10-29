"""FastAPI service backed only by local CEDA data and saved artifacts."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import ROOT, forecast

app = FastAPI(title="Agri Price Predictor API", version="1.0.0")


class PredictionRequest(BaseModel):
	commodity: str
	state: str
	horizon_weeks: int = Field(default=4, ge=1, le=4)
	as_of_week: str | None = None


def _weekly() -> pd.DataFrame:
	return pd.read_csv(ROOT / "data" / "processed" / "ceda_weekly_clean.csv", parse_dates=["week"])


@app.get("/health")
def health() -> dict[str, Any]:
	data = _weekly()
	model_path = ROOT / "models" / "ceda_weekly_global_lightgbm_4w.joblib"
	return {
		"status": "ok",
		"data_last_date": data["week"].max().date().isoformat() if not data.empty else None,
		"model_date": datetime.fromtimestamp(model_path.stat().st_mtime, timezone.utc).isoformat() if model_path.exists() else None,
	}


@app.get("/series")
def series() -> dict[str, Any]:
	data = _weekly()
	return {
		"commodities": sorted(data["commodity"].dropna().unique().tolist()),
		"states": sorted(data["state"].dropna().unique().tolist()),
		"last_date": data["week"].max().date().isoformat() if not data.empty else None,
	}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
	try:
		return forecast(request.commodity, request.state, request.horizon_weeks, request.as_of_week)
	except KeyError as exc:
		raise HTTPException(status_code=404, detail=str(exc)) from exc
	except (ValueError, FileNotFoundError) as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
