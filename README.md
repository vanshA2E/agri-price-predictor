# AI-ML Agri-Horticultural Price Predictor

A reproducible college minor project for collecting real mandi prices and comparing models that forecast onion, potato, tomato, and pulse prices. The project is designed around a strict time-based evaluation so future information cannot leak into training features.

## Current Status

- **Scaffold:** complete.
- **Phase 1:** implemented in `src/data_collection.py`.
- **Phase 2:** implemented in `src/data_cleaning.py` with daily/weekly outputs and quality reporting.
- **Phases 4-5:** implemented in `src/feature_engineering.py` and `src/split_data.py`.
- **Phase 6:** the weekly CEDA comparison is implemented in `src/weekly_modeling.py`; current results are in `data/processed/metrics.csv`.
- **Weekly CEDA modeling:** implemented in `src/features.py` and `src/weekly_modeling.py` with four-week targets, global calendar splits, baselines, and LightGBM/HistGradientBoosting fallback.
- **Phases 3 and 7:** notebook analysis and deployment behavior remain to be completed after real data is available.
- **Results:** generated from the local CEDA files; no synthetic dataset is included.

Current CEDA modeling uses 22 passed series. The two excluded Madhya Pradesh series are Green Gram (source gap 544 days from 2023-12-21) and Black Gram (source gap 228 days from 2023-12-21). The source currently ends at 2025-10-30. With a four-week target and horizon-sized split gaps, the overall test MAE is 302.18 for naive, 957.52 for seasonal-naive, and 364.46 for LightGBM; LightGBM currently beats naive only for Tomato.

## Data Sources

1. **data.gov.in Agmarknet API:** current daily commodity prices. Set `DATA_GOV_API_KEY` in `.env`; responses are saved untouched under `data/raw/` with pagination.
2. **Agmarknet manual CSV fallback:** download historical files from [agmarknet.gov.in](https://agmarknet.gov.in/) and place the original CSV files in `data/raw/`. The collector merges them without silently filling missing history.
3. **Department of Consumer Affairs:** reserved as an optional retail-price source. It remains disabled until an endpoint and schema are verified.
4. **Open-Meteo archive:** optional state-centroid weather data. Enable it in `config.yaml` only when weather features are required.

The CEDA target is weekly state-level data for Rajasthan, Madhya Pradesh, and Uttar Pradesh. CEDA currently ends at 2025-10-30 because of source update lag; this is expected, not truncation. The collector reports what it actually found; it does not fabricate missing observations.

## Setup

```bash
cd agri-price-predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add a real data.gov.in key to `.env`:

```dotenv
DATA_GOV_API_KEY=your_real_key
```

Never commit `.env` or API keys.

## Phase 1: Collect Data

```bash
python -m src.data_collection --config config.yaml
```

Check the logs and `data/raw/`. Expected outputs include raw JSON API pages, `data_gov_mandi_records.csv`, manually supplied Agmarknet files, and optional weather JSON. With no API key, the command skips the API and loads any manual CSVs already in `data/raw/`.

### CEDA Agmarknet API

The verified CEDA downloader is [src/ceda_downloader.py](src/ceda_downloader.py). Put the CEDA credential in the ignored `.env` file as `CEDA_API_KEY`; `.env.example` contains only a placeholder. CEDA data is treated as state-level; probe JSON files under `data/raw/ceda/probe/` are excluded from cleaning.

Run the CEDA workflow in this order:

```bash
# 1. Plan requests without calling the API.
python -m src.ceda_downloader --dry-run

# 2. Make the three documented probe cases and save responses.
python -m src.ceda_downloader --probe

# 3. Run a one-year Onion/Rajasthan pilot.
python -m src.ceda_downloader --commodity Onion --state Rajasthan \
	--start 2024-01-01 --end 2024-12-31

# 4. After inspecting the pilot, run the full configured download.
python -m src.ceda_downloader

# Clean CEDA files into validation output and weekly state series.
python -m src.data_cleaning --config config.yaml

# Build weekly features, evaluate baselines and the global boosting model.
python -m src.weekly_modeling --config config.yaml
```

CEDA files are resumable: existing window CSVs are skipped and successful empty windows are retained as header-only marker files. The probe summary controls the largest safe window for subsequent downloads; without a successful probe, the configured default is 365 days. The state-level cleaner writes `ceda_validation_report.csv`, `mandi_weekly.csv`, and a quality report under `data/processed/`.

Weekly modeling reads `data/processed/ceda_weekly_clean.csv`, writes all validation/test metrics to `data/processed/metrics.csv`, and saves the global model under `models/ceda_weekly_*.joblib`.

## Local API and Dashboard

Start the API locally:

```bash
uvicorn app.api:app --reload --port 8000
```

Useful endpoints:

- `GET /health`
- `GET /series`
- `POST /predict` with `commodity`, `state`, optional `as_of_week`, and `horizon_weeks: 4`

Start the dashboard in another terminal:

```bash
streamlit run app/streamlit_app.py
```

The dashboard supports normal forecasting and **REPLAY** mode, where a historical week is selected and compared with the actual price four weeks later. Both the API and UI use local stored data and saved model artifacts only.

## Docker

```bash
docker compose up --build
```

The API is exposed at `http://localhost:8000` and the dashboard at `http://localhost:8501`.

## Deployment

For Streamlit Community Cloud, push this repository to GitHub, create a new app using `app/streamlit_app.py` as the entry point, and set the repository root as the working directory. The committed `data/processed/`, `models/`, `requirements.txt`, and `src/` files are required by the dashboard. No CEDA API key is needed to run the dashboard because prediction uses stored artifacts.

For the API, deploy the Docker image to a container host and expose port `8000`. The dashboard can call the local prediction helper directly, so a separate API service is optional for the free Streamlit deployment.

Before moving to cleaning, verify:

- dates cover the intended five-year period;
- commodity, state, modal/min/max price columns are present;
- prices have the expected unit and currency;
- all three configured states have the expected commodity windows;
- API pagination did not stop early.

## Planned Methodology

1. **Cleaning:** standardize names, dates, units, duplicates, invalid prices, gaps, and outlier flags; produce daily and weekly datasets plus a quality report.
2. **EDA:** inspect commodity trends, seasonality, yearly patterns, state differences, weather relationships, and volatility.
3. **Features:** calendar variables, lagged prices, rolling statistics, and encoded commodity/state fields. Every feature must be computable using information available at prediction time.
4. **Splitting:** chronological global-calendar 70% training, 15% validation, and 15% test slices with a four-week gap around each boundary.
5. **Models:** naive and 52-week seasonal-naive baselines plus one global LightGBM model, with a sklearn HistGradientBoosting fallback.
6. **Evaluation:** MAE, RMSE, and MAPE for a four-week horizon, overall, per commodity, and per state; filled target weeks are excluded from scoring.
7. **Deployment:** save the best per-commodity model and metadata with joblib, then expose predictions through FastAPI and Streamlit.

## Project Layout

```text
agri-price-predictor/
├── data/{raw,interim,processed}/
├── notebooks/{01_eda.ipynb,02_modeling.ipynb}
├── src/{data_collection,data_cleaning,features,weekly_modeling,feature_engineering,split_data,train,evaluate,predict}.py
├── app/{api.py,streamlit_app.py}
├── models/
├── tests/
├── config.yaml
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Tests and Containers

Run the current tests after installing dependencies:

```bash
pytest -q
```

Start the API and UI together after the deployment modules are implemented:

```bash
docker compose up --build
```

The planned local URLs are `http://localhost:8000/docs` and `http://localhost:8501`.

For a free deployment, the simplest final target is Streamlit Community Cloud for the UI plus a small Render web service for FastAPI. Add the generated model artifacts and environment variables only through the platform secret/configuration UI.

## Results Table

Populate this table only from the final held-out test set:

| Commodity | Horizon | Model | MAE (Rs/qtl) | RMSE | MAPE (%) | R2 |
|---|---:|---|---:|---:|---:|---:|
| Onion | 7 days | TBD | TBD | TBD | TBD | TBD |
| Potato | 7 days | TBD | TBD | TBD | TBD | TBD |
| Tomato | 7 days | TBD | TBD | TBD | TBD | TBD |
| Pulses | 7 days | TBD | TBD | TBD | TBD | TBD |

## Future Scope

Add richer arrival and weather joins, district-level geospatial features, probabilistic forecasts, holiday calendars, drift monitoring, model retraining schedules, and authenticated production APIs.

## Final Weekly Results

| Split | Naive MAE | Global LightGBM MAE | Commodity LightGBM MAE | Blend MAE |
|---|---:|---:|---:|---:|
| Validation | 345.53 | 322.15 | 303.76 | 315.60 |
| Test | 302.18 | 285.26 | 273.82 | 274.05 |

The model registry selects the best model per commodity using validation results only. Full MAE, RMSE, MAPE, MASE, directional accuracy, and improvement values are in `data/processed/metrics.csv` and `metrics_by_series.csv`.

## Limitations

- CEDA data currently ends at 2025-10-30 because of source update lag.
- Two Madhya Pradesh pulse series are excluded because of long source gaps.
- Rajasthan Arhar is thin and loses seven trailing weekly periods during the minimum-observation trim.
- ETS/SARIMA are available as optional state-space evaluations but are substantially slower than the boosting models; SARIMA is limited to one state per commodity.
- The saved model is not a guarantee of future market behavior; weather, arrivals, policy, and shocks are not currently included.
- Streamlit Community Cloud deployment must be performed from the user’s GitHub account; it was not started from this local environment.
