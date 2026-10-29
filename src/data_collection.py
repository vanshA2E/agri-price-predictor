"""Collect real mandi, optional retail, and optional weather data.

No synthetic records are generated in this module. If the data.gov.in API key is
missing or the API cannot be reached, the collector uses CSV files manually
placed in ``data/raw`` and reports the fallback clearly.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration and create configured output directories."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("raw_data", "interim_data", "processed_data", "models"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(config.get("logging", {}).get("file", "logs/agri_price_predictor.log")).parent.mkdir(
        parents=True, exist_ok=True
    )
    return config


def configure_logging(config: dict[str, Any]) -> None:
    """Configure console and file logging once for command-line execution."""
    logging.basicConfig(
        level=getattr(logging, config.get("logging", {}).get("level", "INFO").upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.get("logging", {}).get("file", "logs/agri_price_predictor.log")),
        ],
        force=True,
    )


def build_session(config: dict[str, Any]) -> requests.Session:
    """Build a requests session with bounded retries for transient failures."""
    collection = config["collection"]
    retry = Retry(
        total=collection.get("max_retries", 4),
        read=collection.get("max_retries", 4),
        connect=collection.get("max_retries", 4),
        backoff_factor=collection.get("backoff_factor", 1.5),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def safe_name(value: str) -> str:
    """Create a stable filename component."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def save_json(path: Path, payload: Any) -> None:
    """Save an API response without transforming it."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_data_gov_records(
    session: requests.Session,
    config: dict[str, Any],
    commodity: str,
    state: str,
) -> list[dict[str, Any]]:
    """Fetch all available pages for one commodity/state pair from data.gov.in."""
    api_key = os.getenv("DATA_GOV_API_KEY")
    if not api_key or api_key == "your_data_gov_api_key":
        LOGGER.warning("DATA_GOV_API_KEY is not configured; skipping API collection.")
        return []

    collection = config["collection"]
    base_url = f"{collection['data_gov_base_url'].rstrip('/')}/{collection['data_gov_resource_id']}"
    page_size = int(collection.get("page_size", 1000))
    records: list[dict[str, Any]] = []
    offset = 0
    end_date = date.today()
    start_date = end_date - timedelta(days=365 * int(collection.get("lookback_years", 5)))

    while True:
        params = {
            "api-key": api_key,
            "format": "json",
            "limit": page_size,
            "offset": offset,
            "filters[State]": state,
            "filters[Commodity]": commodity,
        }
        LOGGER.info("Fetching data.gov.in commodity=%s state=%s offset=%s", commodity, state, offset)
        response = session.get(base_url, params=params, timeout=collection.get("timeout_seconds", 30))
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("records", [])
        raw_path = Path(config["paths"]["raw_data"]) / (
            f"data_gov_{safe_name(commodity)}_{safe_name(state)}_{offset}.json"
        )
        save_json(raw_path, payload)
        records.extend(batch)
        total = int(payload.get("total", len(records)))
        if not batch or len(records) >= total or len(batch) < page_size:
            break
        offset += page_size
        time.sleep(float(collection.get("request_delay_seconds", 0.25)))

    LOGGER.info(
        "Fetched %s records for %s/%s; requested lookback starts %s",
        len(records), commodity, state, start_date.isoformat(),
    )
    return records


def load_manual_agmarknet_csvs(raw_dir: str | Path) -> pd.DataFrame:
    """Load manually downloaded Agmarknet CSVs without fabricating missing data."""
    raw_path = Path(raw_dir)
    files = sorted(raw_path.glob("*.csv"))
    if not files:
        LOGGER.warning("No manual Agmarknet CSVs found in %s.", raw_path)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for file_path in files:
        try:
            frame = pd.read_csv(file_path)
            frame["source_file"] = file_path.name
            frames.append(frame)
            LOGGER.info("Loaded %s rows from %s", len(frame), file_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            LOGGER.error("Could not read manual CSV %s: %s", file_path, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_consumer_affairs_prices(
    session: requests.Session, config: dict[str, Any]
) -> pd.DataFrame:
    """Placeholder for the optional retail-price source.

    The Department of Consumer Affairs endpoint and schema vary by publication.
    The project deliberately leaves this disabled until a verified endpoint is
    configured, avoiding an apparently successful but incorrect data mapping.
    """
    del session, config
    LOGGER.info("Consumer Affairs retail source is optional and not configured; skipping.")
    return pd.DataFrame()


def fetch_open_meteo_weather(
    session: requests.Session, config: dict[str, Any]
) -> pd.DataFrame:
    """Fetch optional daily weather observations for configured state centroids."""
    weather = config["collection"].get("weather", {})
    if not weather.get("enabled", False):
        LOGGER.info("Weather collection disabled in config.yaml; skipping.")
        return pd.DataFrame()

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * int(config["collection"].get("lookback_years", 5)))
    frames: list[pd.DataFrame] = []
    for state, coordinates in weather.get("latitude_longitude", {}).items():
        params = {
            "latitude": coordinates[0],
            "longitude": coordinates[1],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "auto",
        }
        response = session.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params,
            timeout=config["collection"].get("timeout_seconds", 30),
        )
        response.raise_for_status()
        payload = response.json()
        save_json(Path(config["paths"]["raw_data"]) / f"weather_open_meteo_{safe_name(state)}.json", payload)
        daily = pd.DataFrame(payload.get("daily", {}))
        if not daily.empty:
            daily["state"] = state
            frames.append(daily)
        time.sleep(float(config["collection"].get("request_delay_seconds", 0.25)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_api_records(records: Iterable[dict[str, Any]], config: dict[str, Any]) -> Path | None:
    """Write the unmodified API records as a convenience CSV for later cleaning."""
    frame = pd.DataFrame(list(records))
    if frame.empty:
        return None
    output = Path(config["paths"]["raw_data"]) / "data_gov_mandi_records.csv"
    frame.to_csv(output, index=False)
    LOGGER.info("Wrote %s untouched API records to %s", len(frame), output)
    return output


def collect(config_path: str | Path = "config.yaml") -> dict[str, int]:
    """Run configured collection sources and return record counts."""
    load_dotenv()
    config = load_config(config_path)
    configure_logging(config)
    session = build_session(config)
    all_records: list[dict[str, Any]] = []

    try:
        if os.getenv("DATA_GOV_API_KEY") in (None, "", "your_data_gov_api_key"):
            LOGGER.warning("DATA_GOV_API_KEY is not configured; using manual CSV fallback.")
        else:
            for commodity in config["collection"]["commodities"]:
                for state in config["collection"]["states"]:
                    try:
                        all_records.extend(fetch_data_gov_records(session, config, commodity, state))
                    except requests.RequestException as exc:
                        LOGGER.error("data.gov.in failed for %s/%s: %s", commodity, state, exc)
    finally:
        session.close()

    api_count = len(all_records)
    write_api_records(all_records, config)
    manual = load_manual_agmarknet_csvs(config["paths"]["raw_data"])
    if not manual.empty:
        manual.to_csv(Path(config["paths"]["raw_data"]) / "agmarknet_manual_merged.csv", index=False)

    weather_session = build_session(config)
    try:
        weather = fetch_open_meteo_weather(weather_session, config)
    except requests.RequestException as exc:
        LOGGER.error("Weather collection failed: %s", exc)
        weather = pd.DataFrame()
    finally:
        weather_session.close()

    return {"data_gov_records": api_count, "manual_csv_rows": len(manual), "weather_rows": len(weather)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    LOGGER.info("Collection summary: %s", collect(args.config))
