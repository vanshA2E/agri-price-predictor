"""Resumable downloader for the verified CEDA Agmarknet prices endpoint."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)
OUTPUT_COLUMNS = [
    "date", "commodity_id", "commodity", "state_id", "state", "district_id", "market_id", "level",
    "min_price", "max_price", "modal_price",
]


class FatalCEDAError(RuntimeError):
    """An API error that must stop the run immediately."""


class UnexpectedCEDAError(RuntimeError):
    """An undocumented API failure that must be surfaced to the user."""


@dataclass
class ChunkResult:
    rows: int
    min_date: str | None
    max_date: str | None
    markets: int
    truncated: bool
    failed: bool = False


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


def api_key() -> str:
    load_dotenv()
    key = os.getenv("CEDA_API_KEY", "").strip()
    if not key or key == "your_ceda_api_key":
        raise FatalCEDAError("CEDA_API_KEY is missing from .env.")
    return key


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def date_windows(start: date, end: date, window_days: int) -> Iterable[tuple[date, date]]:
    if window_days < 1:
        raise ValueError("window_days must be positive")
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=window_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def as_date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def parse_response(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse required price fields and tolerate aggregate rows without geography IDs."""
    records = payload.get("output", {}).get("data", [])
    if not isinstance(records, list):
        raise UnexpectedCEDAError("CEDA response output.data is not a list.")
    required = {"date", "commodity_id", "min_price", "max_price", "modal_price"}
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=[
            "date", "commodity_id", "census_state_id", "census_district_id", "market_id",
            "min_price", "max_price", "modal_price", "level",
        ])
    missing = required.difference(frame.columns)
    if missing:
        raise UnexpectedCEDAError(f"CEDA response is missing documented fields: {sorted(missing)}")
    for column in ("census_state_id", "census_district_id", "market_id"):
        if column not in frame.columns:
            frame[column] = pd.NA
    if "market_id" in records[0]:
        level = "market"
    elif "census_district_id" in records[0]:
        level = "district"
    else:
        level = "state"
    frame["level"] = level
    return frame


def response_dates(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if frame.empty:
        return None, None
    parsed = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.date.dropna()
    return (parsed.min().isoformat(), parsed.max().isoformat()) if not parsed.empty else (None, None)


def looks_truncated(frame: pd.DataFrame, requested_end: date, final_window: bool = False) -> bool:
    _, max_date = response_dates(frame)
    late = (
        not final_window
        and max_date is not None
        and (requested_end - date.fromisoformat(max_date)).days > 45
    )
    count = len(frame)
    round_count = count > 0 and count in {1000, 5000, 10000}
    return bool(late or round_count)


def truncation_warning(frame: pd.DataFrame, requested_end: date, final_window: bool = False) -> str | None:
    if not looks_truncated(frame, requested_end, final_window):
        return None
    _, max_date = response_dates(frame)
    if not final_window and max_date is not None and (requested_end - date.fromisoformat(max_date)).days > 45:
        return f"last date {max_date} is more than 45 days before requested end {requested_end.isoformat()}"
    return f"row count {len(frame)} looks like a capped round number"


def redact(text: str, key: str) -> str:
    return text.replace(key, "<redacted>")


def _header_int(response: requests.Response, name: str) -> int | None:
    try:
        return int(response.headers.get(name, ""))
    except (TypeError, ValueError):
        return None


def respect_rate_limit(response: requests.Response) -> None:
    """Honor documented rate headers after every response."""
    remaining = _header_int(response, "ratelimit-remaining")
    reset = _header_int(response, "ratelimit-reset")
    if remaining is not None and remaining <= 2 and reset is not None:
        wait_seconds = max(0, reset) + 5
        LOGGER.warning("Only %s CEDA requests remain; sleeping %s seconds.", remaining, wait_seconds)
        time.sleep(wait_seconds)


def wait_for_reset(response: requests.Response) -> None:
    reset = _header_int(response, "ratelimit-reset")
    if reset is None:
        raise UnexpectedCEDAError("CEDA returned 429 without ratelimit-reset header.")
    wait_seconds = max(0, reset) + 5
    LOGGER.warning("CEDA rate limit reached; sleeping %s seconds before retry.", wait_seconds)
    time.sleep(wait_seconds)


def request_chunk(
    session: requests.Session,
    config: dict[str, Any],
    body: dict[str, Any],
    key: str,
    request_counter: list[int] | None = None,
    raw_path: Path | None = None,
) -> tuple[requests.Response, pd.DataFrame, dict[str, Any]]:
    """Request a chunk and apply only the documented retry behavior."""
    ceda = config["ceda"]
    url = f"{ceda['base_url'].rstrip('/')}{ceda['endpoint']}"
    max_tries = int(ceda.get("max_server_retries", 5))
    server_try = 0
    while True:
        if request_counter is not None:
            request_counter[0] += 1
        response = session.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=int(ceda.get("timeout_seconds", 60)),
        )
        if raw_path is not None:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(response.content)
        remaining = _header_int(response, "ratelimit-remaining")
        reset = _header_int(response, "ratelimit-reset")
        LOGGER.debug("CEDA response status=%s remaining=%s reset=%s", response.status_code, remaining, reset)
        if response.status_code == 429:
            wait_for_reset(response)
            continue
        respect_rate_limit(response)
        if response.status_code == 500:
            server_try += 1
            if server_try >= max_tries:
                raise UnexpectedCEDAError(f"CEDA returned 500 after {max_tries} tries.")
            delay = 2 ** (server_try - 1)
            LOGGER.warning("CEDA returned 500; retry %s/%s in %s seconds.", server_try, max_tries, delay)
            time.sleep(delay)
            continue
        if response.status_code in (400, 401):
            raise FatalCEDAError(
                f"CEDA returned HTTP {response.status_code}: {redact(response.text, key)}"
            )
        if response.status_code != 200:
            raise UnexpectedCEDAError(
                f"CEDA returned undocumented HTTP {response.status_code}: {redact(response.text, key)}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise UnexpectedCEDAError("CEDA returned non-JSON success response.") from exc
        return response, parse_response(payload), payload


def body_variants(base: dict[str, Any]) -> list[dict[str, Any]]:
    """Return documented-compatible bodies from simplest to most explicit."""
    without_filters = {key: value for key, value in base.items() if key not in {"district_id", "market_id"}}
    empty_filters = {**without_filters, "district_id": [], "market_id": []}
    return [without_filters, empty_filters]


def probe_case(
    session: requests.Session,
    config: dict[str, Any],
    key: str,
    label: str,
    base_body: dict[str, Any],
    to_date: date,
    request_counter: list[int],
) -> dict[str, Any]:
    probe_dir = Path(config["paths"]["raw_data"]) / "ceda" / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for variant_index, body in enumerate(body_variants(base_body), start=1):
        raw_path = probe_dir / f"{label}_variant_{variant_index}.json"
        try:
            response, frame, payload = request_chunk(session, config, body, key, request_counter, raw_path)
            min_date, max_date = response_dates(frame)
            records = payload.get("output", {}).get("data", [])
            first_fields = sorted(records[0].keys()) if records else []
            rows_per_date = (
                pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d").value_counts().sort_index().to_dict()
                if not frame.empty else {}
            )
            warning = truncation_warning(frame, to_date)
            result = {
                "label": label, "status": response.status_code, "rows": len(frame),
                "min_date": min_date, "max_date": max_date,
                "distinct_markets": int(frame["market_id"].nunique()) if not frame.empty else 0,
                "distinct_dates": int(frame["date"].nunique()) if not frame.empty else 0,
                "rows_per_date": rows_per_date, "first_record_fields": first_fields,
                "first_three_records": records[:3], "response_bytes": len(response.content),
                "truncated": warning is not None, "truncation_warning": warning,
                "variant": variant_index,
            }
            print(f"\nProbe {label} (variant {variant_index})")
            print(json.dumps({key: value for key, value in result.items() if key != "first_three_records"}, indent=2, default=str))
            print("first_three_records=")
            print(json.dumps(records[:3], indent=2, default=str))
            LOGGER.info("Probe %s: %s", label, result)
            return result
        except FatalCEDAError as exc:
            if "HTTP 400" not in str(exc) or variant_index == len(body_variants(base_body)):
                raise
            last_error = exc
            LOGGER.warning("Probe %s variant %s rejected; trying the next documented body variant.", label, variant_index)
        except UnexpectedCEDAError:
            raise
    raise UnexpectedCEDAError(f"Probe {label} failed: {last_error}")


def run_probe(config: dict[str, Any]) -> list[dict[str, Any]]:
    key = api_key()
    session = requests.Session()
    request_counter = [0]
    try:
        cases = [
            ("onion_rajasthan_2024", {"commodity_id": 23, "state_id": 8, "from_date": "2024-01-01", "to_date": "2024-12-31"}),
            ("onion_rajasthan_2020_2024", {"commodity_id": 23, "state_id": 8, "from_date": "2020-01-01", "to_date": "2024-12-31"}),
            ("onion_national_2024", {"commodity_id": 23, "from_date": "2024-01-01", "to_date": "2024-12-31"}),
        ]
        results = [probe_case(session, config, key, label, body, date.fromisoformat(body["to_date"]), request_counter) for label, body in cases]
    finally:
        session.close()
    safe_window = int(config["ceda"].get("default_window_days", 365))
    long_probe = next(result for result in results if result["label"] == "onion_rajasthan_2020_2024")
    if long_probe["status"] == 200 and not long_probe["truncated"]:
        safe_window = 1827
    probe_path = Path(config["paths"]["raw_data"]) / "ceda" / "probe" / "probe_summary.json"
    first_probe = next(result for result in results if result["label"] == "onion_rajasthan_2024")
    probe_path.write_text(
        json.dumps(
            {"results": results, "safe_window_days": safe_window, "body_variant": first_probe["variant"], "requests_used": request_counter[0]},
            indent=2,
        ),
        encoding="utf-8",
    )
    return results


def selected_items(config: dict[str, Any], commodity: str | None, state: str | None) -> list[tuple[str, int, str, int]]:
    commodities = config["ceda"]["commodities"]
    states = config["ceda"]["states"]
    if commodity:
        if commodity not in commodities:
            raise ValueError(f"Unknown CEDA commodity: {commodity}")
        commodities = {commodity: commodities[commodity]}
    if state:
        if state not in states:
            raise ValueError(f"Unknown CEDA state: {state}")
        states = {state: states[state]}
    return [(commodity_name, int(commodity_id), state_name, int(state_id)) for commodity_name, commodity_id in commodities.items() for state_name, state_id in states.items()]


def normalized_frame(frame: pd.DataFrame, commodity: str, commodity_id: int, state: str, state_id: int) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    result["commodity_id"] = commodity_id
    result["commodity"] = commodity
    result["state_id"] = state_id
    result["state"] = state
    result["district_id"] = frame["census_district_id"]
    result["market_id"] = frame["market_id"]
    result["level"] = frame["level"]
    for column in ("min_price", "max_price", "modal_price"):
        result[column] = pd.to_numeric(frame[column], errors="coerce")
    return result[OUTPUT_COLUMNS].dropna(subset=["date"])


def chunk_path(output_dir: Path, commodity: str, state: str, start: date, end: date) -> Path:
    return output_dir / f"{safe_name(commodity)}__{safe_name(state)}__{start.isoformat()}__{end.isoformat()}.csv"


def summarize(
    files: list[Path], failed: list[str], truncated: list[str], requests_used: int, run_end: date
) -> None:
    print("\nCEDA download summary")
    print("file | rows | first_date | last_date | status")
    for path in sorted(files):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame(columns=["date"])
        first_date = last_date = None
        if not frame.empty and "date" in frame.columns:
            parsed = pd.to_datetime(frame["date"], errors="coerce").dropna()
            if not parsed.empty:
                first_date, last_date = parsed.min().date().isoformat(), parsed.max().date().isoformat()
        try:
            requested_end = date.fromisoformat(path.stem.rsplit("__", 1)[1])
        except (IndexError, ValueError):
            requested_end = run_end
        final_window = requested_end >= run_end
        warning = truncation_warning(frame, requested_end, final_window)
        status = warning or ("empty" if frame.empty else "ok")
        print(f"{path.name} | {len(frame)} | {first_date} | {last_date} | {status}")
    print(f"failed_chunks={len(failed)} truncated_chunks={len(truncated)} total_requests_used={requests_used}")
    if failed:
        print("failed:")
        print("\n".join(f"- {item}" for item in failed))
    if truncated:
        print("possibly_truncated:")
        print("\n".join(f"- {item}" for item in truncated))


def run_download(config: dict[str, Any], args: argparse.Namespace) -> None:
    key = api_key()
    end = date.fromisoformat(args.end) if args.end else date.today()
    start = as_date(args.start) if args.start else as_date(config["ceda"]["start_date"])
    output_dir = Path(config["paths"]["raw_data"]) / "ceda"
    output_dir.mkdir(parents=True, exist_ok=True)
    window_days = int(config["ceda"].get("default_window_days", 365))
    body_variant = 1
    probe_summary = output_dir / "probe" / "probe_summary.json"
    if probe_summary.exists() and not args.window_days:
        saved_probe = json.loads(probe_summary.read_text(encoding="utf-8"))
        window_days = int(saved_probe["safe_window_days"])
        body_variant = int(saved_probe.get("body_variant", 1))
    if args.window_days:
        window_days = args.window_days
    items = selected_items(config, args.commodity, args.state)
    session = requests.Session()
    files: list[Path] = []
    failed: list[str] = []
    truncated: list[str] = []
    request_counter = [0]
    try:
        for commodity, commodity_id, state, state_id in items:
            for chunk_start, chunk_end in date_windows(start, end, window_days):
                path = chunk_path(output_dir, commodity, state, chunk_start, chunk_end)
                if path.exists():
                    try:
                        existing = pd.read_csv(path)
                    except pd.errors.EmptyDataError:
                        existing = pd.DataFrame()
                    if not existing.empty:
                        LOGGER.info("Skipping existing chunk %s", path.name)
                        files.append(path)
                        continue
                    LOGGER.warning("Removing empty marker %s so the chunk can be retried.", path.name)
                    path.unlink()
                body = {
                    "commodity_id": commodity_id, "state_id": state_id,
                    "from_date": chunk_start.isoformat(), "to_date": chunk_end.isoformat(),
                }
                body = body_variants(body)[body_variant - 1]
                LOGGER.info("Downloading %s / %s: %s to %s", commodity, state, body["from_date"], body["to_date"])
                try:
                    _, raw, _ = request_chunk(session, config, body, key, request_counter)
                    if raw.empty:
                        message = (
                            f"{commodity}, {state}, {chunk_start.isoformat()} to {chunk_end.isoformat()}: "
                            "CEDA returned 0 records"
                        )
                        failed.append(message)
                        LOGGER.warning("Failed empty chunk: %s", message)
                        continue
                    normalized = normalized_frame(raw, commodity, commodity_id, state, state_id)
                    normalized.to_csv(path, index=False, columns=OUTPUT_COLUMNS)
                    files.append(path)
                    if looks_truncated(raw, chunk_end, chunk_end >= end):
                        truncated.append(path.name)
                    time.sleep(float(config["ceda"].get("request_delay_seconds", 1.0)))
                except UnexpectedCEDAError as exc:
                    failed.append(f"{path.name}: {exc}")
                    LOGGER.error("Failed chunk %s; continuing: %s", path.name, exc)
    finally:
        session.close()
    summarize(files, failed, truncated, request_counter[0], end)


def run_dry_run(config: dict[str, Any], args: argparse.Namespace) -> None:
    start = as_date(args.start) if args.start else as_date(config["ceda"]["start_date"])
    end = date.fromisoformat(args.end) if args.end else date.today()
    window_days = args.window_days or int(config["ceda"].get("default_window_days", 365))
    count = sum(1 for _ in date_windows(start, end, window_days)) * len(selected_items(config, args.commodity, args.state))
    print(f"planned_requests={count}")
    print(f"window_days={window_days}")
    print(f"estimated_hours_at_40_per_hour={count / 40:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--commodity")
    parser.add_argument("--state")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--window-days", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config)
    if args.dry_run:
        run_dry_run(config, args)
    elif args.probe:
        run_probe(config)
    else:
        run_download(config, args)


if __name__ == "__main__":
    main()