from argparse import Namespace
from pathlib import Path

import pandas as pd

from src import ceda_downloader
from src.data_cleaning import build_ceda_state_weekly, detect_stuck_prices


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {"output": {"type": "success", "message": "Data exists", "data": []}}
        self.headers = headers or {"ratelimit-remaining": "39", "ratelimit-reset": "3600"}
        self.content = b"{}"
        self.text = "{}"

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json, headers, timeout):
        self.calls.append(json)
        return FakeResponse(
            payload={
                "output": {
                    "type": "success",
                    "message": "Data exists",
                    "data": [{
                        "date": "2025-03-03T00:00:00.000Z",
                        "commodity_id": 1,
                        "census_state_id": 8,
                        "census_district_id": 104,
                        "market_id": 3149,
                        "min_price": 2864,
                        "max_price": 2887,
                        "modal_price": 2880,
                    }],
                }
            }
        )

    def close(self):
        pass


def test_parse_documented_response() -> None:
    frame = ceda_downloader.parse_response({
        "output": {
            "type": "success",
            "message": "Data exists",
            "data": [{
                "date": "2025-03-03T00:00:00.000Z",
                "commodity_id": 1,
                "census_state_id": 8,
                "census_district_id": 104,
                "market_id": 3149,
                "min_price": 2864,
                "max_price": 2887,
                "modal_price": 2880,
            }],
        }
    })
    assert frame.iloc[0]["modal_price"] == 2880
    assert frame.iloc[0]["market_id"] == 3149


def test_parse_state_level_response_without_geography_fields() -> None:
    frame = ceda_downloader.parse_response({
        "output": {
            "type": "success",
            "message": "Data exists",
            "data": [{
                "date": "2025-03-03T00:00:00.000Z",
                "commodity_id": 23,
                "min_price": 2864,
                "max_price": 2887,
                "modal_price": 2880,
            }],
        }
    })
    assert frame.iloc[0]["market_id"] is pd.NA
    assert frame.iloc[0]["census_district_id"] is pd.NA
    assert frame.iloc[0]["level"] == "state"


def test_probe_raw_body_is_saved_before_parse_error(tmp_path: Path) -> None:
    class BadSession:
        def post(self, url, json, headers, timeout):
            response = FakeResponse(payload={"output": {"data": [{"date": "2025-01-01"}]}})
            response.content = b'{"output":{"data":[{"date":"2025-01-01"}]}}'
            return response

    raw_path = tmp_path / "probe.json"
    config = {"ceda": {"base_url": "https://example.invalid", "endpoint": "/prices", "timeout_seconds": 1}}
    try:
        ceda_downloader.request_chunk(BadSession(), config, {}, "test-key", raw_path=raw_path)
    except ceda_downloader.UnexpectedCEDAError:
        pass
    else:
        raise AssertionError("Expected the incomplete payload to fail parsing")
    assert raw_path.read_bytes().startswith(b'{"output"')


def test_rate_limit_sleep_logic(monkeypatch) -> None:
    waits = []
    monkeypatch.setattr(ceda_downloader.time, "sleep", waits.append)
    response = FakeResponse(headers={"ratelimit-remaining": "2", "ratelimit-reset": "7"})
    ceda_downloader.respect_rate_limit(response)
    assert waits == [12]


def test_stuck_price_detection() -> None:
    frame = pd.DataFrame({
        "commodity": ["Onion"] * 7,
        "state": ["Rajasthan"] * 7,
        "market_id": [1] * 7,
        "date": pd.date_range("2025-01-01", periods=7),
        "min_price": [100] * 7,
        "max_price": [120] * 7,
        "modal_price": [110] * 7,
    })
    flags = detect_stuck_prices(frame)
    assert flags.all()


def test_resume_skips_existing_chunk(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "raw" / "ceda"
    output.mkdir(parents=True)
    existing = ceda_downloader.chunk_path(output, "Onion", "Rajasthan", pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-02").date())
    pd.DataFrame([{
        "date": "2024-01-01", "commodity_id": 23, "commodity": "Onion", "state_id": 8,
        "state": "Rajasthan", "district_id": pd.NA, "market_id": pd.NA, "level": "state",
        "min_price": 90, "max_price": 110, "modal_price": 100,
    }], columns=ceda_downloader.OUTPUT_COLUMNS).to_csv(existing, index=False)
    config = {
        "paths": {"raw_data": str(tmp_path / "raw")},
        "ceda": {
            "base_url": "https://example.invalid/v1",
            "endpoint": "/agmarknet/prices",
            "start_date": "2024-01-01",
            "default_window_days": 2,
            "request_delay_seconds": 0,
            "timeout_seconds": 1,
            "max_server_retries": 1,
            "commodities": {"Onion": 23},
            "states": {"Rajasthan": 8},
        },
    }
    fake_session = FakeSession()
    monkeypatch.setattr(ceda_downloader, "api_key", lambda: "test-key")
    monkeypatch.setattr(ceda_downloader.requests, "Session", lambda: fake_session)
    args = Namespace(commodity="Onion", state="Rajasthan", start="2024-01-01", end="2024-01-02", window_days=2)
    ceda_downloader.run_download(config, args)
    assert fake_session.calls == []
    assert "total_requests_used=0" in capsys.readouterr().out


def test_empty_response_is_failed_without_csv(tmp_path: Path, monkeypatch, capsys) -> None:
    class EmptySession:
        def post(self, url, json, headers, timeout):
            return FakeResponse(payload={"output": {"type": "success", "message": "No data", "data": []}})

        def close(self):
            pass

    config = {
        "paths": {"raw_data": str(tmp_path / "raw")},
        "ceda": {
            "base_url": "https://example.invalid/v1",
            "endpoint": "/agmarknet/prices",
            "start_date": "2024-01-01",
            "default_window_days": 2,
            "request_delay_seconds": 0,
            "timeout_seconds": 1,
            "max_server_retries": 1,
            "commodities": {"Onion": 23},
            "states": {"Rajasthan": 8},
        },
    }
    monkeypatch.setattr(ceda_downloader, "api_key", lambda: "test-key")
    monkeypatch.setattr(ceda_downloader.requests, "Session", EmptySession)
    args = Namespace(commodity="Onion", state="Rajasthan", start="2024-01-01", end="2024-01-02", window_days=2)
    ceda_downloader.run_download(config, args)
    output = tmp_path / "raw" / "ceda" / "onion__rajasthan__2024-01-01__2024-01-02.csv"
    assert not output.exists()
    assert "CEDA returned 0 records" in capsys.readouterr().out


def test_final_window_is_not_flagged_for_update_lag() -> None:
    frame = pd.DataFrame({"date": ["2024-10-01"], "modal_price": [100]})
    assert not ceda_downloader.looks_truncated(frame, pd.Timestamp("2026-10-29").date(), final_window=True)
    assert ceda_downloader.looks_truncated(frame, pd.Timestamp("2024-12-28").date(), final_window=False)


def test_state_level_filename_cleaning(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "ceda"
    raw_dir.mkdir(parents=True)
    frame = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "min_price": [90, 91], "max_price": [110, 111], "modal_price": [100, 101],
    })
    first = raw_dir / "onion__rajasthan__2015-01-01__2019-12-30.csv"
    second = raw_dir / "onion__rajasthan__2019-12-31__2024-12-28.csv"
    frame.to_csv(first, index=False)
    frame.to_csv(second, index=False)
    config = {
        "paths": {"interim_data": str(tmp_path / "interim"), "processed_data": str(tmp_path / "processed")},
        "ceda": {"commodities": {"Onion": 23}, "states": {"Rajasthan": 8}},
        "cleaning": {"min_history_weeks": 1, "max_series_missing_fraction": 0.40, "min_days_per_week": 1},
    }
    weekly, report = build_ceda_state_weekly([first, second], config)
    assert report["rows_before"] == 4
    assert report["rows_after"] == 2
    assert len(weekly) == 1
    assert (tmp_path / "processed" / "ceda_validation_report.csv").exists()
