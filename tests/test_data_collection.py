from pathlib import Path

import pandas as pd

from src.data_collection import load_manual_agmarknet_csvs, safe_name


def test_safe_name_is_stable_and_filesystem_friendly() -> None:
    assert safe_name("Tur/Arhar") == "tur_arhar"
    assert safe_name("  Rajasthan  ") == "rajasthan"


def test_missing_manual_directory_returns_empty_frame(tmp_path: Path) -> None:
    result = load_manual_agmarknet_csvs(tmp_path / "missing")
    assert isinstance(result, pd.DataFrame)
    assert result.empty
