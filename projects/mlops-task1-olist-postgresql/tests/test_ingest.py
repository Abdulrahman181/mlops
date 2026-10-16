from pathlib import Path

import pandas as pd
import pytest

from scripts.ingest import get_database_url, prepare_dataframe, validate_source_files


def test_prepare_dataframe_converts_known_types_without_changing_row_count() -> None:
    source = pd.DataFrame(
        {
            "order_item_id": ["1", "", None],
            "price": ["12.50", "3", None],
            "order_purchase_timestamp": ["2017-01-01 10:00:00", "", None],
            "order_id": ["a", "b", None],
        }
    )

    result = prepare_dataframe(source)

    assert len(result) == len(source)
    assert str(result["order_item_id"].dtype) == "Int64"
    assert str(result["price"].dtype) == "float64"
    assert pd.api.types.is_datetime64_any_dtype(result["order_purchase_timestamp"])
    assert str(result["order_id"].dtype) == "string"
    assert source["order_item_id"].iloc[0] == "1"  # input is not mutated


def test_prepare_dataframe_rejects_invalid_numeric_values() -> None:
    with pytest.raises(ValueError, match="Invalid integer values"):
        prepare_dataframe(pd.DataFrame({"review_score": ["not-a-number"]}))


def test_validate_source_files_reports_directory_and_all_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        validate_source_files(tmp_path / "archive")

    message = str(error.value)
    assert "CSV directory does not exist" in message
    assert "archive" in message


def test_get_database_url_requires_explicit_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLIST_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValueError, match="OLIST_DATABASE_URL"):
        get_database_url()


def test_get_database_url_prefers_explicit_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLIST_DATABASE_URL", "postgresql+psycopg2://from-env")

    assert get_database_url("postgresql+psycopg2://explicit") == "postgresql+psycopg2://explicit"
