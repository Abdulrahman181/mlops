"""Load the Olist CSV files into PostgreSQL.

The script deliberately does not download data or create synthetic fallbacks.  It fails
before connecting when the expected CSV files are absent, and all table loads run in one
transaction so a partial refresh is not silently presented as a successful ingestion.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

try:
    import pandas as pd
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Connection
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.types import BigInteger, DateTime, Float, Text
except ImportError as exc:  # pragma: no cover - exercised when dependencies are absent
    raise SystemExit(
        "Missing Python dependencies. Install them with: python -m pip install -r requirements.txt"
    ) from exc


LOGGER = logging.getLogger("olist.ingest")
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV_DIR = BASE_DIR / "archive"

TABLE_MAP: dict[str, str] = {
    "olist_customers_dataset.csv": "customers",
    "olist_geolocation_dataset.csv": "geolocation",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_orders_dataset.csv": "orders",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "product_category_translation",
}

EXPECTED_COLUMNS: dict[str, frozenset[str]] = {
    "customers": frozenset(
        {
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
        }
    ),
    "geolocation": frozenset(
        {
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
            "geolocation_city",
            "geolocation_state",
        }
    ),
    "order_items": frozenset(
        {
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "shipping_limit_date",
            "price",
            "freight_value",
        }
    ),
    "order_payments": frozenset(
        {"order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"}
    ),
    "order_reviews": frozenset(
        {
            "review_id",
            "order_id",
            "review_score",
            "review_comment_title",
            "review_comment_message",
            "review_creation_date",
            "review_answer_timestamp",
        }
    ),
    "orders": frozenset(
        {
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        }
    ),
    "products": frozenset(
        {
            "product_id",
            "product_category_name",
            "product_name_lenght",
            "product_description_lenght",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        }
    ),
    "sellers": frozenset({"seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"}),
    "product_category_translation": frozenset(
        {"product_category_name", "product_category_name_english"}
    ),
}

TEXT_COLUMNS = frozenset(
    {
        "customer_id",
        "customer_unique_id",
        "customer_city",
        "customer_state",
        "geolocation_city",
        "geolocation_state",
        "order_id",
        "product_id",
        "seller_id",
        "order_status",
        "payment_type",
        "review_id",
        "review_comment_title",
        "review_comment_message",
        "product_category_name",
        "product_category_name_english",
        "seller_city",
        "seller_state",
    }
)
INTEGER_COLUMNS = frozenset(
    {
        "customer_zip_code_prefix",
        "geolocation_zip_code_prefix",
        "seller_zip_code_prefix",
        "order_item_id",
        "payment_sequential",
        "payment_installments",
        "review_score",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    }
)
FLOAT_COLUMNS = frozenset(
    {"geolocation_lat", "geolocation_lng", "price", "freight_value", "payment_value"}
)
DATETIME_COLUMNS = frozenset(
    {
        "shipping_limit_date",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "review_creation_date",
        "review_answer_timestamp",
    }
)


def get_database_url(explicit_url: str | None = None) -> str:
    """Return a URL supplied by the caller, never a password embedded in source code."""
    value = explicit_url or os.getenv("OLIST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not value or not value.strip():
        raise ValueError(
            "Database URL is not configured. Set OLIST_DATABASE_URL (or DATABASE_URL), "
            "for example via .env.example."
        )
    return value.strip()


def _coerce_with_validation(
    series: pd.Series, converter: Callable[..., pd.Series], type_name: str
) -> pd.Series:
    """Coerce values while rejecting non-empty values that become null."""
    converted = converter(series, errors="coerce")
    non_empty = series.notna() & series.astype("string").str.strip().ne("")
    invalid_count = int((non_empty & converted.isna()).sum())
    if invalid_count:
        raise ValueError(f"Invalid {type_name} values in column '{series.name}': {invalid_count}")
    return converted


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known columns without mutating the input or inventing replacement values."""
    result = df.copy()
    for column in result.columns:
        if column in DATETIME_COLUMNS:
            result[column] = _coerce_with_validation(
                result[column],
                lambda values, **kwargs: pd.to_datetime(values, format="mixed", **kwargs),
                "datetime",
            )
        elif column in INTEGER_COLUMNS:
            converted = _coerce_with_validation(result[column], pd.to_numeric, "integer")
            fractional = converted.notna() & converted.mod(1).ne(0)
            if fractional.any():
                raise ValueError(
                    f"Invalid integer values in column '{column}': {int(fractional.sum())}"
                )
            result[column] = converted.astype("Int64")
        elif column in FLOAT_COLUMNS:
            result[column] = _coerce_with_validation(result[column], pd.to_numeric, "numeric")
        elif column in TEXT_COLUMNS:
            result[column] = result[column].astype("string")
    return result


def get_sql_types(columns: object) -> dict[str, object]:
    """Map known source fields to explicit SQLAlchemy types; keep unknown fields as text."""
    dtype: dict[str, object] = {}
    for column in columns:
        if column in DATETIME_COLUMNS:
            dtype[column] = DateTime()
        elif column in INTEGER_COLUMNS:
            dtype[column] = BigInteger()
        elif column in FLOAT_COLUMNS:
            dtype[column] = Float()
        else:
            dtype[column] = Text()
    return dtype


def validate_source_files(csv_dir: Path) -> None:
    """Fail early with every missing input rather than creating empty tables."""
    if not csv_dir.is_dir():
        raise FileNotFoundError(
            f"CSV directory does not exist: {csv_dir}. Download the Olist CSV archive "
            "into this directory or pass --csv-dir."
        )
    missing = [filename for filename in TABLE_MAP if not (csv_dir / filename).is_file()]
    if missing:
        formatted = "\n".join(f"  - {csv_dir / filename}" for filename in missing)
        raise FileNotFoundError(
            "Required Olist CSV files are missing; no tables were loaded:\n" + formatted
        )


def test_connection(connection: Connection) -> None:
    """Check the database before beginning the write transaction."""
    version = connection.execute(text("SELECT version();")).scalar_one()
    LOGGER.info("Database connection successful: %s", version)


def _validate_columns(df: pd.DataFrame, table_name: str) -> None:
    missing = EXPECTED_COLUMNS[table_name] - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV schema for '{table_name}' is missing columns: {', '.join(sorted(missing))}"
        )


def _quoted_table_name(table_name: str) -> str:
    # Table names come only from TABLE_MAP, but quoting keeps the generated SQL unambiguous.
    return '"' + table_name.replace('"', '""') + '"'


def load_table(
    connection: Connection, csv_dir: Path, filename: str, table_name: str, if_exists: str
) -> int:
    """Read, validate, and load one table, returning the verified database row count."""
    file_path = csv_dir / filename
    LOGGER.info("Loading %s into %s", file_path, table_name)
    df = pd.read_csv(file_path, low_memory=False)
    _validate_columns(df, table_name)
    prepared = prepare_dataframe(df)
    prepared.to_sql(
        table_name,
        connection,
        schema="public",
        if_exists=if_exists,
        index=False,
        dtype=get_sql_types(prepared.columns),
        chunksize=5_000,
        method="multi",
    )
    count = int(
        connection.execute(
            text(f"SELECT COUNT(*) FROM {_quoted_table_name(table_name)}")
        ).scalar_one()
    )
    if count != len(prepared):
        raise RuntimeError(
            f"Row count mismatch for {table_name}: CSV={len(prepared):,}, DB={count:,}"
        )
    LOGGER.info("%s: %s rows loaded", table_name, f"{count:,}")
    return count


def ingest(csv_dir: Path, database_url: str, if_exists: str = "replace") -> None:
    """Load all tables atomically after validating every source path."""
    validate_source_files(csv_dir)
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            test_connection(connection)
        with engine.begin() as connection:
            for filename, table_name in TABLE_MAP.items():
                load_table(connection, csv_dir, filename, table_name, if_exists)
    finally:
        engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path(os.getenv("OLIST_CSV_DIR", DEFAULT_CSV_DIR)),
        help="Directory containing the nine Olist CSV files (default: archive/).",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy URL; otherwise OLIST_DATABASE_URL or DATABASE_URL is required.",
    )
    parser.add_argument(
        "--if-exists",
        choices=("replace", "fail", "append"),
        default="replace",
        help="How to handle existing tables (default: replace, within one transaction).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        ingest(args.csv_dir.resolve(), get_database_url(args.database_url), args.if_exists)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 2
    except (ValueError, OSError, SQLAlchemyError, RuntimeError) as exc:
        LOGGER.error("Ingestion failed; no successful completion is claimed: %s", exc)
        return 1
    LOGGER.info("All tables loaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
