# Engineering Review — Olist PostgreSQL Ingestion

## Scope and evidence

This review covers every tracked Python, SQL, README, dependency/configuration, and
Notebook candidate in the repository at the time of review. The checkout contains one
Python implementation (`scripts/ingest.py`), one SQL validation script, two reference
PDFs, a README, and no Notebook, model artifact, API service, or local Olist CSV data.
The `archive/` directory is intentionally excluded by `.gitignore`.

No data, synthetic fallback, database result, model metric, or successful end-to-end run
has been invented or claimed. The historical row counts in the README remain clearly
labelled as unverified project notes.

## Findings and remediation

### Correctness and reliability

- Added `requirements.txt` with bounded compatible ranges for pandas, SQLAlchemy, and
  psycopg2-binary.
- Added `pyproject.toml` configuration for Ruff and pytest.
- Reworked ingestion around `pathlib`, a CLI, and explicit environment/CLI configuration.
- The script checks that the CSV directory and all nine required files exist before it
  creates a database engine. Missing data returns exit code `2` with every missing path.
- Added source-column checks for every table and rejected non-empty values that fail
  numeric or datetime conversion. Integer fields reject fractional values and use
  nullable pandas integer dtypes instead of silently truncating or inventing values.
- Added a database connection check and row-count verification for every loaded table.
- All table writes use one SQLAlchemy transaction. A failed load is rolled back rather
  than being reported as a successful partial refresh. The existing-table policy is
  explicit via `--if-exists` (`replace` by default, with `fail` and `append` options).
- Added unit tests for conversion, invalid values, missing input directories, and secure
  database URL resolution.

### Security and configuration

- Removed the hard-coded database password/URL from Python source. The URL must come
  from `OLIST_DATABASE_URL`, `DATABASE_URL`, or `--database-url`; `.env.example` is a
  non-secret template.
- Kept raw data, logs, caches, artifacts, and real environment files out of Git while
  explicitly allowing `.env.example` to be tracked.
- Generated SQL uses table names from the fixed internal map and quotes them before the
  row-count query; no user-provided identifier is interpolated.
- There is no HTTP/API service or authentication surface to audit in this repository.

### SQL validation

- Added `ON_ERROR_STOP` and a required-public-table preflight to `sql/validation.sql`.
  Validation now stops with an explicit missing-table error instead of continuing into
  a cascade of relation errors. Existing relationship, duplicate-review, and delivery
  target checks are retained.

### Leakage, splitting, metrics, and ML readiness

- There is no model, feature pipeline, train/test split, evaluation code, or metric
  output in this Task 1 repository, so no split or metric execution was possible.
- The README continues to document the key leakage rule: actual delivery timestamps
  and post-outcome reviews may construct an analysis target but must not be model inputs
  when predicting at order-purchase time. A future ML task must define prediction time,
  aggregate one row per order, split chronologically (or otherwise justify the split),
  and fit preprocessing only on training data.
- No data leakage can be proven absent from a future pipeline because that pipeline is
  not present; this is an explicit follow-up requirement, not a claimed result.

## Validation performed

The following checks were run in the sandbox after installing the repository's declared
Python dependencies:

- `python3 -m compileall -q scripts tests` — passed.
- AST parsing of all Python files — passed.
- `ruff check .` — passed.
- `ruff format --check .` — passed.
- `pytest -q` — passed for the unit test suite (5 tests).
- `python scripts/ingest.py --database-url ... --csv-dir /tmp/olist-data-does-not-exist`
  — confirmed the fail-fast missing-data path (exit code `2`).

A PostgreSQL server and `psql` client were not available in the sandbox, and all nine
CSV files are absent. Therefore database connectivity, actual ingestion, SQL execution,
row counts, join integrity, and transaction behavior were not integration-tested here.
No full execution is claimed.

## Files changed

- `.env.example` — non-secret database URL template.
- `.gitignore` — cache/artifact exclusions and `.env.example` exception.
- `ENGINEERING_REVIEW.md` — this review.
- `README.md` — accurate status, setup, failure semantics, and runtime limitations.
- `pyproject.toml` — Ruff and pytest configuration.
- `requirements.txt` — runtime dependency ranges.
- `scripts/__init__.py` — package marker.
- `scripts/ingest.py` — secure, validated, transactional ingestion CLI.
- `sql/validation.sql` — fail-fast validation preflight.
- `tests/test_ingest.py` — dependency-light unit tests.

## Reproduction with real prerequisites

1. Install dependencies: `python -m pip install -r requirements.txt`.
2. Download the original Olist CSV files into a local directory (not Git).
3. Start PostgreSQL and export `OLIST_DATABASE_URL` without committing it.
4. Run `python scripts/ingest.py --csv-dir /path/to/archive`.
5. Run `psql "$OLIST_DATABASE_URL" -f sql/validation.sql` (or the repository's Docker
   command after adapting it to the local PostgreSQL setup).

The audit commit is local only; no changes were pushed to GitHub.

## Second Senior Review (2026-10-16)

A second manual pass rechecked ingestion contracts, transaction boundaries, SQL validation, configuration hygiene, and the unit-test suite. No new correctness defect was found in the repaired dependency-light workflow. Missing Olist CSVs and PostgreSQL services remain explicit runtime prerequisites.

Validation: `pytest -q` (5 passed), `compileall`, and `git diff --check` passed.
