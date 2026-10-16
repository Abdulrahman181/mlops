from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main  # noqa: E402


def test_python_sources_parse() -> None:
    for path in ROOT.rglob("*.py"):
        if ".git" in path.parts:
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_missing_artifacts_are_reported_without_fallback(tmp_path: Path) -> None:
    missing = main.missing_artifacts(tmp_path)
    assert missing
    with pytest.raises(FileNotFoundError, match="Required model artifacts are missing"):
        main.load_artifacts(tmp_path)


def test_notebook_is_valid_json_and_has_no_stale_outputs() -> None:
    notebook = json.loads((ROOT / "Customer_Churn_Prediction.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] >= 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"notebook-cell-{index}")
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "/content/" not in source


class _LabelEncoder:
    classes_ = ["DSL", "Fiber optic", "No"]

    def transform(self, values):
        return [self.classes_.index(values[0])]


class _OrdinalEncoder:
    categories_ = [["Month-to-month", "One year", "Two year"]]

    def transform(self, values):
        return [[self.categories_[0].index(values[0][0])]]


class _RecordingScaler:
    def __init__(self):
        self.seen = None

    def transform(self, values):
        self.seen = values.copy()
        return values


def _valid_choices() -> dict[str, object]:
    return {
        "Gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.5,
        "TotalCharges": 900.0,
    }


def test_build_input_matches_notebook_order_and_scales_all_features() -> None:
    scaler = _RecordingScaler()
    artifacts = main.ArtifactBundle(
        encoders={
            "internet_service": _LabelEncoder(),
            "payment_method": type(
                "PaymentEncoder", (), {
                    "classes_": ["Bank transfer", "Credit card", "Electronic check"],
                    "transform": lambda self, values: [self.classes_.index(values[0])],
                }
            )(),
            "contract": _OrdinalEncoder(),
        },
        scaler=scaler,
        models={},
    )

    result = main.build_input(_valid_choices(), artifacts)

    assert result.shape == (1, 18)
    assert scaler.seen is not None
    assert scaler.seen[0].tolist() == [0, 1, 0, 12, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 2, 75.5, 900.0]


def test_build_input_rejects_invalid_binary_values() -> None:
    choices = _valid_choices()
    choices["Partner"] = "maybe"
    artifacts = main.ArtifactBundle(
        encoders={
            "internet_service": _LabelEncoder(),
            "payment_method": _LabelEncoder(),
            "contract": _OrdinalEncoder(),
        },
        scaler=_RecordingScaler(),
        models={},
    )
    with pytest.raises(ValueError, match="Expected Yes or No"):
        main.build_input(choices, artifacts)


def test_artifact_contract_rejects_wrong_feature_count() -> None:
    class WrongModel:
        n_features_in_ = 3

        def predict(self, values):
            return [0]

    with pytest.raises(ValueError, match="18-feature schema"):
        main._validate_artifact_contract(
            {
                "internet_service": _LabelEncoder(),
                "payment_method": _LabelEncoder(),
                "contract": _OrdinalEncoder(),
            },
            _RecordingScaler(),
            {"model": WrongModel()},
        )
