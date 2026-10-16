"""Canonical input schema validation and deterministic churn feature engineering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NUMERIC_COLUMNS = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL_VALUES = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


def validate_input_frame(
    frame: pd.DataFrame,
    *,
    require_target: bool = False,
    strict_categories: bool = True,
) -> pd.DataFrame:
    """Validate and normalize raw IBM Telco rows without fitting anything.

    ``TotalCharges`` may be blank in the source dataset and is intentionally left
    as NaN for the training pipeline's training-only imputer. All other numeric
    fields must be parseable and finite. The returned frame is a copy.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    required = set(RAW_COLUMNS)
    if require_target:
        required.add("Churn")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Input contains duplicate columns: {duplicates}")

    result = frame.copy()
    for column in NUMERIC_COLUMNS:
        original = result[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = converted.isna() & original.notna() & original.astype(str).str.strip().ne("")
        if column != "TotalCharges" and invalid.any():
            raise ValueError(f"Column {column!r} contains non-numeric values.")
        if column == "TotalCharges" and invalid.any():
            raise ValueError("Column 'TotalCharges' contains non-numeric values.")
        result[column] = converted
        if converted.notna().any() and not np.isfinite(converted.dropna()).all():
            raise ValueError(f"Column {column!r} contains non-finite values.")

    if not bool(result["SeniorCitizen"].dropna().isin([0, 1]).all()):
        raise ValueError("SeniorCitizen must contain only 0 or 1.")
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        if (result[column].dropna() < 0).any():
            raise ValueError(f"Column {column!r} cannot contain negative values.")

    if strict_categories:
        for column, allowed in CATEGORICAL_VALUES.items():
            values = set(result[column].dropna().unique())
            unexpected = sorted(values - allowed)
            if unexpected:
                raise ValueError(
                    f"Column {column!r} contains unsupported values: {unexpected}"
                )
    if require_target:
        targets = set(result["Churn"].dropna().unique())
        unexpected_targets = sorted(targets - {"Yes", "No"})
        if unexpected_targets or result["Churn"].isna().any():
            raise ValueError("Churn must contain only non-null 'Yes' or 'No' values.")
    return result


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the deterministic features used by both training and inference."""
    result = validate_input_frame(frame, strict_categories=True)
    result["AvgChargePerMonth"] = result["TotalCharges"] / (result["tenure"] + 1)
    result["CLV"] = result["MonthlyCharges"] * result["tenure"]
    result["ChargeIncreaseRatio"] = result["MonthlyCharges"] / np.maximum(
        result["AvgChargePerMonth"], 1
    )
    result["IsNewCustomer"] = (result["tenure"] < 12).astype(int)
    result["IsMidTerm"] = ((result["tenure"] >= 12) & (result["tenure"] < 36)).astype(int)
    result["IsLoyalCustomer"] = (result["tenure"] >= 36).astype(int)
    result["TenureSquared"] = result["tenure"] ** 2
    service_values = result[SERVICE_COLUMNS].apply(
        lambda column: column.map(
            {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
        )
    )
    result["TotalServices"] = service_values.sum(axis=1, min_count=len(SERVICE_COLUMNS))
    result["ServiceAdoptionRate"] = result["TotalServices"] / len(SERVICE_COLUMNS)
    result["HighMonthlyCharge"] = (result["MonthlyCharges"] > 70).astype(int)
    result["NoSecurityBundle"] = (
        (result["OnlineSecurity"] == "No") & (result["DeviceProtection"] == "No")
    ).astype(int)
    result["MonthToMonthContract"] = (result["Contract"] == "Month-to-month").astype(int)
    result["ElectronicCheckPay"] = (result["PaymentMethod"] == "Electronic check").astype(int)
    result["ChurnRiskScore"] = (
        result["IsNewCustomer"] * 3
        + result["MonthToMonthContract"] * 3
        + result["ElectronicCheckPay"] * 2
        + result["NoSecurityBundle"]
        + result["HighMonthlyCharge"]
    )
    return result


def positive_class_probability(pipeline: Any, frame: pd.DataFrame) -> np.ndarray:
    """Return class-1 probabilities while refusing ambiguous class ordering."""
    predict_proba = getattr(pipeline, "predict_proba", None)
    classes = getattr(pipeline, "classes_", None)
    if not callable(predict_proba) or classes is None:
        raise ValueError("Loaded pipeline must expose predict_proba and classes_.")
    classes = list(classes)
    if 1 not in classes:
        raise ValueError("Loaded pipeline does not expose a positive churn class (1).")
    probabilities = np.asarray(predict_proba(frame), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Pipeline returned probabilities with an invalid shape.")
    probability = probabilities[:, classes.index(1)]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Pipeline returned invalid churn probabilities.")
    return probability
