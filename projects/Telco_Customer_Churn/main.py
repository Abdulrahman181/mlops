"""Streamlit application for the Telco customer-churn model artifacts.

The repository intentionally does not contain trained models or encoders.  The
application therefore validates the artifact directory before rendering a
prediction form and never silently creates or downloads replacements.
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Keep static checks and the artifact validator usable without Streamlit.
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - exercised in deployment only.
    st = None  # type: ignore[assignment]


ARTIFACT_ROOT = Path("Deployment_Churn")
ENCODER_FILES = {
    "internet_service": Path("Encoding/InternetService_le.pkl"),
    "payment_method": Path("Encoding/PaymentMethod_le.pkl"),
    "contract": Path("Encoding/Contract_oe.pkl"),
}
SCALER_FILE = Path("Scaling/scaler.pkl")
MODEL_FILES = {
    "Decision Tree": Path("Models/Decision Tree.pkl"),
    "Logistic Regression": Path("Models/LogisticRegression.pkl"),
    "SVC": Path("Models/SVC.pkl"),
    "KNN": Path("Models/KNN.pkl"),
    "Random Forest": Path("Models/RandomForest.pkl"),
    "Gradient Boosting": Path("Models/GradientBoosting.pkl"),
    "XGBoost": Path("Models/XGBoost.pkl"),
    "AdaBoost": Path("Models/AdaBoost.pkl"),
    "Stacking": Path("Models/Stacking.pkl"),
}

# This is the exact feature order used by the training notebook after it drops
# ``gender``.  The notebook fits StandardScaler on the complete 18-column
# matrix, so inference must build this order and transform all 18 columns.
FEATURE_NAMES = (
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
)
INPUT_FIELDS = set(FEATURE_NAMES) | {"Gender"}


class ArtifactBundle(dict[str, Any]):
    """Loaded encoders, scaler, and models with a predictable mapping."""


def required_artifacts(root: Path) -> list[Path]:
    """Return every file needed by the dashboard, relative to ``root``."""
    relative = list(ENCODER_FILES.values()) + [SCALER_FILE] + list(MODEL_FILES.values())
    return [root / ARTIFACT_ROOT / path for path in relative]


def missing_artifacts(root: Path | None = None) -> list[Path]:
    """Return missing artifact paths without attempting to load any pickle."""
    project_root = (root or Path(__file__).resolve().parent).resolve()
    return [path for path in required_artifacts(project_root) if not path.is_file()]


def _load_pickle(path: Path) -> Any:
    """Load a repository-local pickle using a context manager.

    Pickle is executable serialization.  Only load files supplied by a trusted
    training pipeline; this function deliberately performs no network fetch.
    """
    with path.open("rb") as handle:
        return pickle.load(handle)


def _validate_artifact_contract(
    encoders: dict[str, Any], scaler: Any, models: dict[str, Any]
) -> None:
    """Reject artifacts that cannot consume the notebook's feature schema."""
    expected = len(FEATURE_NAMES)
    scaler_features = getattr(scaler, "n_features_in_", expected)
    if scaler_features != expected or not callable(getattr(scaler, "transform", None)):
        raise ValueError(f"Scaler must transform exactly {expected} features")
    for name, model in models.items():
        model_features = getattr(model, "n_features_in_", expected)
        if model_features != expected or not callable(getattr(model, "predict", None)):
            raise ValueError(f"Model {name!r} is incompatible with the {expected}-feature schema")
    for name in ("internet_service", "payment_method", "contract"):
        encoder = encoders[name]
        if not callable(getattr(encoder, "transform", None)):
            raise ValueError(f"Encoder {name!r} does not provide transform()")


def _encoder_values(encoder: Any) -> list[str]:
    """Return the learned categories for a LabelEncoder or OrdinalEncoder."""
    if hasattr(encoder, "classes_"):
        values = encoder.classes_
    elif hasattr(encoder, "categories_") and len(encoder.categories_) == 1:
        values = encoder.categories_[0]
    else:
        raise ValueError("Categorical encoder has no single learned category list")
    return [str(value) for value in values]


def load_artifacts(root: Path | None = None) -> ArtifactBundle:
    """Load all model artifacts or raise an actionable error.

    No fallback model or synthetic artifact is created when files are absent.
    """
    project_root = (root or Path(__file__).resolve().parent).resolve()
    missing = missing_artifacts(project_root)
    if missing:
        listed = "\n".join(f"- {path.relative_to(project_root)}" for path in missing)
        raise FileNotFoundError(
            "Required model artifacts are missing. Train/export the model first "
            "and place the trusted files under Deployment_Churn/; missing files:\n"
            f"{listed}"
        )

    artifact_root = project_root / ARTIFACT_ROOT
    encoders = {
        name: _load_pickle(artifact_root / relative)
        for name, relative in ENCODER_FILES.items()
    }
    scaler = _load_pickle(artifact_root / SCALER_FILE)
    models = {
        name: _load_pickle(artifact_root / relative)
        for name, relative in MODEL_FILES.items()
    }
    _validate_artifact_contract(encoders, scaler, models)
    return ArtifactBundle(encoders=encoders, scaler=scaler, models=models)


def _yes_no(value: str) -> int:
    if value not in {"Yes", "No"}:
        raise ValueError(f"Expected Yes or No, got {value!r}")
    return int(value == "Yes")


def _encode_category(encoder: Any, value: str, field: str) -> float:
    """Encode one categorical value using either sklearn encoder shape."""
    try:
        if hasattr(encoder, "classes_"):
            encoded = encoder.transform([value])
        else:
            encoded = encoder.transform([[value]])
        result = np.asarray(encoded, dtype=float).reshape(-1)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(f"Unknown {field} value {value!r}") from error
    if result.size != 1 or not np.isfinite(result[0]):
        raise ValueError(f"Encoder for {field} returned an invalid value")
    return float(result[0])


def build_input(choices: dict[str, Any], artifacts: ArtifactBundle) -> np.ndarray:
    """Encode one UI record in the training feature order."""
    missing = INPUT_FIELDS - choices.keys()
    if missing:
        raise ValueError(f"Missing input fields: {', '.join(sorted(missing))}")
    encoders = artifacts["encoders"]
    # Gender is intentionally not included: the notebook drops it before fit.
    encoded = [
        _yes_no(str(choices["SeniorCitizen"])),
        _yes_no(str(choices["Partner"])),
        _yes_no(str(choices["Dependents"])),
        choices["tenure"],
        _yes_no(str(choices["PhoneService"])),
        _yes_no(str(choices["MultipleLines"])),
        _encode_category(
            encoders["internet_service"], choices["InternetService"], "InternetService"
        ),
        _yes_no(str(choices["OnlineSecurity"])),
        _yes_no(str(choices["OnlineBackup"])),
        _yes_no(str(choices["DeviceProtection"])),
        _yes_no(str(choices["TechSupport"])),
        _yes_no(str(choices["StreamingTV"])),
        _yes_no(str(choices["StreamingMovies"])),
        _encode_category(encoders["contract"], choices["Contract"], "Contract"),
        _yes_no(str(choices["PaperlessBilling"])),
        _encode_category(encoders["payment_method"], choices["PaymentMethod"], "PaymentMethod"),
    ]
    encoded.extend([choices["MonthlyCharges"], choices["TotalCharges"]])
    try:
        raw = np.asarray([encoded], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Numeric inputs must be finite numbers") from error
    if raw.shape != (1, len(FEATURE_NAMES)) or not np.isfinite(raw).all():
        raise ValueError("Numeric inputs must be finite and match the training schema")
    if (raw[0, 3] < 0 or raw[0, 3] > 120 or raw[0, 16] < 0 or raw[0, 17] < 0):
        raise ValueError("Tenure must be between 0 and 120; charges cannot be negative")
    return np.asarray(artifacts["scaler"].transform(raw), dtype=float)


def predict_churn(models: dict[str, Any], input_data: np.ndarray) -> dict[str, str]:
    """Return one deterministic class label per model."""
    if input_data.ndim != 2 or input_data.shape[0] != 1:
        raise ValueError("input_data must be a 2-D array containing exactly one record")
    predictions: dict[str, str] = {}
    for name, model in models.items():
        prediction = int(model.predict(input_data)[0])
        if prediction not in {0, 1}:
            raise ValueError(f"Model {name!r} returned unsupported class {prediction!r}")
        predictions[name] = "Churn" if prediction == 1 else "Not Churn"
    return predictions


def save_user_data(choices: dict[str, Any], prediction: str) -> None:
    """Store only the current session's submitted values in Streamlit state."""
    if st is None:
        return
    record = {**choices, "Prediction": prediction}
    st.session_state.setdefault("user_data_list", []).append(record)


def display_predictions(predictions: dict[str, str]) -> None:
    """Display individual predictions and a deterministic majority status."""
    if st is None:
        return
    st.subheader("Prediction results")
    st.dataframe(
        pd.DataFrame(
            [{"Model": model, "Prediction": result} for model, result in predictions.items()]
        ),
        hide_index=True,
        use_container_width=True,
    )
    majority = Counter(predictions.values()).most_common(1)[0][0]
    st.success(f"Customer status: {majority}")


def _render_form(artifacts: ArtifactBundle) -> dict[str, Any]:
    """Render the validated input form and return its values."""
    if st is None:  # pragma: no cover
        raise RuntimeError("Streamlit is required to render the dashboard")
    encoders = artifacts["encoders"]
    with st.form("customer_attributes"):
        col1, col2, col3 = st.columns(3)
        with col1:
            senior = st.selectbox("Senior Citizen", ["Yes", "No"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=120, value=0, step=1
            )
            phone = st.selectbox("Phone Service", ["Yes", "No"])
        with col2:
            multiple = st.selectbox("Multiple Lines", ["Yes", "No"])
            internet = st.selectbox(
                "Internet Service", _encoder_values(encoders["internet_service"])
            )
            security = st.selectbox("Online Security", ["Yes", "No"])
            backup = st.selectbox("Online Backup", ["Yes", "No"])
            device = st.selectbox("Device Protection", ["Yes", "No"])
            support = st.selectbox("Tech Support", ["Yes", "No"])
            tv = st.selectbox("Streaming TV", ["Yes", "No"])
        with col3:
            movies = st.selectbox("Streaming Movies", ["Yes", "No"])
            contract = st.selectbox("Contract", _encoder_values(encoders["contract"]))
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", _encoder_values(encoders["payment_method"]))
            monthly = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
            total = st.number_input("Total Charges", min_value=0.0, value=0.0)
        submitted = st.form_submit_button("Predict churn")
    if not submitted:
        return {}
    return {
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }


def main() -> None:
    """Start the dashboard, stopping cleanly when prerequisites are absent."""
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install requirements.txt and run: "
            "streamlit run main.py"
        )
    st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
    st.title("Telecom Customer Churn Prediction")
    st.caption("Predictions use only trusted, locally supplied model artifacts.")
    try:
        artifacts = load_artifacts()
    except (FileNotFoundError, OSError, ValueError, pickle.UnpicklingError) as error:
        st.error(str(error))
        st.info("No prediction was run because the required artifacts are unavailable or invalid.")
        return

    choices = _render_form(artifacts)
    if choices:
        try:
            predictions = predict_churn(artifacts["models"], build_input(choices, artifacts))
        except (KeyError, ValueError, TypeError) as error:
            st.error(f"Input could not be encoded with the supplied artifacts: {error}")
            return
        save_user_data(choices, Counter(predictions.values()).most_common(1)[0][0])
        display_predictions(predictions)

    if st.button("Show submitted records"):
        records = st.session_state.get("user_data_list", [])
        st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
