"""Focused tests for the shared raw-input and inference contracts."""

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from churn_features import engineer_features, positive_class_probability, validate_input_frame


BASE_ROW = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": "171.0",
}


class StaticPipeline:
    classes_ = np.array([1, 0])

    def predict_proba(self, frame):
        return np.tile([[0.8, 0.2]], (len(frame), 1))


class FeatureContractTests(unittest.TestCase):
    def test_feature_engineering_is_deterministic_and_does_not_mutate_input(self):
        frame = pd.DataFrame([BASE_ROW])
        original = frame.copy(deep=True)
        engineered = engineer_features(frame)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(engineered.loc[0, "TotalServices"], 3)
        self.assertEqual(engineered.loc[0, "ChurnRiskScore"], 10)
        self.assertAlmostEqual(engineered.loc[0, "CLV"], 171.0)

    def test_blank_total_charges_is_allowed_but_bad_numeric_is_rejected(self):
        blank = pd.DataFrame([{**BASE_ROW, "TotalCharges": " "}])
        validated = validate_input_frame(blank)
        self.assertTrue(pd.isna(validated.loc[0, "TotalCharges"]))
        bad = pd.DataFrame([{**BASE_ROW, "tenure": "not-a-number"}])
        with self.assertRaises(ValueError):
            validate_input_frame(bad)

    def test_unknown_category_is_rejected_by_strict_validator(self):
        bad = pd.DataFrame([{**BASE_ROW, "Contract": "Unknown contract"}])
        with self.assertRaises(ValueError):
            validate_input_frame(bad)

    def test_positive_probability_uses_class_label_not_column_position(self):
        result = positive_class_probability(StaticPipeline(), pd.DataFrame([BASE_ROW]))
        np.testing.assert_allclose(result, [0.8])


if __name__ == "__main__":
    unittest.main()
