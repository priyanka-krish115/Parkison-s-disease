"""
Inference Module for Parkinson's Disease Prediction.

This module exposes the `ParkinsonsPredictor` class for predicting Parkinson's Disease likelihood
from acoustic voice measurement inputs.

NOTE: This system is designed solely as a clinical decision-support screening aid and is NOT a diagnostic
replacement for a clinical evaluation by a physician or neurologist.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List, Tuple
import joblib

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ParkinsonsPredictor:
    """
    Inference predictor class loading trained model, scaler, and feature ordering artifacts.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "best_model.joblib")
        self.scaler_path = os.path.join(models_dir, "scaler.joblib")
        self.features_path = os.path.join(models_dir, "feature_names.joblib")
        self.metadata_path = os.path.join(models_dir, "model_metadata.json")

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model, scaler, feature list, and metadata."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model missing at '{self.model_path}'. Please run training first.")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Fitted scaler missing at '{self.scaler_path}'.")
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Feature name registry missing at '{self.features_path}'.")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.required_features: List[str] = joblib.load(self.features_path)

        self.model_name = "Trained ML Model"
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                meta = json.load(f)
                self.model_name = meta.get("best_model_name", self.model_name)

        logger.info(f"Successfully loaded '{self.model_name}' predictor with {len(self.required_features)} required features.")

    @staticmethod
    def classify_risk_band(probability: float) -> Tuple[str, str]:
        """
        Classifies Parkinson's risk probability into clinical risk bands.

        Risk Bands:
          - Low Risk (<0.35): Standard routine monitoring.
          - Moderate Risk (0.35 - 0.65): Consider clinical follow-up & re-screening.
          - High Risk (>=0.65): Recommend full clinical evaluation by a specialist.
        """
        if probability < 0.35:
            return "Low Risk", "Standard routine monitoring."
        elif probability < 0.65:
            return "Moderate Risk", "Consider follow-up vocal & motor screening."
        else:
            return "High Risk", "Recommend clinical evaluation by a specialist/neurologist."

    def validate_features(self, sample_dict: Dict[str, float]) -> np.ndarray:
        """
        Validates feature presence in patient dict and returns ordered feature array.
        """
        missing = [feat for feat in self.required_features if feat not in sample_dict]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required feature(s) for inference:\n"
                f"Missing: {missing}\nRequired: {self.required_features}"
            )
        
        ordered_values = [float(sample_dict[feat]) for feat in self.required_features]
        return np.array(ordered_values).reshape(1, -1)

    def predict_one(self, patient_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Predicts Parkinson's Disease likelihood for a single patient dictionary.

        Args:
            patient_dict: Dictionary containing the 22 required biomedical voice features.

        Returns:
            Dictionary containing prediction label, probability, risk band, advice, and model used.
        """
        X_raw = self.validate_features(patient_dict)
        X_scaled = self.scaler.transform(X_raw)

        prob_pd = float(self.model.predict_proba(X_scaled)[0, 1])
        prediction_class = int(self.model.predict(X_scaled)[0])
        label = "Parkinson's Disease" if prediction_class == 1 else "Healthy"
        risk_band, recommendation = self.classify_risk_band(prob_pd)

        return {
            "prediction_label": label,
            "prediction_class": prediction_class,
            "probability_pd": round(prob_pd, 4),
            "risk_band": risk_band,
            "clinical_recommendation": recommendation,
            "model_used": self.model_name,
            "disclaimer": "Decision-support screening tool only; not a substitute for diagnostic medical evaluation."
        }

    def predict_batch(self, input_source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Predicts Parkinson's Disease risk for a batch DataFrame or CSV file.
        """
        if isinstance(input_source, str):
            if not os.path.exists(input_source):
                raise FileNotFoundError(f"Batch input CSV file not found: {input_source}")
            df = pd.read_csv(input_source)
        elif isinstance(input_source, pd.DataFrame):
            df = input_source.copy()
        else:
            raise TypeError("Input must be a CSV filepath string or a pandas DataFrame.")

        # Validate feature columns
        missing_cols = [feat for feat in self.required_features if feat not in df.columns]
        if missing_cols:
            raise ValueError(f"Batch dataset missing required feature column(s): {missing_cols}")

        X_batch = df[self.required_features].values
        X_scaled = self.scaler.transform(X_batch)

        probs_pd = self.model.predict_proba(X_scaled)[:, 1]
        preds = self.model.predict(X_scaled)

        df_out = df.copy()
        df_out["predicted_class"] = preds
        df_out["predicted_label"] = np.where(preds == 1, "Parkinson's Disease", "Healthy")
        df_out["pd_probability"] = np.round(probs_pd, 4)

        risk_bands = [self.classify_risk_band(p)[0] for p in probs_pd]
        recommendations = [self.classify_risk_band(p)[1] for p in probs_pd]
        
        df_out["risk_band"] = risk_bands
        df_out["clinical_recommendation"] = recommendations

        return df_out


if __name__ == "__main__":
    # Test standalone predictor execution with a sample patient profile
    print("\n--- Running Standalone Predictor Test ---")
    try:
        predictor = ParkinsonsPredictor()
        
        # Sample healthy voice profile
        sample_patient = {
            "MDVP:Fo(Hz)": 197.076,
            "MDVP:Fhi(Hz)": 206.896,
            "MDVP:Flo(Hz)": 192.055,
            "MDVP:Jitter(%)": 0.00289,
            "MDVP:Jitter(Abs)": 0.00001,
            "MDVP:RAP": 0.00166,
            "MDVP:PPQ": 0.00168,
            "Jitter:DDP": 0.00498,
            "MDVP:Shimmer": 0.01098,
            "MDVP:Shimmer(dB)": 0.09700,
            "Shimmer:APQ3": 0.00563,
            "Shimmer:APQ5": 0.00680,
            "MDVP:APQ": 0.00802,
            "Shimmer:DDA": 0.01689,
            "NHR": 0.00339,
            "HNR": 26.77500,
            "RPDE": 0.422229,
            "DFA": 0.648633,
            "spread1": -7.348300,
            "spread2": 0.177551,
            "D2": 1.743867,
            "PPE": 0.085569
        }

        res = predictor.predict_one(sample_patient)
        print("\nSingle Patient Prediction Result:")
        for k, v in res.items():
            print(f"  {k:<25}: {v}")

    except Exception as e:
        print(f"Standalone prediction check failed (requires trained model in 'models/'): {e}")
