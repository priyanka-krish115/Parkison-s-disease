"""
Inference Module for Parkinson's Disease Prediction.

This module exposes the `ParkinsonsPredictor` class for predicting Parkinson's Disease likelihood
from acoustic voice measurement inputs.

Optimized for lightweight, zero-dependency serverless deployment (Vercel, AWS Lambda)
using exported JSON tree and scaling structures.
"""

import os
import json
import math
import logging
from typing import Dict, Any, Union, List, Tuple

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ParkinsonsPredictor:
    """
    Lightweight inference predictor class loading exported model trees, scaler parameters,
    and feature definitions without requiring heavyweight C++/ML dependencies.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.trees_path = os.path.join(models_dir, "xgb_trees.json")
        self.scaler_path = os.path.join(models_dir, "scaler_params.json")
        self.metadata_path = os.path.join(models_dir, "model_metadata.json")

        self.trees = []
        self.scaler_mean = []
        self.scaler_scale = []
        self.required_features: List[str] = []
        self.model_name = "XGBoost Classifier"

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model trees, scaler parameters, and metadata."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.model_name = meta.get("best_model_name", self.model_name)
                self.required_features = meta.get("feature_names", [])

        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
                self.scaler_mean = s_data.get("mean", [])
                self.scaler_scale = s_data.get("scale", [])

        if os.path.exists(self.trees_path):
            with open(self.trees_path, "r", encoding="utf-8") as f:
                self.trees = json.load(f)
        else:
            # Fallback to joblib if available in development environment
            try:
                import joblib
                joblib_model_path = os.path.join(self.models_dir, "best_model.joblib")
                if os.path.exists(joblib_model_path):
                    self.joblib_model = joblib.load(joblib_model_path)
            except Exception as err:
                logger.warning(f"Could not load fallback joblib model: {err}")

        logger.info(f"Loaded '{self.model_name}' predictor with {len(self.required_features)} features and {len(self.trees)} trees.")

    @staticmethod
    def classify_risk_band(probability: float) -> Tuple[str, str]:
        """
        Classifies Parkinson's risk probability into clinical risk bands.
        """
        if probability < 0.35:
            return "Low Risk", "Standard routine monitoring."
        elif probability < 0.65:
            return "Moderate Risk", "Consider follow-up vocal & motor screening."
        else:
            return "High Risk", "Recommend clinical evaluation by a specialist/neurologist."

    def validate_features(self, sample_dict: Dict[str, float]) -> List[float]:
        """
        Validates feature presence in patient dict and returns ordered feature value list.
        """
        missing = [feat for feat in self.required_features if feat not in sample_dict]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required feature(s) for inference:\n"
                f"Missing: {missing}\nRequired: {self.required_features}"
            )
        return [float(sample_dict[feat]) for feat in self.required_features]

    def _evaluate_node(self, node: Dict[str, Any], x_vec: List[float]) -> float:
        """Recursively traverses a single decision tree node."""
        if "leaf" in node:
            return float(node["leaf"])

        split_feat = node["split"]
        if split_feat.startswith("f") and split_feat[1:].isdigit():
            feat_idx = int(split_feat[1:])
        elif split_feat in self.required_features:
            feat_idx = self.required_features.index(split_feat)
        else:
            feat_idx = 0

        val = x_vec[feat_idx]
        cond = float(node["split_condition"])

        target_id = node["yes"] if val < cond else node["no"]
        for child in node.get("children", []):
            if child.get("nodeid") == target_id:
                return self._evaluate_node(child, x_vec)

        return 0.0

    def predict_one(self, patient_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Predicts Parkinson's Disease likelihood for a single patient dictionary.
        """
        raw_vals = self.validate_features(patient_dict)

        # Standard scaling: (x - mean) / scale
        if self.scaler_mean and self.scaler_scale:
            scaled_vals = [
                (raw_vals[i] - self.scaler_mean[i]) / self.scaler_scale[i]
                for i in range(len(raw_vals))
            ]
        else:
            scaled_vals = raw_vals

        if self.trees:
            # Evaluate all decision trees in the ensemble
            score = 0.0
            for tree in self.trees:
                score += self._evaluate_node(tree, scaled_vals)
            # Sigmoid activation for binary probability
            prob_pd = 1.0 / (1.0 + math.exp(-score))
        elif hasattr(self, "joblib_model"):
            import numpy as np
            prob_pd = float(self.joblib_model.predict_proba(np.array(scaled_vals).reshape(1, -1))[0, 1])
        else:
            raise RuntimeError("No inference engine available. Model trees missing.")

        prediction_class = 1 if prob_pd >= 0.5 else 0
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

    def predict_batch(self, input_source: Any) -> Any:
        """
        Predicts Parkinson's Disease risk for a list of dictionaries or a pandas DataFrame.
        """
        if isinstance(input_source, list):
            return [self.predict_one(row) for row in input_source]
        
        try:
            import pandas as pd
            if isinstance(input_source, pd.DataFrame):
                records = input_source.to_dict(orient="records")
                preds = [self.predict_one(r) for r in records]
                df_out = input_source.copy()
                df_out["predicted_class"] = [p["prediction_class"] for p in preds]
                df_out["predicted_label"] = [p["prediction_label"] for p in preds]
                df_out["pd_probability"] = [p["probability_pd"] for p in preds]
                df_out["risk_band"] = [p["risk_band"] for p in preds]
                df_out["clinical_recommendation"] = [p["clinical_recommendation"] for p in preds]
                return df_out
        except ImportError:
            pass

        raise TypeError("Input must be a list of feature dictionaries or a pandas DataFrame.")
