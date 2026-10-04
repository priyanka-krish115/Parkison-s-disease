"""
Data Preprocessing Module for Parkinson's Disease Prediction Pipeline.

This module handles:
1. CSV dataset loading & schema validation.
2. Missing value imputation (median fill with logging warnings).
3. Feature extraction & label separation (`name` dropping, `status` target extraction).
4. Stratified 80/20 train/test splitting.
5. StandardScaler normalization (fit strictly on training set).
6. Joblib serialization of scaler and feature ordering into `models/`.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "status"
IDENTIFIER_COLUMN = "name"


def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Loads raw CSV dataset and performs missing value checks and imputation.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at path: {data_path}")

    logger.info(f"Loading raw dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")

    # Check missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        logger.warning(f"Detected {total_missing} missing values across columns:")
        for col, count in missing_counts[missing_counts > 0].items():
            logger.warning(f"  - Column '{col}': {count} missing value(s)")
        
        # Numeric median imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Imputed missing values in '{col}' with median: {median_val:.4f}")

    return df


def preprocess_data(
    data_path: str = "data/parkinsons.csv",
    test_size: float = 0.20,
    random_state: int = 42,
    save_artifacts: bool = True,
    models_dir: str = "models"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Executes full preprocessing pipeline: loading, splitting, scaling, and artifact saving.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler
    """
    df = load_dataset(data_path)

    # Validate target column presence
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from dataset.")

    # Separate features and target
    y = df[TARGET_COLUMN].values
    X_df = df.drop(columns=[TARGET_COLUMN])

    # Drop identifier column if present
    if IDENTIFIER_COLUMN in X_df.columns:
        X_df = X_df.drop(columns=[IDENTIFIER_COLUMN])
        logger.info(f"Dropped identifier column '{IDENTIFIER_COLUMN}'.")

    feature_names = list(X_df.columns)
    logger.info(f"Extracted {len(feature_names)} acoustic voice measurement features.")

    X = X_df.values

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    logger.info(f"Stratified split completed: Train={len(y_train)} samples, Test={len(y_test)} samples.")
    logger.info(f"Train class balance: PD(1)={np.sum(y_train==1)}, Healthy(0)={np.sum(y_train==0)}")
    logger.info(f"Test class balance:  PD(1)={np.sum(y_test==1)}, Healthy(0)={np.sum(y_test==0)}")

    # StandardScaler fit on training set ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("StandardScaler fitted on training data and applied to train/test splits.")

    if save_artifacts:
        os.makedirs(models_dir, exist_ok=True)
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        features_path = os.path.join(models_dir, "feature_names.joblib")
        
        joblib.dump(scaler, scaler_path)
        joblib.dump(feature_names, features_path)
        logger.info(f"Saved fitted scaler to '{scaler_path}'.")
        logger.info(f"Saved feature list ({len(feature_names)} features) to '{features_path}'.")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess Parkinson's dataset")
    parser.add_argument("--data", type=str, default="data/parkinsons.csv", help="Input dataset CSV path")
    args = parser.parse_args()

    # Create dummy data if missing when running standalone
    if not os.path.exists(args.data):
        logger.warning(f"Data file '{args.data}' does not exist. Triggering synthetic generator...")
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.generate_dataset import generate_synthetic_data
        generate_synthetic_data(output_path=args.data)

    X_train_s, X_test_s, y_train, y_test, features, scaler = preprocess_data(data_path=args.data)
    print(f"\n[SUCCESS] Standalone preprocessing completed.")
    print(f"X_train_scaled shape: {X_train_s.shape}, X_test_scaled shape: {X_test_s.shape}")
