"""
Main End-to-End Orchestrator for Parkinson's Disease Prediction ML System.

Usage:
    python main.py                              # Uses default dataset (downloads real UCI dataset or generates synthetic)
    python main.py --data path/to/dataset.csv   # Uses custom CSV dataset
    python main.py --skip-plots                  # Executes pipeline bypassing plot rendering
"""

import os
import sys
import argparse
import logging
import requests
import pandas as pd

# Setup project import paths
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.generate_dataset import generate_synthetic_data
from src.preprocessing import preprocess_data
from src.train import train_pipeline
from src.eda import run_all_eda
from src.predict import ParkinsonsPredictor

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

UCI_PARKINSONS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"


def ensure_dataset(data_path: str) -> bool:
    """
    Ensures dataset availability.
    1. If data_path exists, use it.
    2. If missing and data_path is default, attempt fetching official UCI dataset.
    3. If network fetch fails, trigger synthetic fallback generator.
    """
    if os.path.exists(data_path):
        logger.info(f"Using existing dataset found at '{data_path}'.")
        return True

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    logger.info(f"Dataset not found at '{data_path}'. Sourcing dataset...")

    # Attempt fetching real UCI dataset
    try:
        logger.info(f"Attempting to download official UCI Parkinson's dataset from:\n  {UCI_PARKINSONS_URL}")
        response = requests.get(UCI_PARKINSONS_URL, timeout=10)
        if response.status_code == 200:
            with open(data_path, "wb") as f:
                f.write(response.content)
            
            # Verify valid CSV format
            df_test = pd.read_csv(data_path)
            logger.info(f"[SUCCESS] Real UCI Parkinson's dataset successfully downloaded to '{data_path}' ({len(df_test)} rows, {len(df_test.columns)} columns).")
            return True
        else:
            logger.warning(f"UCI dataset download failed HTTP status: {response.status_code}.")
    except Exception as e:
        logger.warning(f"Could not reach UCI ML Repository ({e}).")

    # Synthetic fallback
    logger.info("Triggering synthetic fallback dataset generator...")
    generate_synthetic_data(output_path=data_path, num_samples=585, pd_ratio=0.75, random_state=42)
    return False


def run_pipeline(data_path: str = "data/parkinsons.csv", skip_plots: bool = False):
    """
    Executes the complete machine learning pipeline top to bottom.
    """
    print("\n" + "=" * 70)
    print("      PARKINSON'S DISEASE PREDICTION SYSTEM - ML PIPELINE      ")
    print("=" * 70)
    print("  Clinical Decision Support Tool - Acoustic Voice Biomarker Analysis  ")
    print("=" * 70 + "\n")

    # Step 1: Data Acquisition / Sourcing
    logger.info("=== STEP 1: DATA ACQUISITION & VALIDATION ===")
    is_real = ensure_dataset(data_path)
    data_source_label = "Real UCI Dataset" if is_real else "Synthetic Fallback Dataset (Demo Only)"
    logger.info(f"Active Data Source: {data_source_label}\n")

    # Step 2: Model Benchmarking, Tuning & Evaluation
    logger.info("=== STEP 2: MODEL SELECTION, HYPERPARAMETER TUNING & EVALUATION ===")
    train_results = train_pipeline(data_path=data_path, models_dir="models")
    best_model_name = train_results["best_model_name"]
    metrics = train_results["metrics"]

    # Step 3: Visualizations & EDA
    if not skip_plots:
        logger.info("\n=== STEP 3: GENERATING EXPLORATORY & EVALUATION PLOTS ===")
        run_all_eda(data_path=data_path, models_dir="models", output_dir="outputs")
    else:
        logger.info("\n[SKIP] Bypassing plot generation (--skip-plots requested).")

    # Step 4: Inference Demonstration
    logger.info("\n=== STEP 4: INFERENCE PREDICTOR DEMONSTRATION ===")
    predictor = ParkinsonsPredictor(models_dir="models")

    # Sample test patient voice profile (high perturbation profile)
    sample_pd_patient = {
        "MDVP:Fo(Hz)": 119.992,
        "MDVP:Fhi(Hz)": 157.302,
        "MDVP:Flo(Hz)": 74.997,
        "MDVP:Jitter(%)": 0.00784,
        "MDVP:Jitter(Abs)": 0.00007,
        "MDVP:RAP": 0.00370,
        "MDVP:PPQ": 0.00554,
        "Jitter:DDP": 0.01109,
        "MDVP:Shimmer": 0.04374,
        "MDVP:Shimmer(dB)": 0.42600,
        "Shimmer:APQ3": 0.02182,
        "Shimmer:APQ5": 0.03130,
        "MDVP:APQ": 0.02971,
        "Shimmer:DDA": 0.06545,
        "NHR": 0.02211,
        "HNR": 21.03300,
        "RPDE": 0.414783,
        "DFA": 0.815285,
        "spread1": -4.813031,
        "spread2": 0.266482,
        "D2": 2.301442,
        "PPE": 0.284654
    }

    prediction = predictor.predict_one(sample_pd_patient)

    print("\n" + "=" * 70)
    print("                   EXAMPLE PATIENT INFERENCE RESULT                 ")
    print("=" * 70)
    print(f"  Model Used:                {prediction['model_used']}")
    print(f"  Predicted Label:           {prediction['prediction_label']} (Class {prediction['prediction_class']})")
    print(f"  Parkinson's Probability:   {prediction['probability_pd'] * 100:.2f}%")
    print(f"  Clinical Risk Band:        {prediction['risk_band']}")
    print(f"  Recommendation:            {prediction['clinical_recommendation']}")
    print("-" * 70)
    print(f"  Disclaimer: {prediction['disclaimer']}")
    print("=" * 70 + "\n")

    print(f"[SUCCESS] End-to-end pipeline execution complete!")
    print(f"  - Saved Model Artifacts: 'models/'")
    print(f"  - Generated Plots:       'outputs/'")
    print(f"  - Final Test ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  - Final Test Sensitivity:{metrics['recall_sensitivity']:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parkinson's Disease Prediction ML Pipeline")
    parser.add_argument("--data", type=str, default="data/parkinsons.csv", help="Path to input dataset CSV")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating plot artifacts in outputs/")
    args = parser.parse_args()

    run_pipeline(data_path=args.data, skip_plots=args.skip_plots)
