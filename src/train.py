"""
Model Training, Cross-Validation, Tuning, and Evaluation Module.

This module orchestrates:
1. Model benchmark comparison (5-fold Stratified K-Fold CV scored on ROC-AUC):
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (RBF Kernel)
   - XGBoost
2. Class weight balancing across all candidate model families.
3. Selection of the top model family based on CV ROC-AUC performance.
4. Hyperparameter optimization via GridSearchCV.
5. Held-out test set evaluation:
   - Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, ROC-AUC.
   - Confusion matrix & Scikit-learn Classification Report.
6. Feature importance extraction & export (`models/feature_importances.csv`).
7. Model serialization & metadata saving into `models/`.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance
import joblib

# Ensure local imports work when run standalone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess_data, load_dataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes clinical classification evaluation metrics including Sensitivity & Specificity.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall_sensitivity = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall_sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }


def compare_model_families(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Dict[str, float]], str]:
    """
    Compares 4 candidate model families using Stratified 5-Fold Cross Validation.
    """
    logger.info("=== 1. Starting 5-Fold Cross-Validation Model Comparison ===")
    
    # Compute class ratio for XGBoost scale_pos_weight
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", random_state=random_state, max_iter=1000
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=random_state
        ),
        "SVM (RBF Kernel)": CalibratedClassifierCV(
            SVC(kernel="rbf", class_weight="balanced", random_state=random_state), ensemble=False
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight, random_state=random_state, eval_metric="logloss"
        )
    }

    cv_results = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    best_score = -1.0
    best_model_name = ""

    print("\n" + "=" * 65)
    print(f"{'Model Family':<25} | {'Mean CV ROC-AUC':<18} | {'Std Dev':<10}")
    print("=" * 65)

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
        mean_score = scores.mean()
        std_score = scores.std()
        
        cv_results[name] = {"mean_roc_auc": float(mean_score), "std_roc_auc": float(std_score)}
        print(f"{name:<25} | {mean_score:<18.4f} | {std_score:<10.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name

    print("=" * 65)
    logger.info(f"Top performing model family: '{best_model_name}' (Mean CV ROC-AUC = {best_score:.4f})")
    
    return cv_results, best_model_name


def tune_best_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Executes GridSearchCV hyperparameter tuning for the selected model family.
    """
    logger.info(f"=== 2. Tuning Best Model Family ('{model_name}') with GridSearchCV ===")

    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    param_grids = {
        "Logistic Regression": {
            "estimator": LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000),
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"]
            }
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(class_weight="balanced", random_state=random_state),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10, 15],
                "min_samples_split": [2, 5]
            }
        },
        "SVM (RBF Kernel)": {
            "estimator": CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced", random_state=random_state), ensemble=False),
            "param_grid": {
                "estimator__C": [0.1, 1.0, 10.0],
                "estimator__gamma": ["scale", "auto", 0.01, 0.1]
            }
        },
        "XGBoost": {
            "estimator": XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=random_state, eval_metric="logloss"),
            "param_grid": {
                "n_estimators": [50, 100, 150],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        }
    }

    config = param_grids[model_name]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    grid_search = GridSearchCV(
        estimator=config["estimator"],
        param_grid=config["param_grid"],
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    logger.info(f"GridSearchCV complete. Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    logger.info(f"Best Hyperparameters: {grid_search.best_params_}")

    return grid_search.best_estimator_, grid_search.best_params_


def extract_feature_importances(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list
) -> pd.DataFrame:
    """
    Extracts feature importances depending on model type (tree-based, linear, or kernel SVM).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        # Fallback to Permutation Importance for nonlinear kernel models (SVM)
        perm_importance = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42)
        importances = perm_importance.importances_mean

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_importance


def train_pipeline(
    data_path: str = "data/parkinsons.csv",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Runs full training, tuning, evaluation, and saving workflow.
    """
    os.makedirs(models_dir, exist_ok=True)

    # 1. Preprocess & Scale Data
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(
        data_path=data_path, models_dir=models_dir
    )

    # 2. Benchmark Model Families
    cv_results, best_model_name = compare_model_families(X_train, y_train)

    # 3. Tune Best Model
    best_model, best_params = tune_best_model(best_model_name, X_train, y_train)

    # 4. Fit tuned model on entire train split
    best_model.fit(X_train, y_train)

    # 5. Evaluate on Held-out Test Set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, y_pred, y_prob)

    print("\n" + "=" * 65)
    print(f"HELD-OUT TEST SET EVALUATION METRICS ({best_model_name.upper()})")
    print("=" * 65)
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  Precision:           {metrics['precision']:.4f}")
    print(f"  Recall (Sensitivity):{metrics['recall_sensitivity']:.4f}")
    print(f"  Specificity:         {metrics['specificity']:.4f}")
    print(f"  F1 Score:            {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC Score:       {metrics['roc_auc']:.4f}")
    print("-" * 65)
    print(f"  True Negatives (TN):  {metrics['true_negatives']}")
    print(f"  False Positives (FP): {metrics['false_positives']}")
    print(f"  False Negatives (FN): {metrics['false_negatives']}")
    print(f"  True Positives (TP):  {metrics['true_positives']}")
    print("=" * 65)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"]))

    # 6. Extract & Save Feature Importances
    df_importance = extract_feature_importances(best_model, best_model_name, X_test, y_test, feature_names)
    importance_path = os.path.join(models_dir, "feature_importances.csv")
    df_importance.to_csv(importance_path, index=False)
    logger.info(f"Saved feature importances to '{importance_path}'.")

    # 7. Save Model & Metadata
    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)
    logger.info(f"Saved trained best model to '{model_path}'.")

    metadata = {
        "best_model_name": best_model_name,
        "best_params": best_params,
        "cv_results": cv_results,
        "test_metrics": metrics,
        "feature_names": feature_names
    }
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved training metadata to '{metadata_path}'.")

    # Save test set evaluation arrays for plotting module
    test_eval_data = {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "X_test": X_test
    }
    joblib.dump(test_eval_data, os.path.join(models_dir, "test_eval_data.joblib"))

    return {
        "model": best_model,
        "best_model_name": best_model_name,
        "metrics": metrics,
        "df_importance": df_importance,
        "feature_names": feature_names
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Parkinson's Prediction ML model")
    parser.add_argument("--data", type=str, default="data/parkinsons.csv", help="Input dataset path")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        from src.generate_dataset import generate_synthetic_data
        generate_synthetic_data(output_path=args.data)

    train_pipeline(data_path=args.data)
