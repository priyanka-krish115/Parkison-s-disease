"""
Exploratory Data Analysis (EDA) and Model Evaluation Plotting Module.

This module generates publication-quality data visualizations:
1. `class_balance.png`: Target class distribution (Healthy vs Parkinson's Disease).
2. `feature_distributions.png`: Multi-panel KDE plots comparing top vocal features.
3. `correlation_heatmap.png`: High-contrast feature correlation matrix.
4. `feature_importance.png`: Top predictive acoustic voice biomarkers.
5. `roc_curve.png`: Test set ROC curve with AUC score.
6. `confusion_matrix.png`: Heatmap of classification confusion matrix.

All generated plots are saved to the `outputs/` directory.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import joblib

# Set professional plotting aesthetics
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16
})

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Color Palette: Bespoke Editorial Clinical (Mineral Blue & Terracotta Rust)
PALETTE = {
    "Healthy": "#047857",       # Warm Emerald / Sage Green
    "Parkinson's": "#c2410c",   # Terracotta Rust / Crimson
    "Neutral": "#1e3a8a",       # Deep Mineral Blue
    "Heatmap": "Blues"          # Clean High-Contrast Clinical Palette
}


def plot_class_balance(df: pd.DataFrame, output_dir: str = "outputs") -> str:
    """Plots target status class distribution."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    counts = df["status"].value_counts().sort_index()
    labels = ["Healthy (0)", "Parkinson's (1)"]
    colors = [PALETTE["Healthy"], PALETTE["Parkinson's"]]

    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f"{height}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("Patient Count")
    ax.set_title("Dataset Class Balance (Healthy vs. Parkinson's Disease)", pad=15)
    ax.set_ylim(0, max(counts) * 1.18)
    plt.tight_layout()

    file_path = os.path.join(output_dir, "class_balance.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Saved class balance plot to '{file_path}'.")
    return file_path


def plot_feature_distributions(df: pd.DataFrame, output_dir: str = "outputs") -> str:
    """Plots KDE distributions for top discriminatory voice features."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Key vocal biomarkers
    top_features = ["PPE", "spread1", "MDVP:Fo(Hz)", "HNR", "NHR", "RPDE"]
    available_features = [f for f in top_features if f in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    df_copy = df.copy()
    df_copy["Status_Label"] = df_copy["status"].map({0: "Healthy", 1: "Parkinson's"})

    for i, feature in enumerate(available_features):
        ax = axes[i]
        sns.kdeplot(
            data=df_copy, x=feature, hue="Status_Label",
            palette={"Healthy": PALETTE["Healthy"], "Parkinson's": PALETTE["Parkinson's"]},
            fill=True, common_norm=False, alpha=0.4, linewidth=2, ax=ax
        )
        ax.set_title(f"Distribution: {feature}", fontsize=13, fontweight="bold")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")

    plt.suptitle("Acoustic Feature Distributions: Healthy Controls vs. Parkinson's Patients", y=0.98, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    file_path = os.path.join(output_dir, "feature_distributions.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature distribution KDE plots to '{file_path}'.")
    return file_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = "outputs") -> str:
    """Plots correlation heatmap for acoustic features."""
    os.makedirs(output_dir, exist_ok=True)

    numeric_df = df.drop(columns=["name"], errors="ignore")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, mask=mask, cmap=PALETTE["Heatmap"], vmin=-1, vmax=1,
        center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot=False, ax=ax
    )

    ax.set_title("Acoustic Feature Correlation Heatmap", pad=20, fontweight="bold")
    plt.tight_layout()

    file_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Saved correlation heatmap to '{file_path}'.")
    return file_path


def plot_feature_importance(importance_csv_path: str = "models/feature_importances.csv", output_dir: str = "outputs") -> str:
    """Plots top feature importances from trained model."""
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(importance_csv_path):
        logger.warning(f"Feature importance CSV missing at '{importance_csv_path}'. Skipping plot.")
        return ""

    df_imp = pd.read_csv(importance_csv_path).head(15) # Top 15 features

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1], color=PALETTE["Neutral"], edgecolor="black", height=0.65)

    ax.set_xlabel("Relative Importance Score")
    ax.set_ylabel("Acoustic Voice Feature")
    ax.set_title("Top 15 Predictive Voice Features for Parkinson's Detection", pad=15, fontweight="bold")

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.4f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10)

    ax.set_xlim(0, max(df_imp["importance"]) * 1.15)
    plt.tight_layout()

    file_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance plot to '{file_path}'.")
    return file_path


def plot_roc_curve(y_test: np.ndarray, y_prob: np.ndarray, model_name: str = "ML Model", output_dir: str = "outputs") -> str:
    """Plots Receiver Operating Characteristic (ROC) curve."""
    os.makedirs(output_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PALETTE["Healthy"], lw=2.5, label=f"{model_name} (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")

    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title(f"ROC Curve - Held-Out Test Set ({model_name})", pad=15, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()

    file_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve plot to '{file_path}'.")
    return file_path


def plot_confusion_matrix_heatmap(y_test: np.ndarray, y_pred: np.ndarray, model_name: str = "ML Model", output_dir: str = "outputs") -> str:
    """Plots heatmapped confusion matrix."""
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Custom annotations showing counts and percentages
    labels = [
        [f"TN: {tn}\n({tn/len(y_test):.1%})", f"FP: {fp}\n({fp/len(y_test):.1%})"],
        [f"FN: {fn}\n({fn/len(y_test):.1%})", f"TP: {tp}\n({tp/len(y_test):.1%})"]
    ]

    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues", cbar=False,
        xticklabels=["Healthy (0)", "Parkinson's (1)"],
        yticklabels=["Healthy (0)", "Parkinson's (1)"],
        linewidths=1.5, linecolor="black", ax=ax, annot_kws={"fontsize": 12, "fontweight": "bold"}
    )

    ax.set_xlabel("Predicted Diagnosis")
    ax.set_ylabel("Actual True Diagnosis")
    ax.set_title(f"Test Set Confusion Matrix ({model_name})", pad=15, fontweight="bold")

    plt.tight_layout()

    file_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to '{file_path}'.")
    return file_path


def run_all_eda(data_path: str = "data/parkinsons.csv", models_dir: str = "models", output_dir: str = "outputs"):
    """Runs all EDA and evaluation plots."""
    logger.info("=== Generating Exploratory and Evaluation Visualizations ===")
    
    # Load dataset for EDA
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        plot_class_balance(df, output_dir=output_dir)
        plot_feature_distributions(df, output_dir=output_dir)
        plot_correlation_heatmap(df, output_dir=output_dir)

    # Plot feature importances if present
    plot_feature_importance(importance_csv_path=os.path.join(models_dir, "feature_importances.csv"), output_dir=output_dir)

    # Plot post-training evaluation plots if test evaluation data exists
    test_eval_path = os.path.join(models_dir, "test_eval_data.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    if os.path.exists(test_eval_path):
        test_data = joblib.load(test_eval_path)
        model_name = "ML Model"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                model_name = json.load(f).get("best_model_name", model_name)

        plot_roc_curve(test_data["y_test"], test_data["y_prob"], model_name=model_name, output_dir=output_dir)
        plot_confusion_matrix_heatmap(test_data["y_test"], test_data["y_pred"], model_name=model_name, output_dir=output_dir)

    logger.info(f"[SUCCESS] All visualizations successfully saved to '{output_dir}/'.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run EDA and generate plots")
    parser.add_argument("--data", type=str, default="data/parkinsons.csv", help="Input dataset CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.generate_dataset import generate_synthetic_data
        generate_synthetic_data(output_path=args.data)

    run_all_eda(data_path=args.data)
