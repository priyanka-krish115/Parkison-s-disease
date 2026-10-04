# Parkinson's Disease Prediction System (ML)

A clinical decision-support machine learning system that analyzes biomedical voice-measurement data to predict the likelihood of Parkinson's Disease (PD) in patients.

> [!IMPORTANT]
> **Clinical Decision-Support Disclaimer**: This tool is strictly intended as a preliminary screening aid to assist clinicians and neurologists. It is **NOT a diagnostic replacement** for professional medical diagnosis.

---

## 📌 Background & Vocal Acoustic Biomarkers

Parkinson's Disease (PD) is a progressive neurodegenerative disorder affecting motor and vocal control systems. Dysphonia (impairment of voice production) occurs in over **90% of early-stage Parkinson's patients**. 

Because vocal subtle changes (vocal cord tremors, reduced pitch dynamic range, micro-fluctuations in amplitude and frequency) manifest before severe motor symptoms, **acoustic voice measurement** provides a non-invasive, low-cost quantitative biomarker for early screening.

### Key Voice Features Analyzed (22 Features):
- **Fundamental Frequency ($F_0$)**: Average (`MDVP:Fo`), maximum (`MDVP:Fhi`), and minimum (`MDVP:Flo`) vocal pitch frequency.
- **Jitter Parameters**: Cycle-to-cycle frequency variation (`MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`). High jitter signals pitch instability.
- **Shimmer Parameters**: Cycle-to-cycle amplitude variation (`MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`). High shimmer indicates vocal loudness irregularity.
- **Harmonic & Noise Measures**: Harmonics-to-Noise Ratio (`HNR`) and Noise-to-Harmonics Ratio (`NHR`). PD patients exhibit elevated breathiness/noise (`NHR`) and lower pure harmonic energy (`HNR`).
- **Non-Linear Dynamics**: Recurrence Period Density Entropy (`RPDE`), Detrended Fluctuation Analysis (`DFA`), Pitch Period Entropy (`PPE`), and Correlation Dimension (`D2`, `spread1`, `spread2`).

---

## 📁 Project Structure

```
parkinsons_prediction/
├── data/
│   └── parkinsons.csv            # Active dataset (UCI real dataset or synthetic fallback)
├── src/
│   ├── generate_dataset.py       # Realistic synthetic data generator (~585 rows, ~75/25 balance)
│   ├── preprocessing.py          # Data loading, median imputation, stratified 80/20 split, scaling
│   ├── train.py                  # 5-fold CV comparison, GridSearchCV tuning, test evaluation
│   ├── predict.py                # ParkinsonsPredictor inference class for new patients
│   └── eda.py                    # Exploratory + evaluation visualization plots
├── models/                       # Saved best model, fitted scaler, feature names & metadata
├── outputs/                      # Saved high-resolution visualization PNG artifacts
├── main.py                       # Pipeline orchestrator & CLI entry point
├── requirements.txt              # Core project dependencies
└── README.md                     # Documentation
```

---

## ⚡ Quick Start

### 1. Prerequisites & Installation

Ensure Python 3.11+ is installed. Clone the repository and install requirements:

```bash
git clone https://github.com/your-username/parkinsons_prediction.git
cd parkinsons_prediction

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run full End-to-End Pipeline

To execute the entire pipeline (dataset acquisition, preprocessing, cross-validation model benchmarking, hyperparameter tuning, plot rendering, and sample inference):

```bash
python main.py
```

### 3. CLI Command Options

- **Use Custom Data File**:
  ```bash
  python main.py --data path/to/my_patients.csv
  ```
- **Fast Execution (Skip Plot Rendering)**:
  ```bash
  python main.py --skip-plots
  ```
- **Run Individual Standalone Modules**:
  ```bash
  python src/generate_dataset.py --samples 600 --output data/custom_synthetic.csv
  python src/preprocessing.py --data data/parkinsons.csv
  python src/train.py --data data/parkinsons.csv
  python src/eda.py --data data/parkinsons.csv
  python src/predict.py
  ```

---

## 📊 Synthetic vs. Real Dataset

> [!NOTE]
> **Dataset Sourcing**:
> Upon initial execution of `main.py`, the system automatically attempts to fetch the official **Oxford / UCI Parkinson's Disease Detection Dataset** (`https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data`).
>
> If internet access is unavailable or blocked, the system automatically falls back to generating a realistic **synthetic dataset** (~585 rows, ~75% PD positive / 25% healthy control) using `src/generate_dataset.py`.

### Swapping in a Custom / Real Clinical CSV:
The system is built schema-driven. You can drop in any CSV matching the 22 UCI biomedical features + `status` column and execute:
```bash
python main.py --data data/my_real_data.csv
```

---

## 🔬 Modeling Strategy & Clinical Evaluation Rationale

### Model Benchmark Comparison
The pipeline evaluates 4 distinct classifier families via **5-Fold Stratified Cross-Validation** scored on **ROC-AUC**:
1. **Logistic Regression** (`class_weight="balanced"`)
2. **Random Forest Classifier** (`class_weight="balanced"`)
3. **Support Vector Machine (SVM)** (`kernel="rbf"`, `class_weight="balanced"`, `probability=True`)
4. **XGBoost Classifier** (`scale_pos_weight` tuned to class imbalance ratio)

The model family achieving the highest mean cross-validation ROC-AUC is automatically selected for hyperparameter tuning using `GridSearchCV`.

### Clinical Metric Rationale
In clinical screening, **the cost of a False Negative (FN)** (failing to identify a patient with early Parkinson's) far outweighs **the cost of a False Positive (FP)** (flagging a healthy control for follow-up evaluation).

Therefore, evaluation prioritizes:
- **Sensitivity / Recall ($TP / (TP + FN)$)**: Ensures maximum capture of true Parkinson's cases.
- **ROC-AUC**: Evaluates discrimination power across all probability thresholds.
- **Specificity ($TN / (TN + FP)$)**: Measures accurate exclusion of healthy controls.

---

## ⚕️ Clinical & Ethical Considerations

1. **Screening Aid, Not Diagnosis**: High probability scores indicate elevated vocal perturbation risk requiring follow-up neurological assessment (e.g., UPDRS scale evaluation, DaTscan neuroimaging).
2. **Probability Threshold Tuning**: The default classification threshold is set to 0.50. In high-risk screening environments, clinicians can adjust the decision boundary downward (e.g., 0.35) to boost Sensitivity further.
3. **Target Population Validation**: Acoustic biomarkers can vary based on age, gender, recording hardware, background noise, and native language. Models must be validated on the specific target patient demographic prior to deployment.
4. **Data Privacy & Compliance (HIPAA / GDPR)**: When processing real patient voice recordings or clinical measurements, ensure strict adherence to HIPAA, GDPR, and institutional IRB data privacy guidelines. Personal identifiers (`name`) must always be anonymized before ingestion.

---

## 💡 Future Extensions

- **Explainability**: Integrate **SHAP (SHapley Additive exPlanations)** values to generate per-patient waterfall feature contribution plots.
- **Interactive UI**: Build a **Streamlit** or **FastAPI + React** Web UI allowing clinicians to upload voice CSVs or record audio directly.
- **Audio Processing Integration**: Add automated feature extraction from raw `.wav` audio files using `librosa` or `praat-parselmouth`.
- **Model Monitoring & Retraining**: Implement automated data drift tracking and scheduled model retraining loops as new patient cohorts arrive.
