"""
Synthetic Dataset Generator for Parkinson's Disease Voice Biomarkers.

This module generates a realistic synthetic dataset (~585 rows, ~75% PD positive, ~25% healthy control)
matching the UCI / Oxford Parkinson's Disease Detection Dataset schema (22 biomedical voice features).

NOTE: This synthetic generator is strictly intended for demonstration and offline fallback purposes.
Do not present synthetic samples as real patient data.
"""

import os
import argparse
import numpy as np
import pandas as pd

# UCI Parkinson's Dataset Column Order
COLUMN_NAMES = [
    "name",
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "status",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]


def generate_synthetic_data(
    num_samples: int = 585,
    pd_ratio: float = 0.75,
    random_state: int = 42,
    output_path: str = "data/parkinsons.csv"
) -> pd.DataFrame:
    """
    Generates a synthetic DataFrame replicating acoustic voice measurement features of Parkinson's Disease.

    Acoustic distinctions modeled:
      - PD Patients (status=1): Higher jitter, shimmer, NHR, RPDE, D2, spread1, spread2, PPE; lower HNR.
      - Healthy Controls (status=0): Lower perturbation scores, higher harmonic signal purity (HNR).
    """
    np.random.seed(random_state)

    n_pd = int(num_samples * pd_ratio)
    n_healthy = num_samples - n_pd

    # Generate targets (1 for PD, 0 for Healthy)
    status_pd = np.ones(n_pd, dtype=int)
    status_healthy = np.zeros(n_healthy, dtype=int)
    status = np.concatenate([status_pd, status_healthy])

    # Generate subject names (e.g., phon_R01_S01_1)
    names = [f"phon_R01_S{i//6 + 1:02d}_{i%6 + 1}" for i in range(num_samples)]

    # Feature generation with clinical correlations
    # 1. Fundamental Frequencies (Hz)
    fo_healthy = np.random.normal(180.0, 30.0, n_healthy)
    fo_pd = np.random.normal(145.0, 35.0, n_pd)
    fo = np.concatenate([fo_pd, fo_healthy])
    fo = np.clip(fo, 88.0, 260.0)

    fhi = fo + np.random.uniform(20.0, 100.0, num_samples)
    flo = fo - np.random.uniform(10.0, 50.0, num_samples)
    flo = np.clip(flo, 65.0, None)

    # 2. Jitter Features (Pitch frequency perturbation) - Higher in PD
    jitter_pct_healthy = np.random.gamma(shape=2.0, scale=0.0015, size=n_healthy) + 0.002
    jitter_pct_pd = np.random.gamma(shape=3.5, scale=0.0025, size=n_pd) + 0.004
    jitter_pct = np.concatenate([jitter_pct_pd, jitter_pct_healthy])

    jitter_abs = jitter_pct * (1.0 / fo) * np.random.uniform(0.8, 1.2, num_samples)
    rap = jitter_pct * np.random.uniform(0.45, 0.55, num_samples)
    ppq = jitter_pct * np.random.uniform(0.50, 0.60, num_samples)
    ddp = rap * 3.0 + np.random.normal(0, 0.0002, num_samples)
    ddp = np.clip(ddp, 0.0, None)

    # 3. Shimmer Features (Amplitude perturbation) - Higher in PD
    shimmer_healthy = np.random.gamma(shape=2.5, scale=0.006, size=n_healthy) + 0.012
    shimmer_pd = np.random.gamma(shape=4.0, scale=0.010, size=n_pd) + 0.035
    shimmer = np.concatenate([shimmer_pd, shimmer_healthy])

    shimmer_db = shimmer * np.random.uniform(8.0, 10.0, num_samples)
    apq3 = shimmer * np.random.uniform(0.45, 0.55, num_samples)
    apq5 = shimmer * np.random.uniform(0.52, 0.62, num_samples)
    apq = shimmer * np.random.uniform(0.70, 0.85, num_samples)
    dda = apq3 * 3.0 + np.random.normal(0, 0.001, num_samples)
    dda = np.clip(dda, 0.0, None)

    # 4. Noise Ratios (NHR: Noise-to-Harmonics, HNR: Harmonics-to-Noise)
    nhr_healthy = np.random.exponential(scale=0.008, size=n_healthy) + 0.003
    nhr_pd = np.random.exponential(scale=0.035, size=n_pd) + 0.018
    nhr = np.concatenate([nhr_pd, nhr_healthy])

    hnr_healthy = np.random.normal(26.0, 3.0, n_healthy)
    hnr_pd = np.random.normal(20.0, 4.0, n_pd)
    hnr = np.concatenate([hnr_pd, hnr_healthy])
    hnr = np.clip(hnr, 8.0, 35.0)

    # 5. Non-linear Dynamical Complexity & Recurrence Measures
    rpde_healthy = np.random.normal(0.42, 0.08, n_healthy)
    rpde_pd = np.random.normal(0.54, 0.09, n_pd)
    rpde = np.clip(np.concatenate([rpde_pd, rpde_healthy]), 0.2, 0.75)

    dfa_healthy = np.random.normal(0.65, 0.06, n_healthy)
    dfa_pd = np.random.normal(0.72, 0.06, n_pd)
    dfa = np.clip(np.concatenate([dfa_pd, dfa_healthy]), 0.5, 0.85)

    spread1_healthy = np.random.normal(-6.8, 0.6, n_healthy)
    spread1_pd = np.random.normal(-5.1, 0.8, n_pd)
    spread1 = np.concatenate([spread1_pd, spread1_healthy])

    spread2_healthy = np.random.normal(0.16, 0.05, n_healthy)
    spread2_pd = np.random.normal(0.24, 0.06, n_pd)
    spread2 = np.concatenate([spread2_pd, spread2_healthy])

    d2_healthy = np.random.normal(2.0, 0.3, n_healthy)
    d2_pd = np.random.normal(2.4, 0.4, n_pd)
    d2 = np.concatenate([d2_pd, d2_healthy])

    ppe_healthy = np.random.normal(0.12, 0.04, n_healthy)
    ppe_pd = np.random.normal(0.24, 0.07, n_pd)
    ppe = np.clip(np.concatenate([ppe_pd, ppe_healthy]), 0.03, 0.55)

    # Build DataFrame
    df = pd.DataFrame({
        "name": names,
        "MDVP:Fo(Hz)": fo,
        "MDVP:Fhi(Hz)": fhi,
        "MDVP:Flo(Hz)": flo,
        "MDVP:Jitter(%)": jitter_pct,
        "MDVP:Jitter(Abs)": jitter_abs,
        "MDVP:RAP": rap,
        "MDVP:PPQ": ppq,
        "Jitter:DDP": ddp,
        "MDVP:Shimmer": shimmer,
        "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": apq3,
        "Shimmer:APQ5": apq5,
        "MDVP:APQ": apq,
        "Shimmer:DDA": dda,
        "NHR": nhr,
        "HNR": hnr,
        "status": status,
        "RPDE": rpde,
        "DFA": dfa,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe
    })

    # Reorder columns explicitly according to schema
    df = df[COLUMN_NAMES]

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Synthetic dataset created at '{output_path}' ({len(df)} rows, {df['status'].sum()} PD / {len(df)-df['status'].sum()} Healthy)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Parkinson's voice dataset")
    parser.add_argument("--samples", type=int, default=585, help="Total samples (default: 585)")
    parser.add_argument("--output", type=str, default="data/parkinsons.csv", help="Output CSV path")
    args = parser.parse_args()

    generate_synthetic_data(num_samples=args.samples, output_path=args.output)
