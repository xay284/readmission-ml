"""Domain feature engineering (stateless -> no leakage possible)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import RAW_INPUT_COLS, MED_COLS, NUM_COLS, CAT_COLS

TOP_SPECIALTIES = {
    "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology",
    "Surgery-General", "Nephrology", "Orthopedics", "Orthopedics-Reconstructive",
    "Radiologist", "Pulmonology",
}


def icd9_group(code) -> str:
    """Map an ICD-9 code to a broad clinical group."""
    if pd.isna(code):
        return "Missing"
    s = str(code).strip()
    if s[:1] in "VvEe":
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    i = int(v)
    if 250 <= v < 251:
        return "Diabetes"
    if 390 <= i <= 459 or i == 785:
        return "Circulatory"
    if 460 <= i <= 519 or i == 786:
        return "Respiratory"
    if 520 <= i <= 579 or i == 787:
        return "Digestive"
    if 580 <= i <= 629 or i == 788:
        return "Genitourinary"
    if 710 <= i <= 739:
        return "Musculoskeletal"
    if 800 <= i <= 999:
        return "Injury"
    if 140 <= i <= 239:
        return "Neoplasms"
    return "Other"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Raw encounter table -> model-ready columns (NUM_COLS + CAT_COLS).
    Tolerates missing columns, so it also works for API / dashboard input."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in RAW_INPUT_COLS:
            if c not in X.columns:
                X[c] = np.nan
        X = X.replace({"?": np.nan, "None": np.nan})
        out = pd.DataFrame(index=X.index)

        for c in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
                  "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]:
            out[c] = pd.to_numeric(X[c], errors="coerce")

        ab = X["age"].astype(str).str.extract(r"(\d+)-(\d+)").astype(float)
        out["age_mid"] = ab.mean(axis=1)
        out["n_prior_visits"] = out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"]

        med = X[MED_COLS].fillna("No").astype(str)
        out["n_meds_active"] = (med != "No").sum(axis=1)
        out["n_meds_changed"] = med.isin(["Up", "Down"]).sum(axis=1)
        days = out["time_in_hospital"].clip(lower=1)
        out["meds_per_day"] = out["num_medications"] / days
        out["labs_per_day"] = out["num_lab_procedures"] / days

        for c in ["race", "gender", "age", "payer_code", "max_glu_serum", "A1Cresult",
                  "insulin", "change", "diabetesMed"]:
            out[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
            out[c] = pd.to_numeric(X[c], errors="coerce").astype("Int64").astype(str)
        out["specialty_grp"] = X["medical_specialty"].map(
            lambda s: "Missing" if pd.isna(s) else (s if s in TOP_SPECIALTIES else "Other"))
        for i in (1, 2, 3):
            out[f"diag_{i}_grp"] = X[f"diag_{i}"].map(icd9_group)
        return out[NUM_COLS + CAT_COLS]
