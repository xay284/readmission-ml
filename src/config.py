"""Central configuration: paths, seeds and column definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "diabetic_data.csv"
MODELS_DIR = ROOT / "models"
CHAMPION_DIR = MODELS_DIR / "champion"
VERSIONS_DIR = MODELS_DIR / "versions"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
CACHE_DIR = ROOT / ".cache"

SEED = 42
TARGET = "readmit_30d"          # 1 if readmitted < 30 days
GROUP = "patient_nbr"           # patients never straddle train/val/test
PSI_ALERT = 0.2                 # PSI above this = significant drift
PERF_DROP_TOL = 0.03            # absolute PR-AUC drop that triggers retraining

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

RAW_INPUT_COLS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "payer_code", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3",
    "number_diagnoses", "max_glu_serum", "A1Cresult",
] + MED_COLS + ["change", "diabetesMed"]

# Columns produced by FeatureEngineer (the model's real input space)
NUM_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_mid", "n_prior_visits", "n_meds_active", "n_meds_changed",
    "meds_per_day", "labs_per_day",
]
CAT_COLS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "specialty_grp", "payer_code", "diag_1_grp",
    "diag_2_grp", "diag_3_grp", "max_glu_serum", "A1Cresult", "insulin",
    "change", "diabetesMed",
]
