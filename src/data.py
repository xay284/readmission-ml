"""Data loading, cleaning and leakage-safe splitting."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import RAW_PATH, SEED, TARGET, GROUP, MED_COLS

# Discharge dispositions: expired / hospice -> cannot be readmitted
EXCLUDED_DISCHARGE = {"11", "13", "14", "19", "20", "21"}


def load_raw(path=RAW_PATH, synthetic=False, n=20000) -> pd.DataFrame:
    """Load the Diabetes 130-US Hospitals data (UCI id=296), caching it locally."""
    if synthetic:
        return make_synthetic(n)
    if path.exists():
        return pd.read_csv(path)
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=296)
        df = pd.concat([ds.data.ids, ds.data.features, ds.data.targets], axis=1)
    except Exception as e:  # network or package problem
        raise RuntimeError(
            "Could not download the dataset. Download diabetic_data.csv from "
            "https://archive.ics.uci.edu/dataset/296 and put it in data/raw/. "
            f"Original error: {e}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning + binary target. Statistical imputation is done later, in the pipeline."""
    df = df.replace({"?": np.nan}).copy()
    df = df[~df["discharge_disposition_id"].astype(str).isin(EXCLUDED_DISCHARGE)]
    df = df[df["gender"] != "Unknown/Invalid"]
    df[TARGET] = (df["readmitted"].astype(str) == "<30").astype(int)
    df = df.drop(columns=["readmitted", "weight", "encounter_id"], errors="ignore")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, seed=SEED):
    """70/15/15 split grouped by patient, so one patient never appears in two sets."""
    g1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tv_idx, te_idx = next(g1.split(df, groups=df[GROUP]))
    trval, test = df.iloc[tv_idx], df.iloc[te_idx]
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)
    tr_idx, va_idx = next(g2.split(trval, groups=trval[GROUP]))
    return trval.iloc[tr_idx], trval.iloc[va_idx], test


def make_synthetic(n=20000, seed=SEED) -> pd.DataFrame:
    """Small fake dataset with the real schema, for smoke tests only (NOT for results)."""
    rng = np.random.default_rng(seed)
    ch = lambda vals, p=None: rng.choice(np.array(vals, dtype=object), n, p=p)
    ages = [f"[{i}-{i+10})" for i in range(0, 100, 10)]
    inp = np.clip(rng.poisson(0.6, n), 0, 10)
    tih = rng.integers(1, 15, n)
    nmed = np.clip(rng.poisson(16, n), 1, 60)
    diag = ch(["250.01", "428", "414", "486", "585", "V57", np.nan, "786"])
    df = pd.DataFrame({
        "encounter_id": np.arange(n),
        "patient_nbr": rng.integers(1, int(n * 0.8), n),
        "race": ch(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", np.nan], [.7, .18, .03, .01, .02, .06]),
        "gender": ch(["Female", "Male"]),
        "age": ch(ages, [.005, .01, .02, .04, .09, .17, .22, .26, .17, .015]),
        "weight": np.nan,
        "admission_type_id": ch([1, 2, 3, 5], [.53, .18, .18, .11]),
        "discharge_disposition_id": ch([1, 2, 3, 6, 22], [.6, .03, .14, .13, .10]),
        "admission_source_id": ch([1, 4, 7], [.2, .1, .7]),
        "time_in_hospital": tih,
        "payer_code": ch(["MC", "HM", "SP", np.nan]),
        "medical_specialty": ch(["InternalMedicine", "Cardiology", "Surgery-General", "Nephrology", np.nan]),
        "num_lab_procedures": np.clip(rng.normal(43, 19, n), 1, 130).astype(int),
        "num_procedures": rng.integers(0, 7, n),
        "num_medications": nmed,
        "number_outpatient": np.clip(rng.poisson(0.4, n), 0, 20),
        "number_emergency": np.clip(rng.poisson(0.2, n), 0, 20),
        "number_inpatient": inp,
        "diag_1": diag, "diag_2": ch(["250", "401", "427", np.nan]), "diag_3": ch(["272", "E888", np.nan, "599"]),
        "number_diagnoses": rng.integers(1, 17, n),
        "max_glu_serum": ch([np.nan, "Norm", ">200", ">300"], [.9, .05, .03, .02]),
        "A1Cresult": ch([np.nan, "Norm", ">7", ">8"], [.8, .05, .05, .1]),
        "change": ch(["No", "Ch"]), "diabetesMed": ch(["Yes", "No"], [.77, .23]),
    })
    for m in MED_COLS:
        df[m] = "No"
    df["insulin"] = ch(["No", "Steady", "Up", "Down"], [.45, .30, .12, .13])
    df["metformin"] = ch(["No", "Steady", "Up", "Down"], [.8, .18, .01, .01])
    logit = -3.0 + 0.35 * inp + 0.05 * tih + 0.01 * nmed + 0.3 * (diag == "428") + 0.2 * (df["insulin"] == "Up")
    pos = rng.random(n) < 1 / (1 + np.exp(-logit))
    df["readmitted"] = np.where(pos, "<30", rng.choice(["NO", ">30"], n, p=[.55, .45]))
    return df
