import numpy as np
import pandas as pd
import pytest

from src.config import NUM_COLS, CAT_COLS
from src.data import clean, make_synthetic, split
from src.features import FeatureEngineer, icd9_group
from src.monitor import drift_report
from src.schemas import PatientInput


def test_icd9_groups():
    assert icd9_group("250.83") == "Diabetes" and icd9_group("428") == "Circulatory"
    assert icd9_group("V57") == "Other" and icd9_group(np.nan) == "Missing"


def test_split_has_no_patient_overlap():
    tr, va, te = split(clean(make_synthetic(3000)))
    s = [set(x["patient_nbr"]) for x in (tr, va, te)]
    assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])


def test_feature_engineer_on_partial_input():
    X = PatientInput().to_frame()
    out = FeatureEngineer().transform(X)
    assert list(out.columns) == NUM_COLS + CAT_COLS and len(out) == 1


def test_input_validation():
    with pytest.raises(Exception):
        PatientInput(time_in_hospital=99)


def test_no_drift_on_same_distribution():
    f = FeatureEngineer().transform(clean(make_synthetic(4000)))
    rep = drift_report(f.iloc[:2000], f.iloc[2000:])
    assert rep["drift"].sum() == 0
