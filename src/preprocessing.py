"""Preprocessing pipeline: imputation, outlier capping, scaling, encoding, selection.
Everything is fitted inside the sklearn Pipeline, so cross-validation never leaks."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import NUM_COLS, CAT_COLS, SEED
from .features import FeatureEngineer


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap outliers at learned quantiles (fit on train only)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def build_preprocessor() -> Pipeline:
    num = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("win", Winsorizer()),
        ("sc", StandardScaler()),
    ])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False)),
    ])
    ct = ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)], verbose_feature_names_out=False)
    return Pipeline([
        ("fe", FeatureEngineer()),
        ("ct", ct),
        ("var", VarianceThreshold(1e-4)),
        # L1 selection: keeps features with non-zero coefficient
        ("sel", SelectFromModel(
            LogisticRegression(penalty="l1", solver="liblinear", C=0.05,
                               class_weight="balanced", random_state=SEED),
            threshold=1e-6)),
    ])


def feature_names(prep: Pipeline):
    names = prep.named_steps["ct"].get_feature_names_out()
    names = names[prep.named_steps["var"].get_support()]
    return names[prep.named_steps["sel"].get_support()]
