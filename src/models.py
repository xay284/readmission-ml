"""Candidate algorithms and their hyper-parameter search spaces."""
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import SEED


def get_candidates(scale_pos_weight: float):
    """name -> (estimator, param space, 'grid' | 'random').
    Imbalance is handled with class weights (see train.imbalance_study for resampling comparison)."""
    return {
        "logreg": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
            {"model__C": [0.01, 0.1, 1, 10]}, "grid"),
        "tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [20, 50, 100]}, "grid"),
        "rf": (
            RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=SEED),
            {"model__max_depth": [6, 10, 16, None], "model__min_samples_leaf": [5, 10, 30],
             "model__max_features": ["sqrt", 0.3]}, "random"),
        "xgb": (
            XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=-1,
                          scale_pos_weight=scale_pos_weight, random_state=SEED),
            {"model__n_estimators": [200, 400], "model__max_depth": [3, 5, 7],
             "model__learning_rate": [0.03, 0.05, 0.1], "model__subsample": [0.7, 1.0],
             "model__colsample_bytree": [0.6, 0.8, 1.0], "model__min_child_weight": [1, 5, 10]}, "random"),
        "lgbm": (
            LGBMClassifier(class_weight="balanced", verbose=-1, n_jobs=-1, random_state=SEED),
            {"model__n_estimators": [200, 400], "model__num_leaves": [15, 31, 63],
             "model__learning_rate": [0.03, 0.05, 0.1], "model__min_child_samples": [20, 50, 100],
             "model__colsample_bytree": [0.6, 0.8, 1.0], "model__subsample": [0.7, 1.0],
             "model__subsample_freq": [1]}, "random"),
    }
