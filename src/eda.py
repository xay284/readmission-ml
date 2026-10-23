"""Exploratory data analysis -> reports/figures/eda_*.png + reports/eda_summary.json"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIG_DIR, REPORTS_DIR, TARGET
from .data import clean, load_raw
from .features import FeatureEngineer
import sys


def main(synthetic=False):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw(synthetic=synthetic)
    df = clean(raw)
    X = FeatureEngineer().transform(df)
    X[TARGET] = df[TARGET].values
    miss = raw.replace("?", pd.NA).isna().mean().sort_values(ascending=False)
    summary = dict(rows=len(raw), rows_after_cleaning=len(df), unique_patients=int(df["patient_nbr"].nunique()),
                   readmit_30d_rate=float(df[TARGET].mean()), missing_rate_top=miss.head(8).round(3).to_dict())
    (REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4)); miss.head(12)[::-1].plot.barh(ax=ax)
    ax.set_title("Missing rate (top 12)"); fig.tight_layout(); fig.savefig(FIG_DIR / "eda_missing.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(X.select_dtypes("number").corr(), cmap="coolwarm", center=0, ax=ax, annot=False)
    ax.set_title("Correlation (numeric features + target)"); fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_correlation.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, c in zip(axes.ravel(), ["age", "number_inpatient", "diag_1_grp", "A1Cresult"]):
        r = X.groupby(c, dropna=False)[TARGET].mean().sort_index()
        r.plot.bar(ax=ax); ax.set_ylabel("30-day readmission rate"); ax.set_title(f"By {c}")
        ax.axhline(X[TARGET].mean(), color="k", ls="--", lw=.8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_rates.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, c in zip(axes, ["time_in_hospital", "num_medications", "num_lab_procedures"]):
        sns.boxplot(data=X, x=TARGET, y=c, ax=ax, showfliers=True)
    fig.suptitle("Distributions & outliers by outcome"); fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_distributions.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main(synthetic="--synthetic" in sys.argv)
