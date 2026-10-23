"""Data-drift (PSI + KS) and performance tracking.

    python -m src.monitor --batch path/to/new_encounters.csv
If the batch has a `readmitted` column, performance is also logged to reports/performance_log.csv.
"""
import argparse, json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import CHAMPION_DIR, NUM_COLS, CAT_COLS, PSI_ALERT, PERF_DROP_TOL, REPORTS_DIR
from .data import clean
from .evaluate import compute_metrics
from .features import FeatureEngineer
from .serving import load_champion


def _psi(e, a, eps=1e-4):
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(ref, cur, bins=10):
    ref, cur = ref.dropna().astype(float), cur.dropna().astype(float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3 or len(cur) == 0:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    return _psi(np.histogram(ref, edges)[0] / len(ref), np.histogram(cur, edges)[0] / len(cur))


def psi_categorical(ref, cur):
    ref, cur = ref.fillna("Missing").astype(str), cur.fillna("Missing").astype(str)
    cats = sorted(set(ref) | set(cur))
    e = ref.value_counts(normalize=True).reindex(cats, fill_value=0).values
    a = cur.value_counts(normalize=True).reindex(cats, fill_value=0).values
    return _psi(e, a)


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    """ref/cur are engineered feature frames. PSI > 0.2 => significant drift."""
    rows = []
    for c in NUM_COLS:
        rows.append(dict(feature=c, kind="numeric", psi=psi_numeric(ref[c], cur[c]),
                         ks_pvalue=float(ks_2samp(ref[c].dropna(), cur[c].dropna()).pvalue)))
    for c in CAT_COLS:
        rows.append(dict(feature=c, kind="categorical", psi=psi_categorical(ref[c], cur[c]), ks_pvalue=np.nan))
    out = pd.DataFrame(rows).sort_values("psi", ascending=False)
    out["drift"] = out["psi"] > PSI_ALERT
    return out


def performance_check(pipe, meta, labeled_raw: pd.DataFrame) -> dict:
    df = clean(labeled_raw)
    p = pipe.predict_proba(df)[:, 1]
    m = compute_metrics(df["readmit_30d"], p, meta["threshold"])
    base = meta["test_metrics"]["pr_auc"]
    res = dict(timestamp=datetime.now().isoformat(timespec="seconds"), n=len(df),
               pr_auc=m["pr_auc"], roc_auc=m["roc_auc"], recall=m["recall"], precision=m["precision"],
               baseline_pr_auc=base, degraded=bool(base - m["pr_auc"] > PERF_DROP_TOL))
    log = REPORTS_DIR / "performance_log.csv"
    pd.DataFrame([res]).to_csv(log, mode="a", header=not log.exists(), index=False)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", required=True)
    a = ap.parse_args()
    pipe, meta = load_champion()
    batch = pd.read_csv(a.batch)
    ref = pd.read_csv(CHAMPION_DIR / "reference_features.csv")
    rep = drift_report(ref, FeatureEngineer().transform(batch))
    rep.to_csv(REPORTS_DIR / "drift_report.csv", index=False)
    n_drift = int(rep["drift"].sum())
    print(rep.head(10).round(3).to_string(index=False))
    print(f"\n{n_drift} feature(s) with PSI > {PSI_ALERT}")
    retrain = n_drift >= 3
    if "readmitted" in batch.columns:
        perf = performance_check(pipe, meta, batch)
        print(json.dumps(perf, indent=2))
        retrain = retrain or perf["degraded"]
    print("RETRAINING RECOMMENDED" if retrain else "Model looks stable")


if __name__ == "__main__":
    main()
