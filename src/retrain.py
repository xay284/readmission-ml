"""Retraining workflow with champion/challenger promotion.

    python -m src.retrain --new data/new_labeled.csv [--min-gain 0.0]
New file must follow the raw schema (including `readmitted`).
"""
import argparse
from datetime import datetime

import joblib
import pandas as pd

from .config import CHAMPION_DIR, VERSIONS_DIR, RAW_PATH
from .data import clean, load_raw
from .evaluate import compute_metrics
from .serving import load_champion
from .train import promote, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True); ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--n-iter", type=int, default=15); ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--synthetic", action="store_true", help="base data = synthetic (smoke test)")
    a = ap.parse_args()

    base = load_raw(synthetic=a.synthetic)
    combined = pd.concat([base, pd.read_csv(a.new)], ignore_index=True)
    out = VERSIONS_DIR / datetime.now().strftime("v%Y%m%d_%H%M%S")
    chal_meta = run(clean(combined), out, a.n_iter, a.cv, study=False)

    champ_pipe, champ_meta = load_champion()
    test = pd.read_csv(out / "test.csv")           # challenger's untouched test set
    p = champ_pipe.predict_proba(test)[:, 1]
    champ_pr = compute_metrics(test["readmit_30d"], p, champ_meta["threshold"])["pr_auc"]
    chal_pr = chal_meta["test_metrics"]["pr_auc"]
    print(f"Champion PR-AUC={champ_pr:.4f} | Challenger PR-AUC={chal_pr:.4f}")
    if chal_pr >= champ_pr + a.min_gain:
        promote(out)
    else:
        print("Challenger not better -> champion kept (new version stored for audit).")


if __name__ == "__main__":
    main()
