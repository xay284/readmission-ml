"""End-to-end supervised training: tuning, imbalance study, ensembles, evaluation, packaging, MLflow.

Usage (from project root):
    python -m src.train                       # real data (downloads UCI dataset on first run)
    python -m src.train --sample 30000 --n-iter 8 --cv 3     # faster
    python -m src.train --synthetic           # smoke test only
"""
import argparse, json, shutil, time, warnings
from contextlib import nullcontext
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .config import *
from .data import clean, load_raw, split
from .evaluate import (best_threshold, compute_metrics, plot_calibration, plot_confusion,
                       plot_curves, subgroup_report)
from .features import FeatureEngineer
from .models import get_candidates
from .preprocessing import build_preprocessor, feature_names

warnings.filterwarnings("ignore")

try:
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("readmission-30d")
except Exception:  # MLflow is optional
    mlflow = None


def _run(name, nested=True):
    return mlflow.start_run(run_name=name, nested=nested) if mlflow else nullcontext()


def imbalance_study(Xtr, ytr, Xva, yva, beta):
    """Compare: no handling vs class weights vs SMOTE vs random under-sampling."""
    rows = []
    makers = {
        "logreg": lambda cw: LogisticRegression(max_iter=2000, class_weight=cw, random_state=SEED),
        "lgbm": lambda cw: LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight=cw,
                                          verbose=-1, n_jobs=-1, random_state=SEED),
    }
    strategies = {
        "none": ("passthrough", None), "class_weight": ("passthrough", "balanced"),
        "smote": (SMOTE(random_state=SEED), None), "undersample": (RandomUnderSampler(random_state=SEED), None),
    }
    for mname, mk in makers.items():
        for sname, (sampler, cw) in strategies.items():
            # imblearn forbids nested pipelines -> flatten the preprocessing steps
            pipe = ImbPipeline(list(build_preprocessor().steps) + [("sampler", sampler), ("model", mk(cw))],
                               memory=str(CACHE_DIR))
            pipe.fit(Xtr, ytr)
            p = pipe.predict_proba(Xva)[:, 1]
            thr = best_threshold(yva, p, beta)
            m05, mt = compute_metrics(yva, p, 0.5), compute_metrics(yva, p, thr)
            rows.append(dict(model=mname, strategy=sname, pr_auc=m05["pr_auc"], roc_auc=m05["roc_auc"],
                             recall_at_0_5=m05["recall"], precision_at_0_5=m05["precision"],
                             tuned_threshold=thr, recall_tuned=mt["recall"], precision_tuned=mt["precision"],
                             f2_tuned=mt["f2"]))
            print(f"  imbalance | {mname:6s} {sname:12s} PR-AUC={m05['pr_auc']:.3f} recall@0.5={m05['recall']:.2f}")
    return pd.DataFrame(rows)


def run(df, out_dir, n_iter=15, cv_folds=3, beta=2.0, study=True, sample=None, seed=SEED):
    """Train everything on `df` (cleaned data). Saves artefacts in out_dir and returns the metadata dict."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=seed)
    tr, va, te = split(df, seed)
    feats = [c for c in df.columns if c not in (TARGET,)]
    Xtr, ytr, gtr = tr[feats], tr[TARGET], tr[GROUP]
    Xva, yva, Xte, yte = va[feats], va[TARGET], te[feats], te[TARGET]
    print(f"train={len(tr)} val={len(va)} test={len(te)} | positive rate={ytr.mean():.3%}")
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    parent = mlflow.start_run(run_name=f"training-{datetime.now():%Y%m%d-%H%M}") if mlflow else nullcontext()
    with parent:
        if mlflow:
            mlflow.log_params(dict(n_train=len(tr), n_val=len(va), n_test=len(te), n_iter=n_iter,
                                   cv_folds=cv_folds, beta=beta, positive_rate=float(ytr.mean())))
        if study:
            print("Imbalance study (resampling vs class weights)...")
            imbalance_study(Xtr, ytr, Xva, yva, beta).to_csv(REPORTS_DIR / "imbalance_study.csv", index=False)

        # ---- 1. Tuned single models -------------------------------------------------
        cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        models = {}
        for name, (clf, space, kind) in get_candidates(spw).items():
            print(f"Tuning {name} ({kind} search)...")
            pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)], memory=str(CACHE_DIR))
            if kind == "grid":
                s = GridSearchCV(pipe, space, scoring="average_precision", cv=cv, n_jobs=1)
            else:
                s = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="average_precision",
                                       cv=cv, n_jobs=1, random_state=seed)
            s.fit(Xtr, ytr, groups=gtr)
            models[name] = dict(pipe=s.best_estimator_, cv_pr_auc=float(s.best_score_), params=s.best_params_)

        # ---- 2. Ensembles (voting + stacking) on top of the tuned models ----------------
        print("Fitting ensembles...")
        prep = build_preprocessor().fit(Xtr, ytr)
        Xt = prep.transform(Xtr)
        base = [(n, clone(models[n]["pipe"].named_steps["model"])) for n in ("logreg", "rf", "xgb", "lgbm")]
        folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(Xt, ytr, gtr))
        ens = {
            "voting": VotingClassifier(base, voting="soft"),
            "stacking": StackingClassifier(base, final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
                                           cv=folds, stack_method="predict_proba"),
        }
        for n, e in ens.items():
            e.fit(Xt, ytr)
            models[n] = dict(pipe=Pipeline([("prep", prep), ("model", e)]), cv_pr_auc=np.nan, params={})

        # ---- 3. Validation: threshold tuning + model selection -----------------------------
        rows, val_p, test_p = [], {}, {}
        for n, d in models.items():
            pv = d["pipe"].predict_proba(Xva)[:, 1]
            thr = best_threshold(yva, pv, beta)
            d["thr"] = thr
            pt = d["pipe"].predict_proba(Xte)[:, 1]
            val_p[n], test_p[n] = pv, pt
            mv, mt = compute_metrics(yva, pv, thr), compute_metrics(yte, pt, thr)
            mt05 = compute_metrics(yte, pt, 0.5)
            rows.append({"model": n, "cv_pr_auc": d["cv_pr_auc"],
                         **{f"val_{k}": v for k, v in mv.items() if k in ("pr_auc", "roc_auc", "recall", "precision", "f2")},
                         **{f"test_{k}": v for k, v in mt.items() if k not in ("tn", "fp", "fn", "tp")},
                         "test_recall_at_0_5": mt05["recall"], "test_precision_at_0_5": mt05["precision"]})
            with _run(n):
                if mlflow:
                    mlflow.log_params({k.replace("model__", ""): str(v) for k, v in d["params"].items()})
                    mlflow.log_metrics({f"val_{k}": v for k, v in mv.items() if isinstance(v, float)})
                    mlflow.log_metric("threshold", thr)
        table = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
        table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
        print(table[["model", "val_pr_auc", "val_roc_auc", "test_pr_auc", "test_roc_auc", "test_recall", "test_precision"]]
              .round(3).to_string(index=False))

        champ = table.iloc[0]["model"]
        d = models[champ]
        print(f"Champion (best validation PR-AUC): {champ}  threshold={d['thr']:.3f}")

        # ---- 4. Final evaluation of the champion on the untouched test set ---------------------
        pt, thr = test_p[champ], d["thr"]
        m_test = compute_metrics(yte, pt, thr)
        plot_confusion(yte, pt, thr, f"{champ} (test, thr={thr:.2f})", FIG_DIR / "confusion_matrix.png")
        plot_curves(yte, test_p, FIG_DIR / "roc_curves.png", FIG_DIR / "pr_curves.png")
        plot_calibration(yte, test_p, FIG_DIR / "calibration.png")
        subgroup_report(te, yte, pt, thr).to_csv(REPORTS_DIR / "subgroup_analysis.csv", index=False)

        # ---- 5. Package pipeline + metadata + drift reference ---------------------------------------
        joblib.dump(d["pipe"], out_dir / "pipeline.joblib")
        te.to_csv(out_dir / "test.csv", index=False)
        FeatureEngineer().transform(Xtr.sample(min(5000, len(Xtr)), random_state=seed)) \
            .to_csv(out_dir / "reference_features.csv", index=False)
        meta = dict(model=champ, threshold=thr, beta=beta, test_metrics=m_test,
                    val_pr_auc=float(table.iloc[0]["val_pr_auc"]), n_train=len(tr), n_val=len(va), n_test=len(te),
                    positive_rate=float(ytr.mean()), trained_at=datetime.now().isoformat(timespec="seconds"),
                    sklearn_version=sklearn.__version__, best_params={k: str(v) for k, v in d["params"].items()},
                    disclaimer="Research prototype - not clinically validated.")
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        (REPORTS_DIR / "metrics.json").write_text((out_dir / "meta.json").read_text())
        if mlflow:
            mlflow.log_metrics({f"test_{k}": v for k, v in m_test.items() if isinstance(v, float)})
            mlflow.log_param("champion", champ)
            mlflow.log_artifact(str(out_dir / "pipeline.joblib")); mlflow.log_artifact(str(out_dir / "meta.json"))
            mlflow.log_artifact(str(REPORTS_DIR / "model_comparison.csv"))
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    print(f"Done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")
    return meta


def promote(version_dir):
    """Make a trained version the serving 'champion'."""
    shutil.rmtree(CHAMPION_DIR, ignore_errors=True)
    shutil.copytree(version_dir, CHAMPION_DIR)
    print(f"Promoted {version_dir.name} -> {CHAMPION_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--sample", type=int, default=None, help="subsample rows for faster runs")
    ap.add_argument("--n-iter", type=int, default=15); ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--beta", type=float, default=2.0); ap.add_argument("--skip-study", action="store_true")
    a = ap.parse_args()
    df = clean(load_raw(synthetic=a.synthetic, n=a.n))
    out = VERSIONS_DIR / datetime.now().strftime("v%Y%m%d_%H%M%S")
    run(df, out, a.n_iter, a.cv, a.beta, not a.skip_study, a.sample)
    promote(out)


if __name__ == "__main__":
    main()
