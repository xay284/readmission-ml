"""Metrics, thresholds, plots, calibration and subgroup error analysis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, fbeta_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)


def best_threshold(y, p, beta=2.0) -> float:
    """Threshold maximising F-beta (beta=2 favours recall: a missed readmission costs more than a false alarm)."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta**2) * prec * rec / np.clip(beta**2 * prec + rec, 1e-12, None)
    return float(thr[int(np.nanargmax(f))])


def compute_metrics(y, p, thr=0.5) -> dict:
    y = np.asarray(y)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        threshold=float(thr),
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        f2=fbeta_score(y, pred, beta=2, zero_division=0),
        roc_auc=roc_auc_score(y, p),
        pr_auc=average_precision_score(y, p),
        brier=brier_score_loss(y, p),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def plot_confusion(y, p, thr, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y, (p >= thr).astype(int), ax=ax, cmap="Blues",
                                            display_labels=["No readmit", "Readmit <30d"], colorbar=False)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_curves(y, probs: dict, roc_path, pr_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for n, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{n} ({roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(roc_path, dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for n, p in probs.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=f"{n} ({average_precision_score(y, p):.3f})")
    ax.axhline(np.mean(y), color="k", ls="--", lw=.8)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (test)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(pr_path, dpi=130); plt.close(fig)


def plot_calibration(y, probs: dict, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for n, p in probs.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=3, label=f"{n} (Brier {brier_score_loss(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction", title="Calibration (test)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def subgroup_report(df, y, p, thr, cols=("race", "gender", "age"), min_n=50) -> pd.DataFrame:
    """Per-subgroup performance, to expose disparities (recall / FPR gaps)."""
    rows = []
    y = np.asarray(y)
    for c in cols:
        g = df[c].fillna("Missing").astype(str).values
        for v in np.unique(g):
            m = g == v
            if m.sum() < min_n:
                continue
            yy, pp = y[m], p[m]
            pred = (pp >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
            rows.append(dict(
                attribute=c, group=v, n=int(m.sum()), prevalence=yy.mean(),
                recall=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1),
                precision=tp / max(tp + fp, 1), selection_rate=pred.mean(),
                roc_auc=roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan,
            ))
    return pd.DataFrame(rows)
