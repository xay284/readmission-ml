"""Explainability for the champion: permutation importance + SHAP.
    python -m src.explain"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from .config import CHAMPION_DIR, FIG_DIR, REPORTS_DIR, RAW_INPUT_COLS, SEED, TARGET
from .preprocessing import feature_names
from .serving import load_champion


def main(n=3000):
    pipe, meta = load_champion()
    test = pd.read_csv(CHAMPION_DIR / "test.csv").sample(n, random_state=SEED, replace=False) \
        if len(pd.read_csv(CHAMPION_DIR / "test.csv")) > n else pd.read_csv(CHAMPION_DIR / "test.csv")
    X, y = test.drop(columns=[TARGET]), test[TARGET]

    # 1) Permutation importance on RAW columns (model-agnostic, measured in PR-AUC loss)
    cols = [c for c in RAW_INPUT_COLS if c in X.columns]
    pi = permutation_importance(pipe, X[cols], y, scoring="average_precision", n_repeats=5,
                                random_state=SEED, n_jobs=1)
    pim = pd.DataFrame({"feature": cols, "importance": pi.importances_mean, "std": pi.importances_std}) \
        .sort_values("importance", ascending=False)
    pim.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    top = pim.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5)); ax.barh(top["feature"], top["importance"], xerr=top["std"])
    ax.set_xlabel("Drop in PR-AUC when shuffled"); ax.set_title("Permutation importance (test)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "permutation_importance.png", dpi=130); plt.close(fig)

    # 2) SHAP on the (base) model, in transformed feature space
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    if hasattr(model, "named_estimators_"):       # ensemble -> explain its LightGBM base learner
        model = model.named_estimators_["lgbm"]
        print("Champion is an ensemble: SHAP computed on its LightGBM base learner.")
    Xt = pd.DataFrame(prep.transform(X), columns=feature_names(prep))
    if hasattr(model, "feature_importances_"):
        sv = shap.TreeExplainer(model).shap_values(Xt)
    else:
        sv = shap.LinearExplainer(model, Xt).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    for kind, name in (("dot", "shap_summary"), ("bar", "shap_bar")):
        plt.figure(); shap.summary_plot(sv, Xt, plot_type=kind, max_display=15, show=False)
        plt.tight_layout(); plt.savefig(FIG_DIR / f"{name}.png", dpi=130); plt.close()
    pd.DataFrame({"feature": Xt.columns, "mean_abs_shap": np.abs(sv).mean(0)}) \
        .sort_values("mean_abs_shap", ascending=False).to_csv(REPORTS_DIR / "shap_importance.csv", index=False)
    print(pim.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
