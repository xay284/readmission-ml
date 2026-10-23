"""Unsupervised patient subgroups: K-Means + hierarchical clustering.
    python -m src.clustering [--k 4] [--synthetic]"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .config import FIG_DIR, REPORTS_DIR, SEED, TARGET
from .data import clean, load_raw, split
from .preprocessing import build_preprocessor
from .features import FeatureEngineer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true"); a = ap.parse_args()
    df = clean(load_raw(synthetic=a.synthetic))
    tr, _, te = split(df)
    # unsupervised: only imputation / scaling / encoding (no supervised selector)
    prep = build_preprocessor()[:2].fit(tr)
    Z = PCA(n_components=10, random_state=SEED).fit(prep.transform(tr))
    Ztr, Zte = Z.transform(prep.transform(tr)), Z.transform(prep.transform(te))
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(Ztr), min(8000, len(Ztr)), replace=False)

    ks = range(2, 9)
    sil = [silhouette_score(Ztr[sub], KMeans(k, n_init=5, random_state=SEED).fit_predict(Ztr[sub])) for k in ks]
    k = a.k or list(ks)[int(np.argmax(sil))]
    fig, ax = plt.subplots(figsize=(5, 3.5)); ax.plot(list(ks), sil, marker="o")
    ax.set(xlabel="k", ylabel="Silhouette", title="K-Means: choosing k"); fig.tight_layout()
    fig.savefig(FIG_DIR / "cluster_silhouette.png", dpi=130); plt.close(fig)

    km = KMeans(k, n_init=10, random_state=SEED).fit(Ztr)
    lab_te = km.predict(Zte)
    X = FeatureEngineer().transform(te); X[TARGET] = te[TARGET].values; X["cluster"] = lab_te
    prof = X.groupby("cluster").agg(
        size=("cluster", "size"), readmit_rate=(TARGET, "mean"), age=("age_mid", "mean"),
        days=("time_in_hospital", "mean"), meds=("num_medications", "mean"),
        prior_inpatient=("number_inpatient", "mean"), diagnoses=("number_diagnoses", "mean")).round(3)
    prof.to_csv(REPORTS_DIR / "cluster_profiles.csv"); print(f"K-Means k={k}\n", prof)

    fig, ax = plt.subplots(figsize=(6, 5))
    s = ax.scatter(Zte[:, 0], Zte[:, 1], c=lab_te, s=4, cmap="tab10", alpha=.6)
    ax.set(xlabel="PC1", ylabel="PC2", title=f"Patient clusters (test, k={k})"); fig.tight_layout()
    fig.savefig(FIG_DIR / "cluster_pca.png", dpi=130); plt.close(fig)

    # hierarchical (Ward) on a 3000-row sample: O(n^2) memory
    hs = rng.choice(len(Ztr), min(3000, len(Ztr)), replace=False)
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Ztr[hs])
    ari = adjusted_rand_score(hc, km.predict(Ztr[hs]))
    print(f"Hierarchical vs K-Means agreement (ARI) = {ari:.3f}")
    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(linkage(Ztr[hs][:400], "ward"), truncate_mode="lastp", p=20, ax=ax, no_labels=True)
    ax.set_title("Hierarchical clustering (Ward, 400-patient sample)"); fig.tight_layout()
    fig.savefig(FIG_DIR / "cluster_dendrogram.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
