# 30-day hospital readmission risk — end-to-end ML project

Predicts whether a diabetic patient is readmitted **< 30 days** after discharge, using the
[Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296) dataset (UCI, ~100k encounters).

> **Research prototype — not a clinically validated decision-making tool.**

## Quick start
```bash
pip install -r requirements.txt
python -m src.eda                  # EDA figures -> reports/figures
python -m src.train                # tuning + ensembles + evaluation + MLflow  (first run downloads the data)
python -m src.clustering           # K-Means + hierarchical patient subgroups
python -m src.explain              # permutation importance + SHAP
streamlit run app/streamlit_app.py # dashboard
uvicorn api.main:app --reload      # optional API (docs at /docs)
mlflow ui --backend-store-uri sqlite:///mlflow.db
pytest                             # unit tests
```
Faster first run: `python -m src.train --sample 30000 --n-iter 8 --cv 3`.
If the automatic download fails, put `diabetic_data.csv` in `data/raw/`.
`--synthetic` generates fake data with the same schema: **smoke tests only, never for results**.

## Pipeline
| Stage | Where | Notes |
|---|---|---|
| Cleaning, target | `src/data.py` | `?`→NaN, drop `weight`, exclude expired/hospice discharges, target = `readmitted == "<30"` |
| Leak-free split | `src/data.py` | 70/15/15 **grouped by `patient_nbr`** (no patient in two sets) |
| Features | `src/features.py` | ICD-9 → clinical groups, prior-visit count, meds active/changed, per-day rates, age midpoint |
| Preprocessing | `src/preprocessing.py` | median/constant imputation, winsorising (1–99%), scaling, one-hot, variance filter, L1 feature selection — **all fitted inside the pipeline** (no CV leakage) |
| Models | `src/models.py` | LogReg (baseline), Decision Tree, Random Forest, XGBoost, LightGBM |
| Tuning | `src/train.py` | Grid (LR, tree) / Random search (RF, XGB, LGBM), `StratifiedGroupKFold`, scoring = PR-AUC |
| Imbalance | `src/train.py` | class weights for all models + a study comparing none / weights / SMOTE / under-sampling → `reports/imbalance_study.csv`; decision threshold tuned on validation for F2 (recall-weighted) |
| Ensembles | `src/train.py` | soft Voting and Stacking over LR/RF/XGB/LGBM |
| Evaluation | `src/evaluate.py` | accuracy, precision, recall, F1/F2, ROC-AUC, PR-AUC, confusion matrix, calibration + Brier, subgroup analysis (race, gender, age) |
| Unsupervised | `src/clustering.py` | PCA + K-Means (silhouette for k), Ward hierarchical, cluster profiles incl. readmission rate |
| XAI | `src/explain.py` | permutation importance (raw features) + SHAP |
| Serving | `app/`, `api/`, `src/schemas.py` | Streamlit dashboard, FastAPI, Pydantic input validation |
| MLOps | `src/monitor.py`, `src/retrain.py` | PSI/KS drift, performance log, champion/challenger retraining, versioned models in `models/versions/`, MLflow tracking |

Model selection uses **validation** PR-AUC only; the test set is touched once at the end.

## MLOps workflow
```bash
python -m src.monitor --batch new_encounters.csv     # drift (PSI > 0.2) + performance if `readmitted` present
python -m src.retrain --new new_labeled.csv          # retrains on old+new, promotes only if better on its test set
```
Retraining is recommended when ≥ 3 features drift or PR-AUC drops by > 0.03 (`src/config.py`).

## Limitations (read before using the numbers)
- Expect modest discrimination on this dataset (ROC-AUC typically ~0.63–0.69); readmission is hard to predict from administrative data.
- Predicted probabilities come from class-weighted models: they rank risk but are **not calibrated absolute risks** (see calibration plot; isotonic calibration is a natural extension).
- Data are 1999–2008 US hospitals; generalisation to other settings is unknown.
- Check `reports/subgroup_analysis.csv` for recall/FPR gaps across race, gender and age before drawing fairness conclusions.
- The retrain comparison evaluates the old champion on the new test split, which may include patients it saw in training, a conservative bias in the champion's favour.
