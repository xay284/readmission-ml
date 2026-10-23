"""Interactive dashboard.   streamlit run app/streamlit_app.py   (from project root)"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import CHAMPION_DIR, FIG_DIR, REPORTS_DIR, TARGET, PSI_ALERT
from src.data import clean, load_raw
from src.features import FeatureEngineer
from src.monitor import drift_report, performance_check
from src.schemas import AGES, PatientInput
from src.serving import load_champion, predict_frame

st.set_page_config(page_title="30-day readmission risk", layout="wide")
st.title("Hospital readmission risk (30 days)")
st.warning("Research prototype - **not** a clinically validated decision-making tool.")


@st.cache_data(show_spinner="Loading data...")
def get_data():
    return clean(load_raw())


@st.cache_resource
def get_model():
    return load_champion()


have_model = (CHAMPION_DIR / "pipeline.joblib").exists()
t_eda, t_perf, t_pred, t_xai, t_mon = st.tabs(["Data", "Model performance", "Predict", "Explainability", "Monitoring"])

with t_eda:
    try:
        df = get_data()
        c1, c2, c3 = st.columns(3)
        c1.metric("Encounters", f"{len(df):,}"); c2.metric("Patients", f"{df['patient_nbr'].nunique():,}")
        c3.metric("30-day readmission rate", f"{df[TARGET].mean():.1%}")
        X = FeatureEngineer().transform(df); X[TARGET] = df[TARGET].values
        col = st.selectbox("Feature", [c for c in X.columns if c != TARGET], index=8)
        if pd.api.types.is_numeric_dtype(X[col]):
            st.plotly_chart(px.histogram(X, x=col, color=TARGET, barmode="overlay", histnorm="probability density",
                                         opacity=.6), use_container_width=True)
        else:
            r = X.groupby(col, dropna=False)[TARGET].agg(["mean", "size"]).reset_index()
            r[col] = r[col].astype(str)
            st.plotly_chart(px.bar(r, x=col, y="mean", hover_data=["size"], labels={"mean": "readmission rate"}),
                            use_container_width=True)
        st.dataframe(df.head(200))
    except Exception as e:
        st.error(f"Could not load the dataset: {e}")

with t_perf:
    if not have_model:
        st.info("Train a model first: `python -m src.train`")
    else:
        meta = json.loads((CHAMPION_DIR / "meta.json").read_text())
        tm = meta["test_metrics"]
        st.subheader(f"Champion: {meta['model']} (threshold {meta['threshold']:.3f}, trained {meta['trained_at']})")
        cols = st.columns(5)
        for c, k in zip(cols, ["recall", "precision", "f2", "roc_auc", "pr_auc"]):
            c.metric(k.upper().replace("_", "-"), f"{tm[k]:.3f}")
        comp = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
        st.dataframe(comp.round(3), use_container_width=True)
        st.plotly_chart(px.bar(comp, x="model", y=["val_pr_auc", "test_pr_auc"], barmode="group"), use_container_width=True)
        a, b = st.columns(2)
        for name, holder in [("roc_curves", a), ("pr_curves", b), ("calibration", a), ("confusion_matrix", b)]:
            if (FIG_DIR / f"{name}.png").exists():
                holder.image(str(FIG_DIR / f"{name}.png"))
        if (REPORTS_DIR / "subgroup_analysis.csv").exists():
            st.subheader("Subgroup error analysis (test)")
            st.dataframe(pd.read_csv(REPORTS_DIR / "subgroup_analysis.csv").round(3), use_container_width=True)
        if (REPORTS_DIR / "imbalance_study.csv").exists():
            st.subheader("Imbalance handling study (validation)")
            st.dataframe(pd.read_csv(REPORTS_DIR / "imbalance_study.csv").round(3), use_container_width=True)

with t_pred:
    if not have_model:
        st.info("Train a model first: `python -m src.train`")
    else:
        pipe, meta = get_model()
        with st.form("patient"):
            a, b, c = st.columns(3)
            age = a.selectbox("Age", AGES, index=6); gender = a.selectbox("Gender", ["Female", "Male"])
            race = a.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
            tih = b.slider("Days in hospital", 1, 14, 4); nmed = b.slider("Medications", 0, 80, 16)
            nlab = b.slider("Lab procedures", 0, 130, 43)
            nproc = c.slider("Procedures", 0, 6, 1); ndiag = c.slider("Diagnoses", 1, 16, 7)
            nin = c.number_input("Inpatient visits (prior year)", 0, 30, 0)
            d, e, f = st.columns(3)
            nout = d.number_input("Outpatient visits", 0, 50, 0); nem = d.number_input("Emergency visits", 0, 50, 0)
            diag1 = e.text_input("Primary diagnosis (ICD-9)", "428"); a1c = e.selectbox("A1C result", ["None", "Norm", ">7", ">8"])
            ins = f.selectbox("Insulin", ["No", "Steady", "Up", "Down"]); chg = f.selectbox("Medication change", ["No", "Ch"])
            if st.form_submit_button("Estimate risk"):
                try:
                    pt = PatientInput(age=age, gender=gender, race=race, time_in_hospital=tih, num_medications=nmed,
                                      num_lab_procedures=nlab, num_procedures=nproc, number_diagnoses=ndiag,
                                      number_inpatient=nin, number_outpatient=nout, number_emergency=nem,
                                      diag_1=diag1 or None, A1Cresult=a1c, insulin=ins, change=chg)
                    r = predict_frame(pipe, meta, pt.to_frame()).iloc[0]
                    st.session_state["result"] = (float(r.probability), r.risk_band)
                except Exception as err:
                    st.error(f"Invalid input: {err}")
        if "result" in st.session_state:
            p, band = st.session_state["result"]
            fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"},
                                         gauge={"axis": {"range": [0, 100]}, "threshold": {"line": {"color": "red", "width": 3},
                                                "value": meta["threshold"] * 100}}))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Risk band: {band}** (alert threshold {meta['threshold']:.1%}). "
                        "Probabilities come from a class-weighted model and are not calibrated absolute risks.")

with t_xai:
    shown = False
    for name, cap in [("permutation_importance", "Permutation importance"), ("shap_summary", "SHAP summary"),
                      ("shap_bar", "Mean |SHAP|")]:
        if (FIG_DIR / f"{name}.png").exists():
            st.image(str(FIG_DIR / f"{name}.png"), caption=cap); shown = True
    if not shown:
        st.info("Run `python -m src.explain` to generate these.")

with t_mon:
    if not have_model:
        st.info("Train a model first.")
    else:
        pipe, meta = get_model()
        st.write("Upload a new batch of encounters (same schema as the raw data; add `readmitted` to also track performance).")
        up = st.file_uploader("CSV", type="csv")
        if up:
            batch = pd.read_csv(up)
            ref = pd.read_csv(CHAMPION_DIR / "reference_features.csv")
            rep = drift_report(ref, FeatureEngineer().transform(batch))
            st.metric("Features with PSI > %.1f" % PSI_ALERT, int(rep["drift"].sum()))
            st.plotly_chart(px.bar(rep, x="feature", y="psi", color="drift"), use_container_width=True)
            st.dataframe(rep.round(3))
            if "readmitted" in batch.columns:
                perf = performance_check(pipe, meta, batch)
                st.json(perf)
                if perf["degraded"]:
                    st.error("Performance drop detected -> run `python -m src.retrain --new <file>`")
        if (REPORTS_DIR / "performance_log.csv").exists():
            st.subheader("Performance log"); st.dataframe(pd.read_csv(REPORTS_DIR / "performance_log.csv"))
