"""Optional prediction API.   uvicorn api.main:app --reload   (from project root)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import List

from fastapi import FastAPI, HTTPException

from src.schemas import PatientInput
from src.serving import load_champion, predict_frame

app = FastAPI(title="30-day readmission risk API", version="1.0",
              description="Research prototype - not a clinically validated decision tool.")
try:
    PIPE, META = load_champion()
except Exception:
    PIPE = META = None


def _need_model():
    if PIPE is None:
        raise HTTPException(503, "No trained model found. Run `python -m src.train` first.")


@app.get("/health")
def health():
    return {"status": "ok" if PIPE is not None else "no_model"}


@app.get("/model-info")
def model_info():
    _need_model()
    return {k: META[k] for k in ("model", "threshold", "trained_at", "test_metrics", "disclaimer")}


@app.post("/predict")
def predict(patient: PatientInput):
    _need_model()
    r = predict_frame(PIPE, META, patient.to_frame()).iloc[0]
    return {"probability": float(r.probability), "high_risk": bool(r.high_risk), "risk_band": r.risk_band,
            "threshold": META["threshold"], "model": META["model"], "disclaimer": META["disclaimer"]}


@app.post("/predict/batch")
def predict_batch(patients: List[PatientInput]):
    _need_model()
    import pandas as pd
    df = pd.concat([p.to_frame() for p in patients], ignore_index=True)
    return predict_frame(PIPE, META, df).to_dict(orient="records")
