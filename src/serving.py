"""Load the champion pipeline and score data (used by API, dashboard, monitoring)."""
import json

import joblib
import pandas as pd

from .config import CHAMPION_DIR


def load_champion(model_dir=CHAMPION_DIR):
    pipe = joblib.load(model_dir / "pipeline.joblib")
    meta = json.loads((model_dir / "meta.json").read_text())
    return pipe, meta


def risk_band(p: float, thr: float) -> str:
    return "High" if p >= thr else ("Medium" if p >= 0.6 * thr else "Low")


def predict_frame(pipe, meta, df: pd.DataFrame) -> pd.DataFrame:
    p = pipe.predict_proba(df)[:, 1]
    thr = meta["threshold"]
    return pd.DataFrame({"probability": p, "high_risk": p >= thr,
                         "risk_band": [risk_band(x, thr) for x in p]}, index=df.index)
