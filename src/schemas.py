"""Input validation shared by the API and the dashboard (Pydantic)."""
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from .config import MED_COLS

Dose = Literal["No", "Steady", "Up", "Down"]
AGES = [f"[{i}-{i+10})" for i in range(0, 100, 10)]


class PatientInput(BaseModel):
    race: Optional[Literal["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]] = "Caucasian"
    gender: Literal["Female", "Male"] = "Female"
    age: Literal[tuple(AGES)] = "[60-70)"
    admission_type_id: int = Field(1, ge=1, le=8)
    discharge_disposition_id: int = Field(1, ge=1, le=30)
    admission_source_id: int = Field(7, ge=1, le=26)
    time_in_hospital: int = Field(4, ge=1, le=14)
    payer_code: Optional[str] = None
    medical_specialty: Optional[str] = None
    num_lab_procedures: int = Field(43, ge=0, le=150)
    num_procedures: int = Field(1, ge=0, le=6)
    num_medications: int = Field(16, ge=0, le=100)
    number_outpatient: int = Field(0, ge=0, le=50)
    number_emergency: int = Field(0, ge=0, le=50)
    number_inpatient: int = Field(0, ge=0, le=30)
    diag_1: Optional[str] = Field("428", description="ICD-9 code")
    diag_2: Optional[str] = None
    diag_3: Optional[str] = None
    number_diagnoses: int = Field(7, ge=1, le=16)
    max_glu_serum: Literal["None", ">200", ">300", "Norm"] = "None"
    A1Cresult: Literal["None", ">7", ">8", "Norm"] = "None"
    insulin: Dose = "No"
    metformin: Dose = "No"
    glipizide: Dose = "No"
    glyburide: Dose = "No"
    pioglitazone: Dose = "No"
    rosiglitazone: Dose = "No"
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["Yes", "No"] = "Yes"

    def to_frame(self) -> pd.DataFrame:
        d = self.model_dump()
        for m in MED_COLS:
            d.setdefault(m, "No")
        return pd.DataFrame([d])
