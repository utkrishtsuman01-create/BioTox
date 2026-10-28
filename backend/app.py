from __future__ import annotations

import os
import re
from pathlib import Path\nfrom typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

app = FastAPI(
    title="BioTox API",
    description=(
        "Educational computational toxicology API scaffold. "
        "No trained toxicity model is bundled; prediction requests return 503 until "
        "a validated model adapter is explicitly configured."
    ),
    version="0.1.0",
)

origins = [x.strip() for x in os.getenv("BIOTOX_ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class ScreeningRequest(BaseModel):
    compound: str = Field(min_length=1, max_length=2000)
    input_type: Literal["name", "smiles"] = "name"
    endpoint: Literal["general", "acute", "environmental"] = "general"
    target_context: Literal[
        "General / unspecified", "Mammalian", "Aquatic organism", "Cell-based assay"
    ] = "General / unspecified"

    @field_validator("compound")
    @classmethod
    def strip_and_validate(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Compound input cannot be empty.")
        if any(ord(c) < 32 and c not in "\\t\\n\\r" for c in value):
            raise ValueError("Compound input contains unsupported control characters.")
        return value




TOX21_ENDPOINTS = {
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53",
}
MODEL_DIR = Path(os.getenv("BIOTOX_MODEL_DIR", str(Path(__file__).parent / "models")))


class Tox21Request(BaseModel):
    smiles: str = Field(min_length=1, max_length=2000)
    endpoint: str

    @field_validator("smiles")
    @classmethod
    def validate_smiles_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("SMILES cannot be empty.")
        return value

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        if value not in TOX21_ENDPOINTS:
            raise ValueError("Unknown Tox21 assay endpoint.")
        return value


@app.post("/api/v1/tox21/predict")
def tox21_predict(request: Tox21Request):
    """Predict a specific Tox21 assay only when its locally trained artifact exists."""
    safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in request.endpoint)
    artifact = MODEL_DIR / f"{safe_name}.joblib"
    if not artifact.is_file():
        raise HTTPException(
            status_code=503,
            detail={
                "code": "ENDPOINT_MODEL_NOT_TRAINED",
                "message": (
                    f"No trained artifact exists for {request.endpoint}. Run "
                    "backend/train_tox21.py and review evaluation_report.json first."
                ),
            },
        )
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        import joblib
        import numpy as np

        molecule = Chem.MolFromSmiles(request.smiles)
        if molecule is None:
            raise HTTPException(status_code=422, detail="Invalid SMILES representation.")
        bundle = joblib.load(artifact)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            molecule, int(bundle["radius"]), nBits=int(bundle["n_bits"])
        )
        vector = np.zeros((int(bundle["n_bits"]),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, vector)
        probability = float(bundle["model"].predict_proba(vector.reshape(1, -1))[0, 1])
        return {
            "compound_input": request.smiles,
            "endpoint": request.endpoint,
            "assay_activity_probability": probability,
            "interpretation": (
                "Estimated probability of the positive Tox21 assay label, not a universal "
                "toxicity probability or proof of safety."
            ),
            "model_dataset": "Tox21",
            "evaluation": bundle.get("metrics", {}),
            "limitations": [
                "Research baseline trained with a random split; external validation may be absent.",
                "Assay activity is endpoint-specific and is not a universal toxicity category.",
                "Do not use this output for exposure, treatment, regulatory, or safety decisions.",
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        # Avoid returning file paths, serialized data, or internal exception details.
        raise HTTPException(status_code=500, detail="Model inference failed. Check server logs.") from exc
\n\n@app.get("/")
def root():
    return {
        "service": "BioTox API",
        "status": "running",
        "model_status": "not_configured",
        "message": "API scaffold is running; no trained toxicity model is connected.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": False}


@app.post("/api/v1/predict")
def predict(request: ScreeningRequest):
    # Fail closed: never fabricate scientific predictions when a real model is absent.
    raise HTTPException(
        status_code=503,
        detail={
            "code": "MODEL_NOT_CONFIGURED",
            "message": (
                "No validated toxicity model is configured. This deployment does not "
                "generate scientific predictions. Configure and evaluate a real model "
                "before enabling this endpoint."
            ),
        },
    )
