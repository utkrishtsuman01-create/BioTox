from __future__ import annotations

import os
import re
from typing import Literal

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


@app.get("/")
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
