"""
FastAPI service for the Kestrel routing model.

Run with:
    uvicorn app.api:app --reload --port 8000

Example:
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"request_text": "My water purifier is leaking from the bottom",
           "product_family": "Water Purifier", "warranty_status": "in_warranty",
           "channel": "whatsapp"}'
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.predictor import RoutingPredictor, ValidationError


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the singleton at startup so the first request isn't slow, and so
    # a broken model artifact fails loudly at boot rather than per-request.
    RoutingPredictor.instance()
    yield


app = FastAPI(
    title="Kestrel Routing Intelligence API",
    description="Predicts the operational team for an inbound Kestrel service request.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    request_text: str = Field(..., min_length=1, description="Customer's opening message or IVR transcript.")
    product_family: Optional[str] = Field(None, description="e.g. 'Water Purifier'")
    warranty_status: Optional[str] = Field(None, description="in_warranty | shield | out_of_warranty")
    channel: Optional[str] = Field(None, description="ivr | chat | whatsapp | email")
    source: Optional[str] = Field(None, description="crm | legacy_zoho")


class SimilarCase(BaseModel):
    request_text: str
    team: str
    similarity: float


class RunnerUp(BaseModel):
    team: str
    confidence: float


class PredictResponse(BaseModel):
    predicted_team: str
    confidence: float
    confidence_band: str
    review_required: bool
    margin: float
    runner_up: Optional[RunnerUp]
    reasons: list[str]
    similar_cases: list[SimilarCase]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        predictor = RoutingPredictor.instance()
        result = predictor.predict(payload.model_dump())
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    result.pop("input_echo", None)
    return result
