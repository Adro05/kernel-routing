"""
Shared prediction pipeline. Both app/api.py and app/ui.py call this module
so prediction logic is never duplicated between the two front ends.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from src.data import CANONICAL_TEAMS, clean_text
from src.evaluate import band_from_prob, confidence_and_margin
from src.explanations import build_reasons
from src.retrieval import HistoricalRetriever

MODELS_DIR = Path("models")
DATA_DIR = Path("data")

VALID_CHANNELS = {"ivr", "chat", "whatsapp", "email"}
VALID_WARRANTY = {"in_warranty", "shield", "out_of_warranty"}
VALID_PRODUCTS = {
    "Water Purifier", "Air Fryer", "Mixer Grinder", "Induction Cooktop",
    "Room Heater", "Ceiling Fan", "Robot Vacuum",
}


class ValidationError(Exception):
    pass


class RoutingPredictor:
    """Loads model artifacts once and serves predict() for a single request."""

    _instance: Optional["RoutingPredictor"] = None

    def __init__(self):
        self.feature_bundle = joblib.load(MODELS_DIR / "feature_bundle.joblib")
        self.model = joblib.load(MODELS_DIR / "model.joblib")
        with open(MODELS_DIR / "confidence_bands.json") as f:
            self.bands = json.load(f)
        with open(MODELS_DIR / "metadata.json") as f:
            self.metadata = json.load(f)

        retriever_path = MODELS_DIR / "retriever.joblib"
        if retriever_path.exists():
            self.retriever: Optional[HistoricalRetriever] = joblib.load(retriever_path)
        else:
            self.retriever = None

    @classmethod
    def instance(cls) -> "RoutingPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def validate_request(payload: dict) -> dict:
        text = payload.get("request_text")
        if not text or not str(text).strip():
            raise ValidationError("request_text is required and cannot be empty.")

        product_family = payload.get("product_family") or "Unknown"
        warranty_status = payload.get("warranty_status") or "missing"
        channel = payload.get("channel") or "missing"
        source = payload.get("source") or "crm"

        if product_family not in VALID_PRODUCTS and product_family != "Unknown":
            # Don't hard-fail on an unfamiliar product family -- degrade gracefully,
            # the one-hot encoder handles unknown categories via handle_unknown='ignore'.
            pass
        if warranty_status not in VALID_WARRANTY and warranty_status != "missing":
            pass
        if channel not in VALID_CHANNELS and channel != "missing":
            pass

        return {
            "request_text": str(text),
            "product_family": product_family,
            "warranty_status": warranty_status,
            "channel": channel,
            "source": source,
        }

    def predict(self, payload: dict) -> dict:
        clean_payload = self.validate_request(payload)
        row = pd.DataFrame([{
            "request_text": clean_payload["request_text"],
            "request_text_clean": clean_text(clean_payload["request_text"]),
            "product_family": clean_payload["product_family"],
            "warranty_status": clean_payload["warranty_status"],
            "channel": clean_payload["channel"],
            "source": clean_payload["source"],
        }])

        X = self.feature_bundle.transform(row)
        pred, top_prob, margin, proba = confidence_and_margin(self.model, X)
        predicted_team = str(pred[0])
        confidence = float(top_prob[0])
        band = band_from_prob(confidence, self.bands)
        review_required = band != "high"

        similar_cases = []
        if self.retriever is not None:
            similar_cases = self.retriever.query(row["request_text_clean"].iloc[0], k=3)

        reasons = build_reasons(
            request_text=clean_payload["request_text"],
            predicted_team=predicted_team,
            product_family=clean_payload["product_family"],
            confidence_band=band,
            similar_cases=similar_cases,
        )

        classes = list(self.model.classes_)
        class_probs = {c: round(float(p), 4) for c, p in zip(classes, proba[0])}
        top2 = sorted(class_probs.items(), key=lambda kv: -kv[1])[:2]

        return {
            "predicted_team": predicted_team,
            "confidence": round(confidence, 4),
            "confidence_band": band,
            "review_required": review_required,
            "margin": round(float(margin[0]), 4),
            "runner_up": {"team": top2[1][0], "confidence": top2[1][1]} if len(top2) > 1 else None,
            "reasons": reasons,
            "similar_cases": similar_cases,
            "input_echo": clean_payload,
        }
