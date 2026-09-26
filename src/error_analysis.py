"""
Error analysis entrypoint. Run with:
    python -m src.error_analysis

Classifies each misrouted temporal-holdout request into one of a small set
of error categories using only signals available in the data (never
hand-picked examples), and writes reports/error_analysis.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import band_from_prob, confidence_and_margin, load_temporal_artifacts

REPORTS_DIR = Path("reports")

PAYMENT_WORDS = ("paid", "upi", "emi", "card", "payment", "refund", "invoice", "gst", "coupon")
FAULT_WORDS = ("leak", "leaking", "noise", "error", "burnt", "smell", "not working", "tripping",
               "broken", "crack", "blank", "not turning on", "faulty", "stopped")
CONSUMABLE_WORDS = ("filter", "candle", "membrane", "jar", "brush", "blade", "amc", "spare")
DAMAGE_WORDS = ("damaged", "wrong item", "missing parts", "wrong product", "incomplete")
INSTALL_WORDS = ("install", "demo", "wall mount", "wall-mount", "fitting new")
WARRANTY_WORDS = ("warranty", "shield", "claim status", "registration")
ADVICE_WORDS = ("how to", "which setting", "best setting", "recommend", "usage", "how do i")


def categorize_error(row) -> str:
    text = row["request_text_clean"].lower()
    true_t, pred_t = row["true_team"], row["predicted_team"]

    def has(words):
        return any(w in text for w in words)

    signals = {
        "Billing": has(PAYMENT_WORDS),
        "Repairs": has(FAULT_WORDS),
        "Filters & Consumables": has(CONSUMABLE_WORDS),
        "Returns & Replacement": has(DAMAGE_WORDS),
        "Installs & Demo": has(INSTALL_WORDS),
        "Warranty Claims": has(WARRANTY_WORDS),
        "Product Advice": has(ADVICE_WORDS),
    }
    n_signals = sum(signals.values())

    if signals.get(true_t) and signals.get(pred_t) and n_signals >= 2:
        return "multi-intent (multiple team-relevant phrases present)"
    if signals["Billing"] and pred_t == "Billing" and true_t != "Billing":
        return "policy ambiguity (payment word present, real issue elsewhere)"
    if n_signals == 0:
        return "insufficient context (short/generic text, no clear symptom keywords)"
    if {true_t, pred_t} <= {"Repairs", "Filters & Consumables"}:
        return "overlapping team boundary (fault vs consumable request)"
    if {true_t, pred_t} <= {"Returns & Replacement", "Installs & Demo"}:
        return "overlapping team boundary (delivery/damage vs installation)"
    if {true_t, pred_t} <= {"Warranty Claims", "Repairs"}:
        return "overlapping team boundary (warranty coverage vs repair symptom)"
    if row.get("has_mojibake", False):
        return "noisy text (legacy encoding corruption)"
    return "other / unclear boundary"


def main():
    fb, clf, holdout = load_temporal_artifacts()
    X = fb.transform(holdout)
    pred, top_prob, margin, proba = confidence_and_margin(clf, X)

    with open(REPORTS_DIR / "confidence_bands.json") as f:
        bands = json.load(f)

    df = holdout.copy()
    df["predicted_team"] = pred
    df["true_team"] = df["final_team"].astype(str)
    df["confidence"] = top_prob.round(4)
    df["margin"] = margin.round(4)
    df["confidence_band"] = [band_from_prob(p, bands) for p in top_prob]
    df["has_mojibake"] = df["request_text"].str.contains("Ã", na=False)

    errors = df[df["predicted_team"] != df["true_team"]].copy()
    errors["error_category"] = errors.apply(categorize_error, axis=1)

    out_cols = [
        "request_id", "request_text", "true_team", "predicted_team", "confidence",
        "margin", "confidence_band", "product_family", "warranty_status", "channel",
        "source", "error_category",
    ]
    errors[out_cols].to_csv(REPORTS_DIR / "error_analysis.csv", index=False)

    print(f"Total holdout: {len(df)}  Errors: {len(errors)}  Error rate: {len(errors)/len(df):.4f}")
    print(errors["error_category"].value_counts())


if __name__ == "__main__":
    main()
