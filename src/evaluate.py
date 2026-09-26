"""
Evaluation entrypoint. Run with:
    python -m src.evaluate

Uses the TEMPORAL reporting model (models/temporal_model.joblib, fit only on
pre-cutoff data) against the temporal holdout (models/temporal_holdout.csv,
post-cutoff data the model has never seen). Generates:

- reports/confusion_matrix.png
- reports/per_team_metrics.csv
- reports/confidence_bands.json  (thresholds used by app/predictor.py)
- prints overall accuracy / macro F1 to stdout (also written into
  reports/validation_report.md by hand, using these exact numbers)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.data import CANONICAL_TEAMS

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def load_temporal_artifacts():
    fb = joblib.load(MODELS_DIR / "temporal_feature_bundle.joblib")
    clf = joblib.load(MODELS_DIR / "temporal_model.joblib")
    holdout = pd.read_csv(MODELS_DIR / "temporal_holdout.csv", parse_dates=["created_at_ist"])
    return fb, clf, holdout


def confidence_and_margin(clf, X):
    proba = clf.predict_proba(X)
    classes = clf.classes_
    order = np.argsort(-proba, axis=1)
    top_idx = order[:, 0]
    second_idx = order[:, 1]
    top_prob = proba[np.arange(len(proba)), top_idx]
    second_prob = proba[np.arange(len(proba)), second_idx]
    margin = top_prob - second_prob
    pred = classes[top_idx]
    return pred, top_prob, margin, proba


def derive_confidence_bands(top_prob: np.ndarray, correct: np.ndarray) -> dict:
    """Empirically derive confidence-band thresholds from validation behaviour.

    Sort holdout predictions by top_prob DESCENDING and walk down from the
    most confident prediction, tracking cumulative accuracy of "everything at
    least this confident". The high-band threshold is the lowest probability
    at which that cumulative accuracy is still >= 90%; the medium-band
    threshold is the lowest probability at which it is still >= 70%.
    Everything below the medium threshold is 'low'. Computed once here on the
    temporal holdout; the fixed thresholds are then used unchanged at
    inference time (the real test_unlabelled.csv has no labels to tune against
    anyway).
    """
    order = np.argsort(-top_prob)
    sorted_prob = top_prob[order]
    sorted_correct = correct[order].astype(float)
    n = len(sorted_prob)
    cum_acc = np.cumsum(sorted_correct) / np.arange(1, n + 1)

    def threshold_for(target: float, fallback: float) -> float:
        ok = np.where(cum_acc >= target)[0]
        if len(ok) == 0:
            return fallback
        last_ok_idx = ok.max()  # largest subset (lowest prob) still meeting target
        return float(sorted_prob[last_ok_idx])

    high_thr = threshold_for(0.90, fallback=0.85)
    med_thr = threshold_for(0.70, fallback=0.55)
    if med_thr >= high_thr:
        med_thr = max(0.3, high_thr - 0.2)
    return {"high_threshold": round(high_thr, 3), "medium_threshold": round(med_thr, 3)}


def band_from_prob(p: float, bands: dict) -> str:
    if p >= bands["high_threshold"]:
        return "high"
    if p >= bands["medium_threshold"]:
        return "medium"
    return "low"


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    fb, clf, holdout = load_temporal_artifacts()

    X = fb.transform(holdout)
    y_true = holdout["final_team"].astype(str).to_numpy()
    pred, top_prob, margin, proba = confidence_and_margin(clf, X)

    acc = accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, average="macro")
    print(f"Temporal holdout accuracy: {acc:.4f}")
    print(f"Temporal holdout macro F1: {macro_f1:.4f}")

    labels = sorted(set(y_true) | set(pred))
    prec, rec, f1, support = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
    per_team = pd.DataFrame({
        "team": labels, "precision": prec.round(4), "recall": rec.round(4),
        "f1": f1.round(4), "support": support,
    }).sort_values("support", ascending=False)
    per_team.to_csv(REPORTS_DIR / "per_team_metrics.csv", index=False)
    print(per_team.to_string(index=False))

    cm = confusion_matrix(y_true, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax.set_title(f"Kestrel Routing — Temporal Holdout Confusion Matrix\naccuracy={acc:.3f}  macro F1={macro_f1:.3f}")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    correct = (pred == y_true).astype(int)
    bands = derive_confidence_bands(top_prob, correct)
    bands["overall_accuracy"] = round(float(acc), 4)
    bands["overall_macro_f1"] = round(float(macro_f1), 4)
    with open(REPORTS_DIR / "confidence_bands.json", "w") as f:
        json.dump(bands, f, indent=2)
    # Also copy into models/ -- this is what app/predictor.py loads at
    # inference time, since confidence bands are a production artifact of
    # the (train-only) validation run, not just a report.
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "confidence_bands.json", "w") as f:
        json.dump(bands, f, indent=2)
    print("Confidence bands:", bands)

    # sanity: accuracy within each derived band, for the validation report
    band_labels = np.array([band_from_prob(p, bands) for p in top_prob])
    band_report = pd.DataFrame({"band": band_labels, "correct": correct}).groupby("band").agg(
        n=("correct", "size"), accuracy=("correct", "mean")
    ).reset_index()
    band_report.to_csv(REPORTS_DIR / "confidence_band_accuracy.csv", index=False)
    print(band_report.to_string(index=False))


if __name__ == "__main__":
    main()
