"""
Prediction entrypoint for scoring test_unlabelled.csv (the actual held-out
requests this project must predict for submission).

Run with:
    python -m src.predict

Uses the FINAL production model (trained on all of train.csv), NOT the
temporal reporting model, since we want the best model available at
inference time and there is no leakage risk here (test_unlabelled.csv has
no labels for the model to have seen).

Writes the assignment-required submission artifact to the repository root:
    predictions.csv   (columns: request_id, team)

This file is derived from client data (test_unlabelled.csv) and is
therefore listed in .gitignore -- it should be generated locally, not
committed to the public repository.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data import CANONICAL_TEAMS, clean_text, load_test
from src.evaluate import confidence_and_margin

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
PREDICTIONS_PATH = Path("predictions.csv")


def main():
    fb = joblib.load(MODELS_DIR / "feature_bundle.joblib")
    clf = joblib.load(MODELS_DIR / "model.joblib")

    test = load_test()
    X = fb.transform(test)
    pred, top_prob, margin, proba = confidence_and_margin(clf, X)

    out = pd.DataFrame({"request_id": test["request_id"], "team": pred})

    # --- Pre-write validation (assignment compliance checks) ---
    assert len(out) == len(test), (
        f"prediction count ({len(out)}) does not match test_unlabelled row count ({len(test)})"
    )
    assert out["request_id"].reset_index(drop=True).equals(
        test["request_id"].reset_index(drop=True)
    ), "request_id order/values in predictions do not match test_unlabelled.csv"
    assert out["request_id"].notna().all(), "found null request_id in predictions"
    assert out["team"].notna().all(), "found null team in predictions"
    assert set(out["team"].unique()) <= set(CANONICAL_TEAMS), (
        f"predicted team(s) outside the 7 canonical teams: "
        f"{set(out['team'].unique()) - set(CANONICAL_TEAMS)}"
    )

    out.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Scored {len(out)} requests -> {PREDICTIONS_PATH}")
    print("Validation checks passed:")
    print(f"  - row count matches test_unlabelled.csv: {len(out)} == {len(test)}")
    print("  - request_id order/values match test_unlabelled.csv")
    print("  - no null request_id or team values")
    print(f"  - all predicted teams within the 7 canonical teams: {CANONICAL_TEAMS}")
    print(out["team"].value_counts())
    print(f"Mean confidence: {top_prob.mean():.4f}")


if __name__ == "__main__":
    main()
