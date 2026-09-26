"""
Training entrypoint.

Run with:
    python -m src.train

What it does:
1. Loads and joins train.csv + resolution_log.csv (target = final_team).
2. Performs the chronological (temporal) train/holdout split -- the PRIMARY
   evaluation split used everywhere else in this project.
3. Runs three experiments (word TF-IDF, char TF-IDF, char TF-IDF + structured
   features) on the temporal split and writes reports/model_comparison.csv.
4. Also runs a stratified random split of the same training rows as a
   secondary robustness check (reported, not primary).
5. Fits the FINAL model (char TF-IDF + structured features + calibrated
   probabilities) on ALL of train.csv (so the shipped model uses every
   available historical record), and saves it + the feature bundle + metadata
   to models/.
6. The temporal holdout metrics used for reporting throughout the project
   come from the model trained on the temporal train split only (i.e. NOT
   the final production model) -- this avoids ever evaluating a model on
   data it has seen.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.data import CANONICAL_TEAMS, build_training_frame, temporal_split
from src.features import FeatureBundle

RANDOM_SEED = 42
TEMPORAL_CUTOFF = "2026-04-01"
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def _fit_eval(train_df, eval_df, variant: str, label: str) -> dict:
    fb = FeatureBundle()
    Xtr = fb.fit_transform(train_df, variant=variant)
    ytr = train_df["final_team"].astype(str).to_numpy()
    Xev = fb.transform(eval_df)
    yev = eval_df["final_team"].astype(str).to_numpy()

    clf = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    fit_secs = time.time() - t0

    pred = clf.predict(Xev)
    acc = accuracy_score(yev, pred)
    macro_f1 = f1_score(yev, pred, average="macro")
    return {
        "experiment": label,
        "variant": variant,
        "n_train": len(train_df),
        "n_eval": len(eval_df),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "fit_seconds": round(fit_secs, 2),
    }


def run_experiments(temporal_train: pd.DataFrame, temporal_holdout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(
        _fit_eval(temporal_train, temporal_holdout, "word", "1. Word TF-IDF (baseline)")
    )
    rows.append(
        _fit_eval(temporal_train, temporal_holdout, "char", "2. Char TF-IDF")
    )
    rows.append(
        _fit_eval(temporal_train, temporal_holdout, "char_struct", "3. Char TF-IDF + structured features")
    )

    # Secondary robustness check: stratified random split of the SAME rows
    # that make up temporal_train + temporal_holdout, so the comparison is
    # apples-to-apples in terms of which requests are available, just split
    # differently. This is NOT the primary result (see docstring / policy in
    # the assignment: chronological split is primary for a temporal routing
    # problem).
    all_rows = pd.concat([temporal_train, temporal_holdout], ignore_index=True)
    rand_train, rand_holdout = train_test_split(
        all_rows, test_size=len(temporal_holdout) / len(all_rows),
        stratify=all_rows["final_team"], random_state=RANDOM_SEED,
    )
    rows.append(
        _fit_eval(rand_train, rand_holdout, "char_struct", "4. Char TF-IDF + structured (random split, robustness check)")
    )

    df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    return df


def fit_final_model(full_df: pd.DataFrame):
    """Fit the shipped model on ALL of train.csv, with calibrated probabilities.

    CalibratedClassifierCV with cv=5 fits the base LinearSVC on internal
    cross-validation folds of the training data only, so the resulting
    predict_proba is a genuinely calibrated probability (Platt scaling on
    held-in folds) rather than a raw, unscaled decision_function value. No
    holdout or test data is used in this fit.
    """
    fb = FeatureBundle()
    X = fb.fit_transform(full_df, variant="char_struct")
    y = full_df["final_team"].astype(str).to_numpy()

    base = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    clf.fit(X, y)
    return fb, clf


def fit_temporal_model_for_reporting(temporal_train: pd.DataFrame):
    """Same recipe as fit_final_model, but fit ONLY on the pre-cutoff temporal
    training rows. Used to generate all held-out evaluation artifacts
    (confusion matrix, per-team metrics, error analysis, confidence-band
    calibration) so nothing is evaluated on data the reporting model has seen.
    """
    fb = FeatureBundle()
    X = fb.fit_transform(temporal_train, variant="char_struct")
    y = temporal_train["final_team"].astype(str).to_numpy()
    base = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_SEED, max_iter=5000)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    clf.fit(X, y)
    return fb, clf


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    full_df = build_training_frame()
    temporal_train, temporal_holdout = temporal_split(full_df, cutoff=TEMPORAL_CUTOFF)

    print(f"Full training frame: {len(full_df)} rows "
          f"({full_df.created_at_ist.min()} .. {full_df.created_at_ist.max()})")
    print(f"Temporal split @ {TEMPORAL_CUTOFF}: train={len(temporal_train)} holdout={len(temporal_holdout)}")

    print("Running model comparison experiments...")
    comp = run_experiments(temporal_train, temporal_holdout)
    print(comp.to_string(index=False))

    print("Fitting temporal reporting model (train-only) for evaluation artifacts...")
    fb_temporal, clf_temporal = fit_temporal_model_for_reporting(temporal_train)
    joblib.dump(fb_temporal, MODELS_DIR / "temporal_feature_bundle.joblib")
    joblib.dump(clf_temporal, MODELS_DIR / "temporal_model.joblib")
    temporal_holdout.to_csv(MODELS_DIR / "temporal_holdout.csv", index=False)
    temporal_train.to_csv(MODELS_DIR / "temporal_train.csv", index=False)

    print("Fitting FINAL production model on all of train.csv...")
    fb_final, clf_final = fit_final_model(full_df)
    joblib.dump(fb_final, MODELS_DIR / "feature_bundle.joblib")
    joblib.dump(clf_final, MODELS_DIR / "model.joblib")

    print("Building historical similarity retriever...")
    from src.retrieval import build_and_save_retriever
    retriever = build_and_save_retriever(full_df)
    print(f"Retrieval mode: {retriever.mode}")

    metadata = {
        "trained_at": pd.Timestamp.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "temporal_cutoff": TEMPORAL_CUTOFF,
        "n_train_full": len(full_df),
        "n_temporal_train": len(temporal_train),
        "n_temporal_holdout": len(temporal_holdout),
        "classes": sorted(full_df["final_team"].unique().tolist()),
        "canonical_teams": CANONICAL_TEAMS,
        "feature_variant": "char_struct",
        "model": "LinearSVC (calibrated, sigmoid, cv=5)",
        "target_column": "final_team (from resolution_log.csv, NOT train.csv team_label)",
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done. Artifacts written to models/ and reports/model_comparison.csv")


if __name__ == "__main__":
    main()
