from pathlib import Path

import pandas as pd

from src.data import CANONICAL_TEAMS, load_test

PREDICTIONS_PATH = Path("predictions.csv")


def test_predictions_csv_matches_test_unlabelled():
    """Runs against the predictions.csv produced by `python -m src.predict`.

    Requires predictions.csv to already exist (generated via `python -m
    src.predict`); this test checks the artifact's schema/alignment, it does
    not re-run the model itself.
    """
    if not PREDICTIONS_PATH.exists():
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "src.predict"], check=True)

    preds = pd.read_csv(PREDICTIONS_PATH)
    test = load_test()

    assert list(preds.columns) == ["request_id", "team"]
    assert len(preds) == len(test)
    assert preds["request_id"].reset_index(drop=True).equals(
        test["request_id"].reset_index(drop=True)
    )
    assert preds["request_id"].notna().all()
    assert preds["team"].notna().all()
    assert set(preds["team"].unique()) <= set(CANONICAL_TEAMS)
