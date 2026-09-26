import pandas as pd

from src.data import build_training_frame, canonicalize_team, clean_text, temporal_split


def test_clean_text_fixes_mojibake_and_whitespace():
    assert clean_text("thÃ© mcb   tripping") == "the mcb tripping"
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_canonicalize_team_renames():
    assert canonicalize_team("Installations") == "Installs & Demo"
    assert canonicalize_team("Consumables") == "Filters & Consumables"
    assert canonicalize_team("Repairs") == "Repairs"


def test_build_training_frame_has_final_team_not_only_team_label():
    df = build_training_frame()
    assert "final_team" in df.columns
    assert "team_label" in df.columns
    # target must differ from the legacy bot's own decision for a meaningful
    # fraction of rows -- if this ever became 0, something upstream broke.
    mismatch_rate = (df["team_label"] != df["final_team"]).mean()
    assert mismatch_rate > 0.05


def test_temporal_split_is_chronological_and_non_overlapping():
    df = build_training_frame()
    train, holdout = temporal_split(df, cutoff="2026-04-01")
    assert len(train) + len(holdout) == len(df)
    assert train["created_at_ist"].max() < pd.Timestamp("2026-04-01")
    assert holdout["created_at_ist"].min() >= pd.Timestamp("2026-04-01")
