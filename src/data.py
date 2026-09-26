"""
Data loading and canonicalization for the Kestrel routing project.

Key design decision (documented at length in reports/validation_report.md and
MEMO.md): the prediction TARGET is `final_team` from resolution_log.csv, not
`team_label` in train.csv.

`team_label` is the queue assigned by the *legacy vendor routing bot at
creation time* (confirmed identical to resolution_log.first_team for every
row). It is the very system we are replacing, and it is demonstrably biased:
of the requests in train.csv that merely mention a payment word (paid/UPI/
EMI/card/payment), the legacy bot sent 823/1239 (66%) to Billing, while the
team that actually closed those requests (final_team) was Billing only
283/1239 (23%) of the time. The "burnt smell ... paid on UPI" example from
the assignment brief is literally in the data: the bot always routed it to
Billing, and every single one was actually closed by Repairs.

Training a classifier to reproduce `team_label` would therefore reproduce the
bot's own routing bug -- exactly what Kestrel is trying to fix. `final_team`
is the closest thing this dataset has to ground truth for "which team should
this request go to", so it is used as `y` throughout. `team_label` /
`first_team` are never used as model features (the assignment explicitly
forbids this), but the mismatch between them and `final_team` is used in
reports/EDA as evidence for the policy layer and as the "incumbent bot
accuracy" baseline quoted in MEMO.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

DATA_DIR = "data"

# Team renames effective 2026-01-15 (ops-policy.pdf §5). Responsibilities did
# not change, so old and new names are the same class.
TEAM_RENAMES = {
    "Installations": "Installs & Demo",
    "Consumables": "Filters & Consumables",
}

CANONICAL_TEAMS = [
    "Installs & Demo",
    "Repairs",
    "Filters & Consumables",
    "Billing",
    "Returns & Replacement",
    "Warranty Claims",
    "Product Advice",
]

# Common mojibake patterns seen in text migrated from the legacy Zoho desk
# (UTF-8 bytes re-decoded as Windows-1252 / Latin-1). We fix the small,
# high-frequency set actually observed in the data rather than attempting a
# universal encoding fixer.
_MOJIBAKE_MAP = {
    "Ã¢â‚¬â„¢": "'",
    "Ã¢â‚¬Å“": '"',
    "Ã¢â‚¬ï¿½": '"',
    "Ã¢â‚¬â€œ": "-",
    "Ã¢â‚¬â€": "-",
    "Ã¢â‚¬Â¦": "...",
    "Ã©": "e",
    "Ã¨": "e",
    "Ã¢": "a",
    "Â ": " ",
    "Â°": " degrees",
}


def clean_text(text: str) -> str:
    """Normalize whitespace and repair known mojibake sequences.

    This is deliberately conservative: it fixes the specific garbled byte
    sequences observed in the legacy_zoho rows (see EDA) rather than
    guessing at a general re-encoding, which could silently corrupt clean
    text from the crm source.
    """
    if not isinstance(text, str):
        return ""
    out = text
    for bad, good in _MOJIBAKE_MAP.items():
        out = out.replace(bad, good)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def canonicalize_team(name: str) -> str:
    """Map a possibly-old team name to its current (post 15-Jan-2026) name."""
    if not isinstance(name, str):
        return name
    return TEAM_RENAMES.get(name.strip(), name.strip())


@dataclass
class Dataset:
    train_full: pd.DataFrame  # all of train.csv, target attached, chronological
    test_unlabelled: pd.DataFrame


def load_raw(data_dir: str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["created_at_ist"])
    res = pd.read_csv(f"{data_dir}/resolution_log.csv", parse_dates=["resolved_at"])
    test = pd.read_csv(f"{data_dir}/test_unlabelled.csv", parse_dates=["created_at_ist"])
    return train, res, test


def build_training_frame(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Join train.csv with resolution_log.csv and attach the canonical target.

    Every row of train.csv gets exactly one resolution_log row (verified
    1:1 on request_id, see EDA), so this is a simple inner merge. Only
    `final_team` and `transfers` are pulled in for analytics/target purposes;
    `resolved_at` is intentionally NOT attached to the modelling frame,
    since resolution time is post-hoc information that would never be
    available at prediction time (it is used only in the separate
    operations-analytics path, see src/ops_analytics equivalent in
    app/ui.py "Operations" page).
    """
    train, res, _ = load_raw(data_dir)
    merged = train.merge(res[["request_id", "final_team", "transfers"]], on="request_id", how="inner")
    assert len(merged) == len(train), "expected 1:1 join between train.csv and resolution_log.csv"

    merged["team_label"] = merged["team_label"].map(canonicalize_team)
    merged["final_team"] = merged["final_team"].map(canonicalize_team)
    merged["request_text_clean"] = merged["request_text"].map(clean_text)
    merged = merged.sort_values("created_at_ist").reset_index(drop=True)
    return merged


def load_test(data_dir: str = DATA_DIR) -> pd.DataFrame:
    _, _, test = load_raw(data_dir)
    test = test.copy()
    test["request_text_clean"] = test["request_text"].map(clean_text)
    return test


def temporal_split(df: pd.DataFrame, cutoff: str = "2026-04-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: everything before `cutoff` is train, on/after is holdout.

    Chosen cutoff gives ~12 months train / ~3 months holdout out of the
    15 months of history in train.csv (2025-04-01 .. 2026-06-30), which is
    roughly a 80/20 split by volume and lines up with a calendar quarter
    boundary. See reports/validation_report.md for the actual row counts.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    train_part = df[df["created_at_ist"] < cutoff_ts].reset_index(drop=True)
    holdout_part = df[df["created_at_ist"] >= cutoff_ts].reset_index(drop=True)
    return train_part, holdout_part
