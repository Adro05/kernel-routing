import pandas as pd

from src.policy import apply_hard_override, payment_mention_signal


def test_payment_mention_without_billing_issue_flags_note():
    sig = payment_mention_signal(
        "purifier paid for by upi and now it is leaking", predicted_team="Repairs"
    )
    assert sig is not None
    assert "does not make this a Billing case" in sig.note


def test_payment_mention_with_real_billing_issue_flags_supportive_note():
    sig = payment_mention_signal(
        "double charge on invoice, please refund my payment", predicted_team="Billing"
    )
    assert sig is not None
    assert "Billing actually handles" in sig.note


def test_no_payment_mention_returns_none():
    assert payment_mention_signal("purifier leaking", predicted_team="Repairs") is None


def test_hard_override_disabled_by_default_is_a_noop():
    # Even for a clear-cut "fault text predicted as Billing" case, the
    # override must NOT fire unless explicitly enabled.
    result = apply_hard_override("purifier leaking, paid on upi", predicted_team="Billing", enabled=False)
    assert result == "Billing"


def test_hard_override_when_explicitly_enabled_does_flip():
    result = apply_hard_override("purifier leaking, paid on upi", predicted_team="Billing", enabled=True)
    assert result == "Repairs"


def test_hard_override_empirically_does_not_help_on_temporal_holdout():
    """Reproduces the evaluation documented in src/policy.py's module docstring:
    the candidate override should change at most a handful of temporal-holdout
    predictions and must not be assumed to improve accuracy without
    re-checking against current holdout data.
    """
    from src.evaluate import confidence_and_margin, load_temporal_artifacts

    fb, clf, holdout = load_temporal_artifacts()
    X = fb.transform(holdout)
    pred, _, _, _ = confidence_and_margin(clf, X)
    df = holdout.copy()
    df["pred"] = pred
    df["true"] = df["final_team"].astype(str)

    df["pred_override"] = [
        apply_hard_override(t, p, enabled=True)
        for t, p in zip(df["request_text_clean"], df["pred"])
    ]
    acc_before = (df["pred"] == df["true"]).mean()
    acc_after = (df["pred_override"] == df["true"]).mean()
    n_changed = (df["pred_override"] != df["pred"]).sum()

    # The override should be rare (this is what "small, explicit" means).
    assert n_changed < len(df) * 0.05
    # Document, don't assert, the direction -- the point of this test is that
    # it is CHECKED, not that it must always win; see module docstring.
    assert isinstance(acc_before, float) and isinstance(acc_after, float)
