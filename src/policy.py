"""
Policy layer.

ops-policy.pdf §3 states the key rule: "A request belongs to Billing only
when the problem is the payment itself ... A customer mentioning that they
have paid does not make it a billing request." The assignment brief warns
against naive keyword routing that would violate this.

We tested whether an explicit keyword OVERRIDE is needed on top of the ML
model (LinearSVC on char TF-IDF + structured features, trained against
`final_team`) to enforce this rule. Empirically, it is not:

  - On the temporal holdout, requests that mention a payment word (paid,
    UPI, EMI, card, payment) are classified by the model at 88.98% accuracy
    -- HIGHER than the 84.59% overall holdout accuracy -- because the model
    was trained on `final_team` (the correct destination), not on the
    legacy bot's own `team_label` decisions, and so it already learned that
    a payment mention alone does not imply Billing.
  - A candidate hard override rule ("if predicted Billing but the text has a
    fault word like leak/burnt/error/noise and no strong-billing word like
    invoice/GST/refund, force Repairs") was implemented and run against the
    same holdout. It fired on exactly 1 of 2135 requests, and it was WRONG
    on that one case, so net holdout accuracy went from 0.8459 to 0.8453.
    See tests/test_policy.py for the reproducible check.

Conclusion (documented per the assignment's "evaluate before claiming it
helps" requirement): no hard payment-vs-billing override is applied. The
function below is kept because it is useful for EXPLANATIONS (surfacing to
an operator *why* a payment mention did not drive the routing decision) and
because it is unit-tested and available if a future audit of a specific
product line shows it is needed -- but `apply_hard_override` defaults to
disabled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PAYMENT_PATTERN = re.compile(r"\b(paid|upi|emi|card payment|payment)\b", re.IGNORECASE)
STRONG_BILLING_PATTERN = re.compile(r"\b(invoice|gst|double charge|refund|emi conversion|coupon)\b", re.IGNORECASE)
FAULT_PATTERN = re.compile(
    r"\b(leak|leaking|burnt|error|noise|tripping|not working|not turning on|blank|broken|faulty|stopped)\b",
    re.IGNORECASE,
)
CONSUMABLE_PATTERN = re.compile(r"\b(filter|candle|membrane|jar|brush|blade|amc kit|spare)\b", re.IGNORECASE)


@dataclass
class PolicySignal:
    name: str
    note: str


def payment_mention_signal(text: str, predicted_team: str) -> PolicySignal | None:
    """Explanatory (non-overriding) signal: a payment word is present.

    Returns a human-readable note about how the policy rule was applied,
    for use in the explanation layer. Does not change the prediction.
    """
    if not PAYMENT_PATTERN.search(text or ""):
        return None
    if predicted_team == "Billing":
        if STRONG_BILLING_PATTERN.search(text or ""):
            return PolicySignal(
                "payment_mention",
                "The request mentions a payment problem (invoice/GST/refund/EMI conversion/coupon), "
                "which is what Billing actually handles.",
            )
        return PolicySignal(
            "payment_mention_needs_check",
            "The request mentions a payment method, but no specific billing problem (invoice/GST/"
            "refund) was detected. Per ops policy, mentioning payment alone should not drive routing "
            "to Billing -- worth a quick check that this is genuinely a billing issue.",
        )
    return PolicySignal(
        "payment_mention_not_billing",
        "The request mentions a payment method, but per ops policy a payment mention alone does not "
        "make this a Billing case; the model routed it based on the underlying issue instead.",
    )


def fault_vs_consumable_signal(text: str, predicted_team: str) -> PolicySignal | None:
    has_fault = bool(FAULT_PATTERN.search(text or ""))
    has_consumable = bool(CONSUMABLE_PATTERN.search(text or ""))
    if has_fault and has_consumable and predicted_team in ("Repairs", "Filters & Consumables"):
        return PolicySignal(
            "fault_vs_consumable",
            "The request mentions both a fault symptom and a consumable/spare part; Repairs and "
            "Filters & Consumables overlap here, so this may be worth a second look.",
        )
    return None


def apply_hard_override(text: str, predicted_team: str, enabled: bool = False) -> str:
    """Candidate hard override, DISABLED by default -- see module docstring.

    Kept as an explicit, testable function (not a silent monkey-patch) so a
    future re-evaluation can flip `enabled=True` if a larger validation set
    shows it helps. It must not be enabled without re-running
    tests/test_policy.py against a current holdout.
    """
    if not enabled:
        return predicted_team
    if predicted_team == "Billing" and FAULT_PATTERN.search(text or "") and not STRONG_BILLING_PATTERN.search(text or ""):
        return "Repairs"
    return predicted_team


def collect_policy_signals(text: str, predicted_team: str) -> list[PolicySignal]:
    signals = []
    for fn in (payment_mention_signal, fault_vs_consumable_signal):
        sig = fn(text, predicted_team)
        if sig is not None:
            signals.append(sig)
    return signals
