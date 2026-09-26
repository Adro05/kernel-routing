"""Human-readable, operator-facing explanations for a routing decision.

Deliberately does NOT expose model internals (feature weights, n-gram
contributions). Reasons are built from: the team's documented scope, detected
symptom/keyword signals, policy signals, and similar historical cases. All
strings are template-based from signals that were actually detected -- never
fabricated.
"""
from __future__ import annotations

from src.policy import CONSUMABLE_PATTERN, FAULT_PATTERN, collect_policy_signals

TEAM_SCOPE = {
    "Installs & Demo": "new-product installation, demo and wall-mounting visits",
    "Repairs": "product faults, breakdowns, error codes, noise, leaks and other technician-fixable problems",
    "Filters & Consumables": "filters, candles, membranes, jars, brushes, blades and AMC kits (not faults)",
    "Billing": "invoices, GST, double charges, payment refunds, EMI conversion and coupons",
    "Returns & Replacement": "damaged, wrong or incomplete deliveries and returns/exchanges",
    "Warranty Claims": "warranty/Kestrel Shield registration, coverage questions and claim status",
    "Product Advice": "pre/post-purchase usage questions where no product fault is reported",
}

_SYMPTOM_WORDS = ["leak", "leaking", "burnt", "smell", "error", "noise", "tripping", "not working",
                  "not turning on", "blank", "broken", "faulty", "stopped", "damaged", "missing parts",
                  "wrong item", "install", "demo", "warranty", "claim", "filter", "spare", "amc",
                  "how to", "recommend", "usage", "refund", "invoice", "gst", "coupon", "emi"]


def _detected_symptoms(text: str) -> list[str]:
    text_l = (text or "").lower()
    return [w for w in _SYMPTOM_WORDS if w in text_l]


def build_reasons(
    request_text: str,
    predicted_team: str,
    product_family: str | None,
    confidence_band: str,
    similar_cases: list[dict],
) -> list[str]:
    reasons: list[str] = []

    symptoms = _detected_symptoms(request_text)
    if symptoms:
        shown = ", ".join(f'"{s}"' for s in symptoms[:3])
        reasons.append(f"The request contains the signal word(s) {shown}.")
    else:
        reasons.append("The request text did not contain a clear symptom or intent keyword; "
                        "the model relied mainly on overall wording and request metadata.")

    scope = TEAM_SCOPE.get(predicted_team)
    if scope:
        reasons.append(f"{predicted_team} handles {scope}.")

    if product_family:
        reasons.append(f"The request concerns a {product_family}.")

    for sig in collect_policy_signals(request_text, predicted_team):
        reasons.append(sig.note)

    same_team_cases = [c for c in similar_cases if c["team"] == predicted_team]
    if same_team_cases:
        reasons.append(
            f"{len(same_team_cases)} of the {len(similar_cases)} most similar historical requests "
            f"were also routed to {predicted_team}."
        )
    elif similar_cases:
        other_teams = sorted({c["team"] for c in similar_cases})
        reasons.append(
            "The most similar historical requests were routed to " + ", ".join(other_teams) +
            " -- worth a second look if this doesn't look right."
        )

    if confidence_band == "low":
        reasons.append("Confidence is low: this request looks meaningfully different from a clear "
                        "single-team pattern in the training history.")
    elif confidence_band == "medium":
        reasons.append("Confidence is moderate: the request text is short or ambiguous enough that "
                        "more than one team is plausible.")

    return reasons
