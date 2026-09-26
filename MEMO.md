# MEMO — Kestrel Service Routing: Replace the Bot

**To:** Ritu Deshpande, Head of D2C Operations
**From:** Kabir Nanda's team (Account Lead)
**Re:** Routing bot replacement — recommendation

## DECISION
Do not switch off the legacy vendor routing bot yet. Move the locally-run classifier into a 1–2 week shadow-mode pilot alongside the existing bot, with human review and no production routing changes initially.

The model reaches 84.6% accuracy on a chronological holdout against `final_team`, compared with 77.2% for the legacy bot against the same outcome. This is promising enough to justify a controlled pilot, but not enough evidence to claim that the bot can be safely retired today.

Do **not** treat the legacy bot's historical routing decisions (`team_label`) as ground truth to match — they are the very source of the transfer problem Meenal flagged. We trained instead against the team that *actually closed* each request (`final_team`), which is the closest thing this data has to ground truth.

## EVIDENCE
On a chronological holdout (3 months of requests the model never trained on):
- **Accuracy: 84.6%** (macro F1 0.85), vs. the legacy bot's own 77.2% "accuracy" against the same true outcome (i.e. its 22.8% historical misroute rate).
- That is roughly a **one-third relative reduction** in misrouting.
- The system already gets the exact scenario the team flagged right: a payment mention (UPI/EMI/card) does not send a leaking-purifier request to Billing — because it learned this from actual outcomes, not keyword rules. We tested a manual keyword override for this and it made things *worse* on validation data, so it was left disabled.
- The dominant remaining error type (76% of mistakes) is requests with genuinely too little information ("please call back regarding purifier") — no model fixes that; the system flags these for a human instead of guessing.

**Note on the 90% target:** the original ask used a 90%+ match-rate threshold, but the supplied historical `team_label` is the legacy bot's own routing decision rather than an independent ground-truth label. This creates a metric-definition problem: optimizing only for agreement with `team_label` could reproduce the same routing behaviour we are trying to improve. The current model reaches 84.6% accuracy against `final_team`, the available post-resolution outcome. The 90% threshold should therefore be revisited with the business team before it is used as a go/no-go criterion.

## MONEY
- **Paid inference cost: ₹0/request.** The model runs locally (linear SVM over local text features) — no external LLM API, no per-request bill, addressing Farhan's concern directly.
- **Not zero total cost:** hosting, storage, monitoring and engineering upkeep are real and deployment-dependent; we have not sized these without a target environment and will not invent a number.
- **Legacy bot licence:** ₹3.2 lakh/year, for reference.
- **Misrouting friction, current bot:** an estimated **₹18.3 lakh over the last 15 months** (≈₹14.7 lakh/year) in transfer handling time (₹305/transfer) and extra customer contacts (₹260/misroute), per ops-policy.pdf's own cost figures — i.e. the bot's mistakes already cost more per year than its licence fee. If friction scales roughly with misroute rate, the ~33% relative error reduction above implies **on the order of ₹4.5–5 lakh/year** in avoided transfer/contact cost — a rough estimate, not a guarantee, and worth re-checking after a live pilot.

## RISK
- Genuinely ambiguous, low-information requests (a real ~12% of volume) will still need a human; the system flags them rather than guessing confidently.
- Repairs is the model's weakest team on precision (0.79) — it's the most common wrong guess when unsure; worth a light manual spot-check policy for Repairs routes early on.
- A small share of wrong predictions are confidently wrong and won't be auto-flagged — recommend a manual audit sample even within "high confidence" routes for the first few weeks.
- **Human review is triggered automatically** for ~8% of traffic (confidence below the high band), which historically captures a large share of the actual errors.

## NEXT WEEK
1. Run a shadow-mode pilot: score live requests with the new model alongside the existing bot for 1–2 weeks, without acting on its predictions yet.
2. Have Meenal's team spot-check high-confidence Repairs and Billing routes and review the automatically flagged low-confidence requests.
3. Decide the production hosting/monitoring setup and agree with Ritu/Tanmay on the final success metric and go/no-go threshold before considering bot retirement.