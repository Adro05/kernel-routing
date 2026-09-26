# Screen Recording Script (~3 minutes)

**0:00–0:30 — Problem + system overview**
"Kestrel's legacy routing bot misroutes about 23% of requests — Meenal's team spends a lot of the day just transferring tickets. The key insight: the training data's `team_label` column is the bot's *own* decision, not ground truth — so I trained against `final_team`, the team that actually closed each request. Here's the resulting system: a local classifier plus a FastAPI service and an ops console."

**0:30–1:20 — Live request: input → prediction → confidence → explanation → evidence**
Open Streamlit "Route Request" page. Type: *"My purifier was paid for by UPI and now it is leaking."* Click Route. Show: predicted team **Repairs** (not Billing), confidence and band, the "payment mention doesn't mean Billing" policy note in the reasons, and the 3 similar historical requests (all Repairs). Call out: this is the exact example from the brief, and it's literally in the training data with the legacy bot getting it wrong every time.

**1:20–1:50 — Operations dashboard**
Switch to "Operations" page. Show the misroute-rate metric, the ₹18.3L/15-month friction cost estimate, and the transfer-rate-by-team table (Filters & Consumables and Repairs both over 38% transfer rate under the old bot).

**1:50–2:20 — Model evaluation + experiments**
Switch to "Model Performance" page. Show the 4-experiment comparison table, per-team metrics, and the confusion matrix. Mention: temporal holdout accuracy 84.6%, vs the legacy bot's own 77.2% against the same true outcome.

**2:20–2:45 — What was discarded / limitations**
"I tried a hard keyword override for the payment-vs-billing rule — tested it, it only fired once on the holdout and got it wrong, so it's disabled by default and documented as a passing test. Also: confidence is bimodal, so a few wrong predictions are confidently wrong and won't auto-flag — that's called out honestly in the model card, not hidden."

**2:45–3:00 — Repository + deliverables + reproducibility**
Show the repo tree, `README.md`, and run `python -m src.train && pytest tests/ -v` in a terminal to show it's fully reproducible end to end.
