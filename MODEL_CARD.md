# Model Card — Kestrel Routing Intelligence

## Purpose
Predicts which of Kestrel's 7 operational teams should handle a new inbound service request (chat/WhatsApp/IVR/email), replacing the legacy vendor routing bot.

## Intended users
Kestrel D2C Operations service-desk agents and supervisors, via the FastAPI `/predict` endpoint and the Streamlit "Route Request" console. Not intended as a fully autonomous router with no human in the loop — see "Human review behavior" below.

## Intended use
Route a *new* request at creation time, using only information available at that moment (request text, product family, warranty status, channel). Not intended for: re-routing already-resolved historical tickets, predicting resolution time, or any decision beyond team assignment.

## Training data
`train.csv` joined with `resolution_log.csv` on `request_id` — 10,822 Kestrel service requests, 2025-04-01 to 2026-06-30. See `reports/validation_report.md` for full data characteristics.

## Target labels
**`final_team`** (from `resolution_log.csv`) — the team that actually closed the request — canonicalized to 7 current team names: Installs & Demo, Repairs, Filters & Consumables, Billing, Returns & Replacement, Warranty Claims, Product Advice.

`team_label` (the legacy bot's own initial decision) is deliberately **not** the target: it disagrees with `final_team` on 22.8% of training rows, and is demonstrably biased (see validation report §2). It is also never used as a model feature.

## Feature types
- Free text: `request_text`, character (3,5)-gram TF-IDF.
- Structured/categorical: `product_family`, `warranty_status`, `channel`, `source` (one-hot).
- Explicitly excluded: `team_label`, `resolution_log.transfers`, `resolution_log.resolved_at`, any information not available at request creation.

## Model architecture
`LinearSVC` (linear SVM, `class_weight="balanced"`) over char-TF-IDF + one-hot structured features, wrapped in `CalibratedClassifierCV` (Platt/sigmoid scaling, 5-fold internal CV) for genuinely calibrated probabilities. Runs entirely locally — no external LLM or paid API in the routing path.

## Evaluation methodology
Chronological (temporal) train/holdout split: train on 2025-04-01 to 2026-03-31 (8,687 rows), evaluate on the never-seen 2026-04-01 to 2026-06-30 holdout (2,135 rows). A stratified random split is reported as a secondary robustness check only. Full detail: `reports/validation_report.md`.

## Actual metrics (temporal holdout, `reports/*.csv`)
- Accuracy: **0.8459**
- Macro F1: **0.8462**
- Per-team precision/recall/F1: `reports/per_team_metrics.csv`
- Confusion matrix: `reports/confusion_matrix.png`
- For comparison, the legacy bot's own "accuracy" against `final_team` on the same data is 77.17% (1 − 22.83% misroute rate).

## Known limitations
- The largest error category (76% of holdout errors) is requests with genuinely insufficient text (e.g. "please call back regarding purifier") — no text model can resolve these; they need a human callback, which is exactly what the review flag is for.
- Confidence is bimodal, not smoothly graded: most wrong predictions get low confidence, but a small number are confidently wrong (see validation report §6) and will not be flagged for review.
- Repairs has the lowest precision (0.79) of any team — it absorbs a disproportionate share of the model's false positives.
- Historical-similarity retrieval falls back to TF-IDF cosine similarity in this environment (no network access to download a sentence-transformer model); semantic nuance in retrieved "similar cases" is more limited than an embedding model would give.
- Evaluated on 15 months of one Kestrel product line set (7 appliance types); performance on new product categories or a materially different request-text style is unknown.

## Ambiguity behavior
When more than one team is plausible (calibrated top-class probability below the empirically-derived thresholds), the system exposes the runner-up team and probability, and lists any detected multi-intent/overlapping-boundary signals in the explanation, rather than silently picking the top class with false confidence.

## Human-review behavior
`review_required = True` whenever the confidence band is "medium" or "low" (top-class probability below 0.261, derived from holdout calibration). This flags ~7.9% of holdout traffic, which contains a large share of the model's actual errors (20.5% accuracy in the medium band vs. 90.0% in the high band).

## Data leakage safeguards
See `reports/validation_report.md` §8: transformers fit train-only; `team_label`/resolution fields never used as features; confidence thresholds frozen after one calibration pass on the holdout, never tuned against unlabelled production data.

## Cost considerations
- Paid inference/API cost: **₹0 per request** (fully local model, no external LLM calls).
- NOT zero total cost: hosting/infrastructure, storage, monitoring and engineering maintenance are deployment-dependent and are not estimated here without a specific deployment target (see MEMO.md).
- Legacy bot licence: ₹3.2 lakh/year (ops-policy.pdf §4), for comparison.

## Operational risks
- Silent model drift if Kestrel's product mix, channels, or team scopes change without retraining (e.g. a new appliance line, another team rename).
- Over-trust in the "high confidence" band, given the small-but-real rate of confidently-wrong predictions.
- Retrieval evidence quality depends on the reference corpus staying representative; it is not refreshed automatically.
