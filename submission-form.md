# Submission Form — Kestrel Routing Intelligence

## What did you build?

A local, reproducible routing-intelligence system for Kestrel service requests: a calibrated linear-SVM classifier (char TF-IDF + structured features) predicting the operational team, with confidence scoring, human-readable explanations, historical-similarity evidence, a policy layer, a FastAPI service, a Streamlit operations console, temporal evaluation, error analysis, and operations/friction analytics — all runnable with no external LLM or paid API.

The seven canonical teams are:

- Installs & Demo
- Repairs
- Filters & Consumables
- Billing
- Returns & Replacement
- Warranty Claims
- Product Advice

---

## What business decision does this support?

The recommended decision is **not to switch off the legacy vendor routing bot immediately**.

The new model should first be run in **shadow mode for 1–2 weeks** alongside the existing bot, with human review and no production routing changes initially.

The model reaches **84.59% accuracy against `final_team` on the chronological holdout**, compared with **77.17% for the legacy bot against the same outcome**.

The existing bot licence costs **₹3.2 lakh/year**. The new model has **₹0/request paid inference cost** because inference runs locally with no external LLM or paid API.

The current validation evidence is sufficient to justify a controlled pilot, but not to claim that the legacy bot can safely be retired today.

---

## What is the target label, and why?

`final_team` from `resolution_log.csv` (the team that actually closed the request), not `train.csv`'s `team_label` (the legacy bot's own initial decision).

Evidence: `team_label` disagrees with `final_team` on approximately **22.8%** of rows, and for the **1,239 requests** merely mentioning a payment word, the bot sent **66% to Billing** while only **23% actually belonged there** — including the exact "paid by UPI, purifier leaking → should be Repairs" example from the assignment brief, which is also present in the training data with `team_label=Billing` and `final_team=Repairs`.

Training on `team_label` would reproduce the bot's own routing behaviour rather than learning from the observed resolution outcome.

Full reasoning: `reports/validation_report.md` §2.

---

## What did you try?

Four experiments are recorded in `reports/model_comparison.csv`:

1. Word TF-IDF baseline — **0.8379 accuracy**
2. Character TF-IDF — **0.8407 accuracy**
3. Character TF-IDF + structured features — **0.8375 accuracy** in the experiment comparison; this is the shipped variant
4. Stratified random-split robustness check — **0.8417 accuracy**

The primary reported production-validation result comes from the chronological holdout, where the shipped model reaches **0.8459 accuracy / 0.8462 macro F1**.

We also tested a hand-written policy override rule that force-routes payment-mentioning Billing predictions with fault language to Repairs.

---

## What changed from the initial plan?

The architecture in the assignment brief assumes the classifier needs an explicit policy layer to enforce "payment mention ≠ billing."

In practice, once trained on `final_team` (the available post-resolution outcome) rather than `team_label`, the classifier already handles this scenario well. It achieves **88.98% accuracy on the payment-mention subset**, compared with **84.59% overall temporal-holdout accuracy**.

The policy layer was therefore scaled down accordingly: it is implemented, tested, and available in `src/policy.py`, but the hard override is **disabled by default** because it was evaluated and did not improve validation performance.

---

## What did you throw away?

- The hard-coded payment → Repairs override rule: tested on the temporal holdout, it fired on only **1 of 2,135 requests** and got that one wrong, changing accuracy from **0.8459 → 0.8453**. It is kept in the codebase disabled, with the experiment covered by `tests/test_policy.py::test_hard_override_empirically_does_not_help_on_temporal_holdout`.

- A sentence-transformer embedding model for historical retrieval: no network access to a model hub was available in this environment, so retrieval runs on TF-IDF cosine similarity with automatic fallback logic already written for a sentence-transformer if one becomes available.

- Treating `team_label` as a usable structured feature: excluded because it represents the legacy bot's own routing decision and would not be an independent feature for a new incoming request once that bot is retired.

---

## Expected hidden score / metric / why / estimate

The primary offline metric used for this project is **accuracy against `final_team` on a chronological holdout**, with macro F1 reported alongside it.

The chronological split is:

- Training: **2025-04-01 to 2026-03-31 — 8,687 rows**
- Holdout: **2026-04-01 to 2026-06-30 — 2,135 rows**

Observed temporal-holdout results:

- Accuracy: **84.59%**
- Macro F1: **0.8462**

The legacy bot's historical accuracy against the same `final_team` outcomes is **77.17%**.

The original assignment discussed a 90%+ threshold, but `team_label` is the legacy bot's own routing decision rather than an independent ground-truth label. Therefore, the 90% threshold should be revisited with the business team before being used as a go/no-go criterion.

**Expected hidden-score estimate:** approximately **mid-80% accuracy**, with **~84–85%** as the central estimate based on the chronological holdout.

This is an estimate, not a guarantee, because the hidden test distribution may differ from the validation period.

---

## Actual metrics

Temporal holdout:

- Accuracy: **0.8459**
- Macro F1: **0.8462**

Legacy bot against `final_team`:

- Accuracy: **0.7717**
- Historical misroute rate: **22.83%**

Full per-team breakdown: `reports/per_team_metrics.csv`

Confusion matrix: `reports/confusion_matrix.png`

Error analysis: `reports/error_analysis.csv`

---

## How do you know it works?

The system was evaluated using a chronological holdout rather than relying only on a random split.

The holdout contains requests from a later time period that were not used to train the validation model. The primary result is **84.59% accuracy / 0.8462 macro F1**.

The implementation is also covered by automated tests for:

- text preprocessing
- temporal split correctness
- target-label sanity (`final_team` vs `team_label`)
- policy-signal generation
- empirical evaluation of the disabled hard override
- predictor schema and validation
- missing-field handling
- retrieval fallback
- FastAPI request/response behaviour
- error cases
- `predictions.csv` submission-artifact schema

The current suite contains **28 passing tests and 1 unrelated httpx/Starlette deprecation warning**.

---

## Known gaps / failure modes

- **76% of remaining holdout errors** are requests with genuinely insufficient information, such as "please call back regarding purifier." These cannot reliably be resolved from text alone.
- **Repairs has the lowest precision at approximately 0.79**, making it a useful focus for targeted review during a pilot.
- A small share of incorrect predictions are confidently wrong and therefore are not automatically sent to human review.
- Confidence is bimodal rather than smoothly graded.
- Historical retrieval currently uses TF-IDF rather than semantic embeddings in this environment.
- Hosting, storage, monitoring and engineering-maintenance costs have not been sized because the target production environment has not yet been specified.

---

## What changed or narrowed from the original ask?

The original ask proposed a 90%+ routing match threshold and described switching off the existing bot if that threshold was reached.

The project narrowed that decision because the supplied `team_label` is the legacy bot's own routing decision, while `final_team` records the eventual resolution team.

The resulting recommendation is therefore:

1. Evaluate the new model against `final_team`.
2. Run a shadow-mode pilot alongside the existing bot.
3. Review human-review and high-confidence routing behaviour.
4. Agree on the final business metric and go/no-go threshold before considering bot retirement.

The hard-coded payment policy was also narrowed from an active override to a tested, disabled-by-default policy layer because the empirical override test made validation performance slightly worse.

---

## Handoff / data bugs / shortcuts / untrusted columns / bad rows

### Data / label issue

The most important data issue is that `train.csv`'s `team_label` is identical to `resolution_log.first_team` and represents the legacy bot's initial routing decision rather than an independent ground-truth label.

`resolution_log.final_team` was therefore used as the prediction target.

### Team renames

The supplied team taxonomy contains two historical renames effective **2026-01-15**:

- `Installations` → `Installs & Demo`
- `Consumables` → `Filters & Consumables`

These are canonicalized automatically.

### Timestamp issue

Resolution-log timestamps associated with legacy-Zoho rows are in UTC rather than IST. They are corrected in `src/ops_analytics.py` for analytics only and are not used as modelling features.

### Encoding issue

Approximately 4.4% of training rows contain legacy-Zoho mojibake/encoding corruption. Character TF-IDF was selected partly because it is more robust to spelling variation, abbreviations, and such text corruption.

### Untrusted / excluded column

`team_label` is explicitly excluded as a model feature because it represents the legacy bot's own routing decision and would reproduce the behaviour being evaluated.

### Shortcuts deliberately avoided

- No paid LLM/API calls.
- No external inference dependency.
- No use of `final_team` as an input feature.
- No random-only validation as the primary evaluation.
- No hard-coded payment keyword override in production inference.

---

## What did you deliberately leave out?

- No paid external LLM/API inference.
- No automatic production replacement of the legacy bot.
- No active payment → Repairs hard override.
- No claim that the 90% threshold has been achieved.
- No invented hosting or infrastructure cost.
- No public repository upload of the supplied customer/operational CSV data.
- No model dependence on `final_team` or other post-resolution information at prediction time.
- No semantic embedding dependency for the default retrieval path.

---

## What is the human-review policy?

The system produces a prediction, confidence score, confidence band, runner-up team, human-readable reasons, and historical-similarity evidence.

Low-confidence requests are flagged for human review rather than being treated as equally reliable as high-confidence routes.

Approximately **8% of production test traffic** falls below the high-confidence band and is flagged for review.

Because some incorrect predictions can still be confidently wrong, the recommended pilot also includes a manual audit sample of high-confidence Repairs and Billing routes.

---

## Money / cost

- **Paid inference cost:** **₹0/request**
- **Legacy bot licence:** **₹3.2 lakh/year**
- Hosting, storage, monitoring and engineering upkeep are real but deployment-dependent and have intentionally not been invented without a target environment.

The current bot's estimated misrouting friction is approximately **₹18.3 lakh over the last 15 months** (≈**₹14.7 lakh/year**) using the transfer-handling and extra-contact cost figures from the supplied operations policy.

If friction scales roughly with the observed relative reduction in routing errors, the model's approximately one-third relative error reduction implies an estimated **₹4.5–5 lakh/year** in avoided transfer/contact cost.

This is a rough estimate, not a guarantee, and should be re-checked after a live pilot.

---

## One prediction cost + monthly ~₹700 arithmetic

The model uses no paid external inference API, so the direct paid inference cost is:

**₹0 per prediction.**

For the assignment's requested monthly-cost arithmetic:

**₹700/month × 12 months = ₹8,400/year**

This is an arithmetic planning example, not a measured production infrastructure cost.

The actual hosting, storage, monitoring and engineering cost depends on the final deployment environment and should be priced before production cutover.

---

## Monday handoff — 3 things

1. Run a **1–2 week shadow-mode pilot** alongside the existing bot without acting on the new model's predictions.
2. Have Meenal's team **spot-check high-confidence Repairs and Billing routes** and review automatically flagged low-confidence requests.
3. Decide the **production hosting/monitoring setup** and agree with Ritu/Tanmay on the final success metric and go/no-go threshold before considering bot retirement.

---

## AI use / tools / models / help / discarded work

AI tools were used during development for implementation assistance, debugging, documentation drafting, code review, and reasoning about model architecture and evaluation.

The primary AI tools used were **Claude and ChatGPT**.

The final routing model itself does **not** depend on an external LLM or paid API.

The modelling stack is local:

- Python
- pandas
- scikit-learn
- Character TF-IDF
- LinearSVC
- calibrated probabilities
- FastAPI
- Streamlit
- pytest

AI-assisted/discarded work included:

- evaluation of the explicit payment-routing override
- exploration of semantic sentence-transformer retrieval
- implementation and testing of the policy layer
- review of model-validation methodology
- documentation and submission-material drafting

The payment override was empirically tested and disabled. Sentence-transformer retrieval was not used as the default because the environment could not download the model from the model hub.

No paid model/API calls were made.

The screen recording will disclose the AI-assisted development workflow and the approaches that were tried and discarded.

---

## Extra

The project deliberately keeps the prediction path and operations analytics separate.

`resolution_log.csv` is used for historical outcome analysis, transfer-friction analysis, and operations visibility, but post-resolution information is not fed into the model as an input feature.

The service exposes the same prediction path through FastAPI and Streamlit so that the demonstrated UI and JSON endpoint do not contain separate routing logic.

---

## Screen recording

**Recording link:** `[ADD RECORDING LINK]`

The recording will cover:

1. What was tried.
2. What changed after inspecting the data.
3. Why `final_team` was used instead of `team_label`.
4. What was tested and discarded.
5. The working API/UI.
6. The validation result and limitations.
7. The final business decision.

---

## Public Drive link

**Public Drive / submission link:** `[ADD PUBLIC DRIVE LINK]`

---

## Honest development time

**Approximate hands-on development time:** `[ADD HOURS]`

This should reflect the actual time spent on implementation, testing, debugging, evaluation, documentation, and submission preparation.

---

## GitHub

**Repository:** https://github.com/Adro05/kernel-routing

The repository does not contain the supplied client CSV data, generated model artifacts, or the client-data-derived `predictions.csv`.

---

## How to reproduce

See `README.md` → **Installation** and **Running Instructions**.

The intended workflow is:

```bash
python -m src.train
python -m src.evaluate
python -m src.error_analysis
python -m src.predict
pytest tests/ -v