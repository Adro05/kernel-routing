# \# 🧭 Kestrel Routing Intelligence

#

# \*\*A local, reproducible routing-intelligence system for Kestrel Home service requests\*\*

#

# Predicts which of Kestrel's 7 operational teams should handle a new inbound service request — evaluating a local alternative to the legacy vendor routing bot — with confidence scoring, human-readable explanations, historical evidence, and an operations analytics console.

#

# \---

#

# \*\*Tech stack:\*\* `Python 3.12` · `scikit-learn` (LinearSVC) · `FastAPI` · `Streamlit` · `pandas` · `pytest`

#

# \*\*Model:\*\* `Character TF-IDF + structured features` · \*\*Inference:\*\* `Fully local — no external LLM or paid API`

#

# \*\*Status:\*\* `Internship submission — implemented, locally validated, and tested`

#

# \---

#

# \## 📖 Contents

#

# \[Project Overview](#project-overview) · \[Business Problem](#business-problem) · \[Architecture](#architecture) · \[Why This Model](#why-this-model) · \[Data](#data) · \[Team Taxonomy](#team-taxonomy) · \[Training \& Evaluation Methodology](#training--evaluation-methodology) · \[Validation Results](#validation-results) · \[Error Analysis](#error-analysis) · \[API Usage](#api-usage) · \[Streamlit Usage](#streamlit-usage) · \[Repository Structure](#repository-structure) · \[Installation](#installation) · \[Running Instructions](#running-instructions) · \[Testing](#testing) · \[Reproducibility](#reproducibility) · \[Limitations](#limitations) · \[Cost Considerations](#cost-considerations) · \[Security \& Data Handling](#security--data-handling) · \[Submission Deliverables](#submission-deliverables)

#

# \---

#

# \## Project Overview

#

# Kestrel Home Appliances runs a service desk across IVR, chat, WhatsApp and email. A vendor routing bot currently assigns each new request to one of 7 teams; agents transfer requests that land in the wrong place. This project replaces that bot with a locally-run classifier that predicts a team, explains why, shows supporting historical evidence, flags ambiguous cases for human review, and includes an operations-analytics view of routing friction.

#

# \## Business Problem

#

# Per the email thread supplied with this assignment, Kestrel's ops team (Meenal Joshi) reports that transfers eat a large share of the service desk's day, with specific pain points: payment-mention requests wrongly landing in Billing, product breakdowns landing in Filters \& Consumables, and low-information requests ("please call me about my purifier") that can't be routed from text alone. Finance (Farhan Sheikh) wants the new system's ongoing cost in writing before switching off the ₹3.2 lakh/year bot licence.

#

# \*\*A critical data finding drives this whole project.\*\* `train.csv` ships a `team\_label` column, but it is not ground truth — it is the \*legacy bot's own initial routing decision\*, confirmed identical to `resolution\_log.first\_team` for all 10,822 rows. `resolution\_log.final\_team` is the team that actually closed the request, after any agent transfers.

#

# | | `team\_label` | `final\_team` |

# |---|---|---|

# | What it represents | The legacy bot's own routing guess, at request creation | The team that actually resolved the request |

# | Source | `train.csv` (= `resolution\_log.first\_team`) | `resolution\_log.csv` |

# | Available at prediction time? | Yes, but it's the thing being replaced | No — it's the outcome, known only after the fact |

# | Used as a model feature? | \*\*Never\*\* | N/A (this is the \*target\*, not a feature) |

# | Used as the prediction target? | No | \*\*Yes\*\* |

#

# `team\_label` disagrees with `final\_team` on \*\*\~22.8%\*\* of requests, and the disagreement is specifically biased on payment mentions (66% routed to Billing vs. only 23% actually belonging there). \*\*This project therefore trains against `final\_team`, not `team\_label`\*\* — training on the bot's own decisions would simply reproduce its bug. Full evidence trail: `reports/validation\_report.md` §2.

#

# \## Architecture

#

# ```

# REQUEST

# &#x20; ↓

# Input validation / normalization        (app/predictor.py, src/data.py)

# &#x20; ↓

# Supervised routing model                (LinearSVC, char TF-IDF + structured features)

# &#x20; +

# Historical similarity retrieval         (src/retrieval.py — TF-IDF, sentence-transformer optional)

# &#x20; +

# Policy checks                           (src/policy.py — explanatory, tested override disabled by default)

# &#x20; ↓

# Routing decision layer                  (app/predictor.py::RoutingPredictor)

# &#x20; ↓

# Prediction + confidence + explanation + evidence

# &#x20; ↓

# FastAPI API (app/api.py)  +  Streamlit ops console (app/ui.py)

# ```

#

# FastAPI and Streamlit both call the same `RoutingPredictor` class — prediction logic is never duplicated.

#

# \## Why This Model

#

# Character `(3,5)`-gram TF-IDF is robust to the spelling mistakes, abbreviations, and legacy-encoding corruption (e.g. `Ã©`, `Ã¢â‚¬Â¦` — mojibake from a legacy Zoho export, see below) present in \~4.4% of the training text. A linear SVM (`LinearSVC`) trains fast, is interpretable at the team-scope level, and needs no GPU or external API. Wrapped in `CalibratedClassifierCV` (sigmoid/Platt scaling via internal 5-fold CV), it produces genuinely calibrated probabilities rather than raw decision-function values. No LLM is used anywhere in the routing path — see \[Cost Considerations](#cost-considerations).

#

# \## Data

#

# | File | Purpose |

# |---|---|

# | `data/train.csv` | 10,822 historical requests, 2025-04-01 to 2026-06-30, incl. legacy `team\_label` |

# | `data/resolution\_log.csv` | `first\_team` / `final\_team` / `transfers` / `resolved\_at` per request (`final\_team` = actual target) |

# | `data/test\_unlabelled.csv` | 2,178 requests to score, 2026-07-01 to 2026-09-30, no labels |

# | `data/teams.csv` | Team scopes and the 2026-01-15 rename mapping |

# | `data/sample\_submission.csv` | Expected output format for scoring `test\_unlabelled.csv` |

#

# Data quality notes (full detail in `reports/validation\_report.md` §1): no missing values, no exact duplicates, 2 team renames mid-history (cleanly split, no overlap), \~4.4% of rows have legacy-Zoho mojibake, resolution-log timestamps for legacy-Zoho rows are in UTC (not IST) and are corrected in `src/ops\_analytics.py` for analytics only, never for modelling.

#

# \*\*Data is never committed to this repository\*\* — see \[Security \& Data Handling](#security--data-handling).

#

# \## Team Taxonomy

#

# Installs \& Demo · Repairs · Filters \& Consumables · Billing · Returns \& Replacement · Warranty Claims · Product Advice

#

# (`Installations`→`Installs \& Demo` and `Consumables`→`Filters \& Consumables` renamed 2026-01-15; canonicalized automatically.)

#

# \## Training \& Evaluation Methodology

#

# Chronological (temporal) split is primary: train on 2025-04-01–2026-03-31 (8,687 rows), evaluate on the untouched 2026-04-01–2026-06-30 holdout (2,135 rows). A stratified random split of the same rows is reported as a secondary robustness check only. The shipped production model is refit on all of `train.csv`, but every reported metric below comes from the temporal-holdout-only model. Full methodology: `reports/validation\_report.md`.

#

# \## Validation Results

#

# | Metric | Value |

# |---|---|

# | Temporal holdout accuracy | \*\*84.59%\*\* |

# | Temporal holdout macro F1 | \*\*\~0.846\*\* |

# | Legacy bot's own accuracy against `final\_team` | 77.17% (its 22.83% misroute rate) |

#

# These numbers reflect the local evaluation described above. \*\*This does not claim to meet the originally-discussed 90% bar\*\* — that figure was framed against the legacy bot's own (biased) labels; see `MEMO.md` for the full discussion of why match-rate against `final\_team` is the metric that matters operationally, and where 84.59% leaves room to improve.

#

# Full experiment comparison: `reports/model\_comparison.csv`. Per-team metrics: `reports/per\_team\_metrics.csv`. Confusion matrix: `reports/confusion\_matrix.png`.

#

# \## Error Analysis

#

# 329/2,135 (15.4%) holdout errors, categorized in `reports/error\_analysis.csv`. Dominant category (76%): genuinely insufficient request text ("please call back regarding purifier") — matches the ops team's own complaint in the email thread. Full breakdown: `reports/validation\_report.md` §7.

#

# \## API Usage

#

# ```bash

# uvicorn app.api:app --reload --port 8000

# ```

#

# ```bash

# curl -X POST http://localhost:8000/predict \\

# &#x20; -H "Content-Type: application/json" \\

# &#x20; -d '{"request\_text": "My water purifier is leaking from the bottom",

# &#x20;      "product\_family": "Water Purifier", "warranty\_status": "in\_warranty",

# &#x20;      "channel": "whatsapp"}'

# ```

#

# Response (fields as actually implemented):

#

# ```json

# {

# &#x20; "predicted\_team": "Repairs",

# &#x20; "confidence": 0.94,

# &#x20; "confidence\_band": "high",

# &#x20; "review\_required": false,

# &#x20; "margin": 0.91,

# &#x20; "runner\_up": {"team": "Returns \& Replacement", "confidence": 0.03},

# &#x20; "reasons": \["The request contains the signal word(s) \\"leak\\", \\"leaking\\".", "..."],

# &#x20; "similar\_cases": \[{"request\_text": "...", "team": "Repairs", "similarity": 0.87}]

# }

# ```

#

# `GET /health` for a liveness check. Missing optional fields (`product\_family`, etc.) degrade gracefully; empty/missing `request\_text` returns `422` with a clear message.

#

# \## Streamlit Usage

#

# ```bash

# streamlit run app/ui.py

# ```

#

# Four pages: \*\*Route Request\*\* (single-request routing with explanation + evidence), \*\*Operations\*\* (workload, transfer friction, monthly volume, resolution time — from `resolution\_log.csv`, never fed into the model), \*\*Model Performance\*\* (experiment comparison, per-team metrics, confusion matrix, error categories), \*\*About / Model Card\*\* (renders `MODEL\_CARD.md`).

#

# \## Repository Structure

#

# ```

# kestrel-routing/

# ├── app/                    # thin front-end layer

# │   ├── api.py              # FastAPI service (POST /predict, GET /health)

# │   ├── predictor.py        # shared RoutingPredictor — the single prediction path

# │   └── ui.py                # Streamlit ops console (4 pages)

# ├── src/                    # modelling, evaluation and supporting logic

# │   ├── data.py              # loading, mojibake cleanup, target derivation, temporal split

# │   ├── features.py          # TF-IDF + structured feature encoders

# │   ├── train.py              # experiment comparison + final model training

# │   ├── evaluate.py          # confusion matrix, per-team metrics, confidence bands

# │   ├── error\_analysis.py    # error categorization on the temporal holdout

# │   ├── predict.py            # scores test\_unlabelled.csv -> predictions.csv

# │   ├── policy.py              # explanatory + (disabled-by-default) tested policy rule

# │   ├── retrieval.py           # historical similarity (TF-IDF, sentence-transformer optional)

# │   ├── explanations.py       # human-readable reason generation

# │   └── ops\_analytics.py      # workload/friction analytics, separate from the model

# ├── models/                  # generated artifacts (gitignored)

# ├── reports/                 # generated reports/CSVs/plots

# ├── tests/                    # pytest suite

# ├── data/                     # supplied CSVs (gitignored, see Security)

# ├── README.md · MEMO.md · MODEL\_CARD.md · submission-form.md · SCREEN\_RECORDING\_SCRIPT.md

# └── requirements.txt · pyproject.toml · .gitignore · .env.example

# ```

#

# \## Installation

#

# ```bash

# python -m venv .venv

# pip install -r requirements.txt

# ```

#

# Activate the virtual environment first:

#

# ```bash

# \# macOS / Linux

# source .venv/bin/activate

#

# \# Windows (PowerShell)

# .\\.venv\\Scripts\\Activate.ps1

# ```

#

# Place the 5 supplied CSVs into `data/` (see \[Data](#data) for filenames). No external model download is required; retrieval uses TF-IDF by default (an optional `sentence-transformers` line is commented in `requirements.txt` if you want to try semantic retrieval — it downloads `all-MiniLM-L6-v2` from the Hugging Face hub on first use, and the code falls back to TF-IDF automatically if that's unavailable).

#

# \## Running Instructions

#

# ```bash

# python -m src.train              # trains all 4 experiments + final model, writes models/ and reports/model\_comparison.csv

# python -m src.evaluate           # confusion matrix, per-team metrics, confidence bands

# python -m src.error\_analysis     # reports/error\_analysis.csv

# python -m src.predict            # scores data/test\_unlabelled.csv -> predictions.csv (repo root)

# uvicorn app.api:app --reload     # API on :8000

# streamlit run app/ui.py          # ops console

# ```

#

# \## Testing

#

# ```bash

# pytest tests/ -v

# ```

#

# \*\*28 passed, 1 warning\*\* (an unrelated `httpx`/Starlette deprecation notice — not a project issue).

#

# Covers: text preprocessing, temporal split correctness, target-label sanity (`final\_team` vs `team\_label`), policy-signal generation, the (disabled-by-default) hard override and its empirical evaluation, predictor schema/validation/missing-field handling, retrieval fallback, FastAPI request/response behaviour including error cases, and the `predictions.csv` submission-artifact schema. All tests execute real code paths against the actual trained model artifacts — none merely assert `True`.

#

# \## Reproducibility

#

# Fixed random seed (42) throughout. `python -m src.train \&\& python -m src.evaluate \&\& python -m src.error\_analysis \&\& python -m src.predict` regenerates every number in this README, `MEMO.md`, `MODEL\_CARD.md` and `reports/`, from the supplied CSVs, deterministically. No paid external API is required anywhere in the pipeline.

#

# \## Limitations

#

# See `MODEL\_CARD.md` "Known limitations" — in short: text-insufficiency errors can't be fixed by any classifier, confidence is bimodal rather than smoothly graded (a few wrong predictions are confidently wrong), Repairs has the lowest precision, and retrieval quality depends on TF-IDF rather than semantic embeddings in this environment.

#

# \## Cost Considerations

#

# Paid model/API inference cost for this evaluation: \*\*₹0/request\*\* (fully local, no external LLM or paid API). Hosting, storage, monitoring and engineering maintenance are real but deployment-dependent and are not invented here. Legacy bot licence, for comparison: \*\*₹3.2 lakh/year\*\*. Full breakdown and friction-cost discussion: `MEMO.md`.

#

# \## Security \& Data Handling

#

# Per `ops-policy.pdf` §10, customer and operational data must not be published or uploaded to public repositories. `.gitignore` excludes all supplied CSVs, generated model artifacts, and the client-data-derived `predictions.csv` submission file — none of these are committed to the public repository. Example text used in this README and other docs is drawn from the assignment's own worked example, not raw customer PII.

#

# \## Submission Deliverables

#

# `README.md` (this file), `MEMO.md`, `MODEL\_CARD.md`, `submission-form.md`, `SCREEN\_RECORDING\_SCRIPT.md`, `reports/validation\_report.md`, `reports/model\_comparison.csv`, `reports/per\_team\_metrics.csv`, `reports/confusion\_matrix.png`, `reports/error\_analysis.csv`, `predictions.csv` (scored `test\_unlabelled.csv`, repo root, gitignored as client-data-derived), full `app/` + `src/` implementation, `tests/` (28 passing), `requirements.txt`, `.gitignore`.
