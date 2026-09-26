# \# Kestrel Home — Submission Form

#

# \## 1. What did you build, and what business decision does it support? State the number and the rupees.

#

# I built a locally-run service-request routing system that predicts which of Kestrel Home's seven operational teams should handle a new request. It includes a character TF-IDF + linear classifier, confidence/review logic, human-readable explanations, similar historical cases, a JSON API, and a Streamlit UI.

#

# On a chronological holdout of 2,135 requests, the model achieved 84.59% accuracy and 0.8462 macro F1 against `final\_team`, the available post-resolution outcome. The historical legacy bot achieved 77.17% accuracy against the same outcome, corresponding to a 22.83% historical misroute rate.

#

# The business decision I support is not to switch off the legacy bot immediately, but to run the new system in a 1–2 week shadow-mode pilot before making a production replacement decision.

#

# The legacy bot licence is ₹3.2 lakh/year. The historical misrouting analysis estimates approximately ₹18.3 lakh of transfer/contact friction over the 15 months in the supplied data, using the cost assumptions in the operations policy. The new model has ₹0 paid inference cost per request because it runs locally, although production hosting and maintenance costs would still need to be sized.

#

# \---

#

# \## 2. What score do you expect `predictions.csv` to get on the hidden outcomes, on which metric, and why that metric? Say how you estimated it.

#

# I expect approximately 84–85% accuracy on the hidden outcomes, with accuracy as the primary metric and macro F1 as a secondary metric because there are seven routing teams and I want performance across teams to matter rather than relying only on the largest class.

#

# My estimate comes from a chronological holdout that mimics the production setting better than a random split: the model trained on 8,687 earlier requests and was evaluated on 2,135 later requests. It achieved 84.59% accuracy and 0.8462 macro F1. A random-split robustness check produced 84.22% accuracy.

#

# The estimate is uncertain because the hidden test outcomes are unavailable and the historical `final\_team` label is a post-resolution outcome rather than an independently verified initial-routing ground truth.

#

# \---

#

# \## 3. How do you know it works? How you validated, on what split, error rate, and the kind of case it gets wrong.

#

# I used chronological validation rather than relying only on a random train/test split. The model trained on 8,687 earlier requests and was evaluated on the subsequent 2,135-request holdout.

#

# Results: 84.59% accuracy, 0.8462 macro F1, and 329/2,135 errors (15.4%). Per-team precision, recall, F1, and a confusion matrix were also generated. The implementation has 28 passing automated tests covering the API, prediction behavior, data handling, policy logic, confidence/review behavior, explanations, retrieval, and `predictions.csv`.

#

# The largest error category was genuinely insufficient-context requests, accounting for about 76% of holdout errors. Examples include requests such as "please call back regarding purifier", where the text does not contain enough information to reliably distinguish the correct team. The system therefore exposes confidence and a human-review flag rather than pretending these cases are certain.

#

# Repairs has the lowest precision at approximately 0.79, so it is an area I would monitor closely during a pilot.

#

# \---

#

# \## 4. Did you change, narrow, or push back on the client's ask? What, when, and why?

#

# Yes. I pushed back on treating the original 90%+ match-rate requirement as an immediate go/no-go criterion.

#

# The supplied `team\_label` is the legacy bot's historical routing decision, while `final\_team` represents the team that ultimately closed the request. Optimizing only to match `team\_label` could reproduce the same routing behavior that the analysis shows has historically required transfers.

#

# I therefore trained and evaluated against `final\_team`, the available post-resolution outcome, and used chronological validation. The resulting 84.59% accuracy is below the original 90% threshold, so I did not claim that the threshold was achieved.

#

# I narrowed the operational recommendation to a 1–2 week shadow-mode pilot with human review, followed by agreement on the final success metric and go/no-go threshold.

#

# \---

#

# \## 5. What is wrong with what you are handing us, or with the data we handed you?

#

# The main data/measurement issue is that `team\_label` and `final\_team` represent different stages of the routing process. `team\_label` is the legacy bot's initial routing decision, while `final\_team` is the post-resolution destination, so neither should automatically be treated as an independently verified "correct initial route".

#

# The historical dataset also contains genuinely low-information requests, which limits how accurately any text-based routing model can classify them.

#

# I treated `resolution\_log.csv` as an evaluation/operations source rather than using resolution information as prediction features, to avoid leakage. Historical similarity retrieval is evidence shown to the user and is not used to determine the classifier's label.

#

# The supplied historical data covers approximately 15 months and one product-line setup, so performance on materially different future products, channels, or team definitions is unknown.

#

# I also did not include generated model binaries, private reports, or client data in the public repository.

#

# \---

#

# \## 6. What did you deliberately leave out, and why that rather than something else?

#

# I deliberately left out:

#

# \- External LLM/API inference, because the assignment required a clean-machine workflow without a paid API key and the routing task can be handled locally.

# \- A hard payment-keyword override, because the operations policy says payment language alone does not imply Billing, and testing the override did not improve validation performance.

# \- Automatic production routing, because the validation result is below the original 90% threshold and a small number of predictions remain confidently wrong.

# \- Training on `resolution\_log` fields or other post-resolution information that would not be available at prediction time.

# \- A sentence-transformer dependency for retrieval because the target environment did not have network access to download the model; the system falls back to TF-IDF cosine similarity.

#

# \---

#

# \## 7. Anything you built or found that nobody asked for?

#

# Yes. I added several supporting capabilities to make the system safer and easier to hand off:

#

# \- Confidence bands and an automatic human-review flag for lower-confidence predictions.

# \- Human-readable routing explanations.

# \- Runner-up team and confidence.

# \- Similar historical requests as supporting evidence.

# \- An Operations Analytics view showing historical workload, transfer rates, estimated friction cost, monthly request volume, and resolution-time statistics.

# \- A Model Performance view with temporal validation metrics, per-team metrics, confusion matrix, and error categories.

# \- An explicit model/data-leakage section documenting which fields are and are not used for prediction.

#

# \---

#

# \## 8. What did you use AI for? Which tools and models, where they helped, where they wasted time, what you threw away. Link your three-minute screen recording here.

#

# I used Claude and ChatGPT as development assistants.

#

# Claude was used primarily for implementation assistance, including project structure, Python/API/UI code, tests, documentation, and debugging suggestions. ChatGPT was used primarily for architecture review, validation of the modeling approach, error-analysis reasoning, documentation review, and checking the submission against the assignment requirements.

#

# I reviewed and tested the generated code locally rather than treating AI output as trusted. I discarded or changed approaches when validation did not improve, including a hard payment-keyword override and approaches that risked reproducing the legacy routing labels. Retrieval also uses a TF-IDF fallback rather than depending on an unavailable downloaded embedding model.

#

# No paid AI/API inference was used by the submitted application.

#

# \*\*Screen recording:\*\* \[https://drive.google.com/drive/folders/1zBhvps_NLENol86lhXh9Kelhu0hpZ93c?usp=sharing]

#

# \---

#

# \## 9. Someone picks this up on Monday and you are unreachable. The three things they need to know.

#

# 1\. \*\*Run and validate:\*\* activate the virtual environment, run `python -m pytest tests/`, then start the API with `uvicorn app.api:app --reload --port 8000` and the UI with `streamlit run app/ui.py`.

#

# 2\. \*\*Do not treat 90% as achieved:\*\* the primary chronological holdout result is 84.59% accuracy / 0.8462 macro F1 against `final\_team`. The recommended next step is a 1–2 week shadow-mode pilot with human review.

#

# 3\. \*\*Protect the data boundary:\*\* the files in `data/`, generated models/reports, and `predictions.csv` are intentionally gitignored. Do not publish the client data. Retrain/revalidate if the product mix, team definitions, or routing policy changes.

#

# \---

#

# \## 10. Honest hours spent

#

# 12 hours

#

# \---

#

# \## 11. GitHub Repo Link

#

# https://github.com/Adro05/kernel-routing

#

# \---

#

# \## 12. What does one prediction cost, and what would a month cost at Kestrel's volume? Show the arithmetic. If you used no paid calls, say so.

#

# Paid inference cost: ₹0 per prediction. The submitted model runs locally and does not make external LLM/API calls.

#

# At approximately 700 requests/month:

#

# ₹0 × 700 = ₹0/month in paid inference/API charges.

#

# The assignment also asks for the ₹700/month planning arithmetic:

#

# ₹700/month × 12 = ₹8,400/year.

#

# This ₹8,400 figure is a planning/example figure requested by the assignment, not a measured production hosting cost. Actual hosting, storage, monitoring, and engineering maintenance costs depend on the deployment environment and have not been estimated without a target environment.

#

# \---

#

# \## 13. Public Drive / final submission link

#

# \[https://drive.google.com/drive/folders/1zBhvps_NLENol86lhXh9Kelhu0hpZ93c?usp=sharing]
