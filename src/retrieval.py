"""
Historical similarity retrieval (evidence, not the primary prediction).

Tries to use a local sentence-transformer for semantic similarity. If that
package/model is unavailable in the runtime environment (no network, not
installed, etc.), falls back to TF-IDF cosine similarity over the same
reference corpus, so the application never crashes for lack of an optional
dependency.

Only rows from the training/reference corpus are ever used as evidence --
never validation/test rows -- and only `final_team` (the ground-truth
routing target) is shown as the historical outcome, never the legacy
`team_label`.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = Path("models")

_SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"


class HistoricalRetriever:
    def __init__(self, reference_df: pd.DataFrame, max_reference: int = 4000):
        # Cap the reference corpus for latency; a random-but-seeded sample is
        # fine since retrieval is supporting evidence, not the classifier.
        if len(reference_df) > max_reference:
            reference_df = reference_df.sample(max_reference, random_state=42)
        self.reference_df = reference_df.reset_index(drop=True)
        self.texts = self.reference_df["request_text_clean"].fillna("").tolist()
        self.mode = "tfidf"
        self._embedder = None
        self._ref_embeddings = None
        self._tfidf = None
        self._ref_tfidf = None
        self._build()

    def _build(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._embedder = SentenceTransformer(_SENTENCE_MODEL_NAME)
            self._ref_embeddings = self._embedder.encode(self.texts, show_progress_bar=False, normalize_embeddings=True)
            self.mode = "sentence-transformer"
        except Exception:
            self.mode = "tfidf"
            self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20000)
            self._ref_tfidf = self._tfidf.fit_transform(self.texts)

    def query(self, text: str, k: int = 3) -> list[dict]:
        text = text or ""
        if self.mode == "sentence-transformer" and self._embedder is not None:
            q = self._embedder.encode([text], show_progress_bar=False, normalize_embeddings=True)
            sims = (self._ref_embeddings @ q[0]).ravel()
        else:
            q_vec = self._tfidf.transform([text])
            sims = cosine_similarity(q_vec, self._ref_tfidf).ravel()

        top_idx = np.argsort(-sims)[:k]
        results = []
        for i in top_idx:
            row = self.reference_df.iloc[i]
            results.append({
                "request_text": row["request_text"],
                "team": row["final_team"],
                "similarity": round(float(sims[i]), 4),
            })
        return results


def build_and_save_retriever(reference_df: pd.DataFrame, path: Path = MODELS_DIR / "retriever.joblib"):
    retriever = HistoricalRetriever(reference_df)
    joblib.dump(retriever, path)
    return retriever


def load_retriever(path: Path = MODELS_DIR / "retriever.joblib") -> HistoricalRetriever:
    return joblib.load(path)
