"""Feature construction for the routing classifier.

Two representations are used across the model-comparison experiments
(see src/train.py):

1. Word-level TF-IDF on the cleaned request text (baseline).
2. Character-level TF-IDF (3-5 grams) on the cleaned request text, which is
   robust to spelling mistakes, abbreviations and the legacy-encoding noise
   documented in src/data.py.
3. Character TF-IDF + one-hot structured features (product_family,
   warranty_status, channel, source) -- the final chosen model.

`team_label` and any resolution-log field are never included as features.
All vectorizers/encoders are fit on the training split only and re-used
(never re-fit) on validation/test/holdout data, to avoid leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

STRUCTURED_COLS = ["product_family", "warranty_status", "channel", "source"]
TEXT_COL = "request_text_clean"


@dataclass
class FeatureBundle:
    """Holds fitted transformers so train/eval/predict all share one code path."""

    word_vec: TfidfVectorizer = None
    char_vec: TfidfVectorizer = None
    ohe: OneHotEncoder = None
    variant: str = "char_struct"  # 'word', 'char', 'char_struct'

    def fit(self, df, variant: str = "char_struct"):
        self.variant = variant
        text = df[TEXT_COL].fillna("")
        if variant == "word":
            self.word_vec = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True
            )
            self.word_vec.fit(text)
        else:
            self.char_vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60000, sublinear_tf=True
            )
            self.char_vec.fit(text)
        if variant == "char_struct":
            self.ohe = OneHotEncoder(handle_unknown="ignore")
            self.ohe.fit(df[STRUCTURED_COLS].fillna("missing"))
        return self

    def transform(self, df):
        text = df[TEXT_COL].fillna("")
        if self.variant == "word":
            X = self.word_vec.transform(text)
        else:
            X = self.char_vec.transform(text)
        if self.variant == "char_struct":
            X_struct = self.ohe.transform(df[STRUCTURED_COLS].fillna("missing"))
            X = sp.hstack([X, X_struct], format="csr")
        return X

    def fit_transform(self, df, variant: str = "char_struct"):
        self.fit(df, variant)
        return self.transform(df)
