"""Runnable TF-IDF, BM25, and LSI baselines.

These are the classical information-retrieval baselines used for the
Stage 2 baseline comparison and as input rankings for the Proposed
Framework's ``hybrid``/``hybrid3`` retrievers (see
``scripts/run_llm_baselines.py``). Each baseline exposes a common
``fit(code_ids, code_texts)`` / ``score_all(requirement_texts)`` interface
returning a dense requirement-by-code similarity matrix; ``create_baseline``
builds one from the ``ir`` section of ``configs/datasets.yaml``. All scoring
here is local (scikit-learn / hand-rolled BM25); no network or LLM calls are
made in this module.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils import tokenize


class IRBaseline:
    name = "base"

    def fit(self, code_ids: list[str], code_texts: list[str]) -> "IRBaseline":
        raise NotImplementedError

    def score_all(self, requirement_texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class TFIDFBaseline(IRBaseline):
    name = "tfidf"

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 1.0,
    ):
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
        )
        self.code_ids: list[str] = []
        self.code_matrix = None

    def fit(self, code_ids: list[str], code_texts: list[str]) -> "TFIDFBaseline":
        self.code_ids = code_ids
        self.code_matrix = self.vectorizer.fit_transform(code_texts)
        return self

    def score_all(self, requirement_texts: list[str]) -> np.ndarray:
        if self.code_matrix is None:
            raise RuntimeError("Call fit() before score_all().")
        requirement_matrix = self.vectorizer.transform(requirement_texts)
        return cosine_similarity(requirement_matrix, self.code_matrix)


class BM25Baseline(IRBaseline):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.code_ids: list[str] = []
        self.documents: list[Counter[str]] = []
        self.document_lengths: np.ndarray | None = None
        self.average_length = 0.0
        self.idf: dict[str, float] = {}

    def fit(self, code_ids: list[str], code_texts: list[str]) -> "BM25Baseline":
        self.code_ids = code_ids
        tokens = [tokenize(text) for text in code_texts]
        self.documents = [Counter(document) for document in tokens]
        self.document_lengths = np.array([len(document) for document in tokens])
        self.average_length = float(self.document_lengths.mean())
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(document.keys())
        count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        return self

    def score_all(self, requirement_texts: list[str]) -> np.ndarray:
        if self.document_lengths is None:
            raise RuntimeError("Call fit() before score_all().")
        scores = np.zeros((len(requirement_texts), len(self.documents)), dtype=float)
        for query_index, text in enumerate(requirement_texts):
            query_terms = set(tokenize(text))
            for document_index, frequencies in enumerate(self.documents):
                length_ratio = (
                    self.document_lengths[document_index] / self.average_length
                    if self.average_length
                    else 0.0
                )
                score = 0.0
                for term in query_terms:
                    frequency = frequencies.get(term, 0)
                    if not frequency:
                        continue
                    denominator = frequency + self.k1 * (
                        1.0 - self.b + self.b * length_ratio
                    )
                    score += self.idf.get(term, 0.0) * (
                        frequency * (self.k1 + 1.0) / denominator
                    )
                scores[query_index, document_index] = score
        return scores


class LSIBaseline(IRBaseline):
    name = "lsi"

    def __init__(
        self,
        num_topics: int = 100,
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
    ):
        self.num_topics = num_topics
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
        self.svd: TruncatedSVD | None = None
        self.code_ids: list[str] = []
        self.code_lsi: np.ndarray | None = None
        self.code_matrix = None

    def fit(self, code_ids: list[str], code_texts: list[str]) -> "LSIBaseline":
        self.code_ids = code_ids
        self.code_matrix = self.vectorizer.fit_transform(code_texts)
        max_components = min(
            self.num_topics,
            self.code_matrix.shape[0] - 1,
            self.code_matrix.shape[1] - 1,
        )
        if max_components < 1:
            self.svd = None
            return self
        self.svd = TruncatedSVD(n_components=max_components, random_state=42)
        self.code_lsi = self.svd.fit_transform(self.code_matrix)
        return self

    def score_all(self, requirement_texts: list[str]) -> np.ndarray:
        if self.code_matrix is None:
            raise RuntimeError("Call fit() before score_all().")
        requirement_matrix = self.vectorizer.transform(requirement_texts)
        if self.svd is None or self.code_lsi is None:
            return cosine_similarity(requirement_matrix, self.code_matrix)
        requirement_lsi = self.svd.transform(requirement_matrix)
        return cosine_similarity(requirement_lsi, self.code_lsi)


def create_baseline(method: str, config: dict) -> IRBaseline:
    method = method.casefold()
    ir_config = config["ir"]
    if method == "tfidf":
        item = ir_config["tfidf"]
        return TFIDFBaseline(
            max_features=int(item["max_features"]),
            ngram_range=tuple(item["ngram_range"]),
            min_df=int(item["min_df"]),
            max_df=float(item["max_df"]),
        )
    if method == "bm25":
        item = ir_config["bm25"]
        return BM25Baseline(k1=float(item["k1"]), b=float(item["b"]))
    if method == "lsi":
        item = ir_config["lsi"]
        return LSIBaseline(
            num_topics=int(item["num_topics"]),
            max_features=int(item["max_features"]),
            ngram_range=tuple(item["ngram_range"]),
        )
    raise ValueError(f"Unknown IR method: {method}")
