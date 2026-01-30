"""
Slightly modified GI-based impostor method (ASGALF-style scoring).

This module provides a drop-in variant of the existing ImpostorDetector that
only changes the per-round scoring aggregation, as described by
Khonji & Iraqi (2014).

The paper’s larger feature set, body richness feature, and score correction offsets remain unimplemented here
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import re

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from statsmodels.stats.proportion import binom_test

from genai_detection.detectors.components.scorer import Scorer
from genai_detection.detectors.components.vector_similarity import minmax_similarity
from genai_detection.detectors.impostor import ImpostorDetector


class ASGALFScorer(Scorer):
    """
    Scorer that replaces the binary win-count with a ratio-based score.

    The ratio follows the "slightly modified" GI-based verification method:
        score += sim(v1, v2)^2 / (sim(v1, x1) * sim(v2, x2))

    where x1 and x2 are the most similar impostors to v1 and v2, respectively.
    """

    def score_pair(self, pair: Dict[str, Any], vectorizer) -> [float, Dict]:
        assert (
            vectorizer.vocabulary_ is not None
        ), "Vectorizer must be fitted before scoring."

        feature_count = len(vectorizer.vocabulary_)
        total_score = 0.0

        # Keep boolean counts for downstream p-value logic (compatibility only).
        left_round_hits = 0
        right_round_hits = 0

        for _ in range(self.rounds):
            # 1) Select random features
            keep_idxs = self._select_random_features(feature_count)

            # 2) Reduce TF-IDF vectors
            left_vec = self._reduce_vector(pair["left"]["tfidf"], keep_idxs)
            right_vec = self._reduce_vector(pair["right"]["tfidf"], keep_idxs)
            left_impostor_vecs = [
                self._reduce_vector(vec, keep_idxs)
                for vec in pair["left"]["impostors_tfidf"]
            ]
            right_impostor_vecs = [
                self._reduce_vector(vec, keep_idxs)
                for vec in pair["right"]["impostors_tfidf"]
            ]

            # 3) Compute similarities
            sim_lr = self.similarity_fn(left_vec, right_vec)
            max_left = max(self.similarity_fn(left_vec, iv) for iv in left_impostor_vecs)
            max_right = max(
                self.similarity_fn(right_vec, iv) for iv in right_impostor_vecs
            )

            # 4) Ratio-based score update (guard against divide-by-zero)
            denom = max_left * max_right
            if denom > 0:
                total_score += (sim_lr * sim_lr) / denom

            # Compatibility counters for the existing binomial significance logic
            left_round_hits += sim_lr > max_left
            right_round_hits += sim_lr > max_right

        p_values = {
            "left_disputed_right_candidate_uncorrected_p_value": binom_test(
                count=int(left_round_hits),
                nobs=self.rounds,
                prop=1 / (1 + len(pair["left"]["impostors_tfidf"])),
                alternative="larger",
            ),
            "right_disputed_left_candidate_uncorrected_p_value": binom_test(
                count=int(right_round_hits),
                nobs=self.rounds,
                prop=1 / (1 + len(pair["right"]["impostors_tfidf"])),
                alternative="larger",
            ),
        }

        return total_score, p_values


class ASGALFFeatureExtractor:
    """
    Feature extractor for ASGALF (Khonji & Iraqi, 2014).

    Features:
      - Letter, word, function word, word shape, POS tag, and POS-word n-grams (n in 1..10).
      - Body richness = unique words / total words.

    Frequency filtering:
      - Keep features that appear at least `min_token_count` times in any single document.
    """

    def __init__(
        self,
        ngram_max: int = 10,
        min_token_count: int = 5,
        function_words: Iterable[str] | None = None,
    ):
        self.ngram_max = ngram_max
        self.min_token_count = min_token_count
        self.function_words = (
            {w.lower() for w in function_words}
            if function_words is not None
            else set(ENGLISH_STOP_WORDS)
        )
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self.vectorizer = SimpleNamespace(vocabulary_={})

    @staticmethod
    def _letters_only(text: str) -> str:
        return re.sub(r"[^A-Za-z]", "", text)

    @staticmethod
    def _word_tokens(text: str) -> List[str]:
        # Wordpunct tokenization without external resources.
        return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)

    @staticmethod
    def _word_shapes(tokens: List[str]) -> List[str]:
        shapes = []
        for token in tokens:
            shape = []
            for ch in token:
                if ch.isupper():
                    shape.append("C")
                elif ch.islower():
                    shape.append("c")
                elif ch.isdigit():
                    shape.append("N")
                else:
                    shape.append("x")
            shapes.append("".join(shape))
        return shapes

    @staticmethod
    def _pos_tags(tokens: List[str]) -> List[str]:
        try:
            import nltk
            from nltk import pos_tag
        except Exception as exc:
            raise RuntimeError(
                "NLTK is required for POS features. Install nltk and its tagger data "
                "(averaged_perceptron_tagger)."
            ) from exc
        try:
            return [tag for _word, tag in pos_tag(tokens)]
        except LookupError as exc:
            raise RuntimeError(
                "Missing NLTK tagger data: download 'averaged_perceptron_tagger'."
            ) from exc

    @staticmethod
    def _ngrams(tokens: List[str], n: int) -> List[str]:
        if n <= 0 or len(tokens) < n:
            return []
        return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _analyze(self, text: str) -> List[str]:
        tokens = self._word_tokens(text)

        # Letter n-grams
        letters = self._letters_only(text)
        features: List[str] = []
        for n in range(1, self.ngram_max + 1):
            for i in range(0, max(0, len(letters) - n + 1)):
                features.append(f"char:{letters[i:i+n]}")

        # Word n-grams
        for n in range(1, self.ngram_max + 1):
            for gram in self._ngrams(tokens, n):
                features.append(f"word:{gram}")

        # Function word n-grams
        func_tokens = [t.lower() for t in tokens if t.lower() in self.function_words]
        for n in range(1, self.ngram_max + 1):
            for gram in self._ngrams(func_tokens, n):
                features.append(f"func:{gram}")

        # Word shape n-grams
        shapes = self._word_shapes(tokens)
        for n in range(1, self.ngram_max + 1):
            for gram in self._ngrams(shapes, n):
                features.append(f"shape:{gram}")

        # POS tag and POS-word n-grams
        pos_tags = self._pos_tags(tokens)
        for n in range(1, self.ngram_max + 1):
            for gram in self._ngrams(pos_tags, n):
                features.append(f"pos:{gram}")

        pos_words = [f"{w}-{t}" for w, t in zip(tokens, pos_tags)]
        for n in range(1, self.ngram_max + 1):
            for gram in self._ngrams(pos_words, n):
                features.append(f"posword:{gram}")

        return features

    def _build_vocabulary(self, corpus: List[str]) -> Dict[str, int]:
        max_counts: Dict[str, int] = {}
        for doc in corpus:
            counts: Dict[str, int] = {}
            for token in self._analyze(doc):
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                prev = max_counts.get(token, 0)
                if count > prev:
                    max_counts[token] = count

        tokens = [t for t, c in max_counts.items() if c >= self.min_token_count]
        tokens.sort()
        return {token: idx for idx, token in enumerate(tokens)}

    def _body_richness(self, corpus: List[str]) -> np.ndarray:
        values = []
        for doc in corpus:
            tokens = self._word_tokens(doc)
            if not tokens:
                values.append(0.0)
            else:
                values.append(len(set(tokens)) / float(len(tokens)))
        return np.asarray(values, dtype=float)

    def fit_transform(self, corpus: List[str]) -> sp.csr_matrix:
        if not corpus:
            raise ValueError("The corpus cannot be empty.")

        vocabulary = self._build_vocabulary(corpus)
        self._tfidf_vectorizer = TfidfVectorizer(
            analyzer=self._analyze,
            vocabulary=vocabulary,
        )
        tfidf = self._tfidf_vectorizer.fit_transform(corpus)

        body_richness = self._body_richness(corpus)
        body_richness_col = sp.csr_matrix(body_richness.reshape(-1, 1))
        combined = sp.hstack([tfidf, body_richness_col], format="csr")

        # Extend vocabulary with body richness feature.
        extended_vocab = dict(vocabulary)
        extended_vocab["body_richness"] = len(vocabulary)
        self.vectorizer.vocabulary_ = extended_vocab

        return combined


class ASGALFImpostorDetector(ImpostorDetector):
    """
    Impostor detector variant using ASGALF-style score aggregation.

    This class keeps the full impostor pipeline intact and only swaps the scorer
    to reflect the ratio-based aggregation in Khonji & Iraqi (2014).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap only the scoring logic and feature inventory; keep the rest unchanged.
        self.feature_extractor = ASGALFFeatureExtractor(
            ngram_max=10,
            min_token_count=5,
        )
        self.scorer = ASGALFScorer(
            rounds=self.rounds,
            portion_delete=self.portion_delete,
            similarity_fn=minmax_similarity,
        )


__all__ = ["ASGALFImpostorDetector"]
