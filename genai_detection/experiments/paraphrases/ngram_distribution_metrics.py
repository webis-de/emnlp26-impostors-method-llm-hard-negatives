# Copyright 2026 Klara M. Gutekunst, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""N-gram distribution divergence metrics for paraphrase experiments."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


@dataclass(frozen=True)
class NGramDivergenceResult:
    """Divergence scores for one original/paraphrase pair."""

    kl_divergence: float
    js_divergence: float
    original_ngram_count: int
    paraphrase_ngram_count: int
    vocabulary_size: int


class NGramDistributionDivergenceCalculator:
    """Compute KL and Jensen-Shannon divergence over n-gram distributions."""

    def __init__(self, n: int = 4, max_features: int = 100_000, smoothing: float = 1e-12):
        if n < 1:
            raise ValueError("n must be at least 1.")
        if max_features < 1:
            raise ValueError("max_features must be at least 1.")
        if smoothing <= 0:
            raise ValueError("smoothing must be positive.")

        self.n = n
        self.max_features = max_features
        self.smoothing = smoothing

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase word tokens."""
        return TOKEN_PATTERN.findall(str(text).lower())

    def count_ngrams(self, text: str) -> Counter[tuple[str, ...]]:
        """Count n-grams in one text."""
        tokens = self.tokenize(text)
        if len(tokens) < self.n:
            return Counter()
        return Counter(tuple(tokens[i : i + self.n]) for i in range(len(tokens) - self.n + 1))

    def _select_vocabulary(
        self,
        original_counts: Counter[tuple[str, ...]],
        paraphrase_counts: Counter[tuple[str, ...]],
    ) -> list[tuple[str, ...]]:
        """Select the most frequent n-grams from the combined distribution."""
        combined_counts = original_counts + paraphrase_counts
        if len(combined_counts) <= self.max_features:
            return list(combined_counts.keys())
        return [ngram for ngram, _ in combined_counts.most_common(self.max_features)]

    def _probabilities(
        self,
        counts: Counter[tuple[str, ...]],
        vocabulary: Iterable[tuple[str, ...]],
    ) -> np.ndarray:
        """Convert counts into a smoothed probability vector over a fixed vocabulary."""
        values = np.asarray([counts.get(ngram, 0) for ngram in vocabulary], dtype=float)
        values += self.smoothing
        total = values.sum()
        if total <= 0:
            raise ValueError("Cannot normalize an empty probability vector.")
        return values / total

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Compute asymmetric KL divergence D_KL(P || Q)."""
        return float(np.sum(p * np.log(p / q)))

    @classmethod
    def js_divergence(cls, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence with natural logarithms."""
        m = 0.5 * (p + q)
        return float(0.5 * cls.kl_divergence(p, m) + 0.5 * cls.kl_divergence(q, m))

    def compare(self, original_text: str, paraphrase_text: str) -> NGramDivergenceResult:
        """Compare one original text with one paraphrase."""
        original_counts = self.count_ngrams(original_text)
        paraphrase_counts = self.count_ngrams(paraphrase_text)

        if not original_counts or not paraphrase_counts:
            logger.warning(
                "Skipping robust n-gram comparison fallback for short text pair: "
                "original_ngrams=%d paraphrase_ngrams=%d",
                sum(original_counts.values()),
                sum(paraphrase_counts.values()),
            )

        vocabulary = self._select_vocabulary(original_counts, paraphrase_counts)
        if not vocabulary:
            return NGramDivergenceResult(
                kl_divergence=np.nan,
                js_divergence=np.nan,
                original_ngram_count=0,
                paraphrase_ngram_count=0,
                vocabulary_size=0,
            )

        p = self._probabilities(original_counts, vocabulary)
        q = self._probabilities(paraphrase_counts, vocabulary)
        return NGramDivergenceResult(
            kl_divergence=self.kl_divergence(p, q),
            js_divergence=self.js_divergence(p, q),
            original_ngram_count=sum(original_counts.values()),
            paraphrase_ngram_count=sum(paraphrase_counts.values()),
            vocabulary_size=len(vocabulary),
        )
