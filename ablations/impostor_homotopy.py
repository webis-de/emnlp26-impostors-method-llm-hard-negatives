"""
Homotopy-based classification (HBC) impostor variant.

This module adapts the ImpostorDetector to the Homotopy-based Classification
scheme described by Gutierrez et al. (2015). The only differences from the
original impostor method are:
  - Document representation (bag of words frequencies, word bigrams frequencies,
  punctuation counts for non-English texts (hence: here ignored due to only English texts), and character trigram
  frequencies (authors specify letters, hence no numbers; they say 'up to three', hence one, two and three)).
  - Scoring via sparse reconstruction (homotopy-style L1) instead of similarity-based "wins".
  - No role swap
  - Here, we allow any impostor generation, but in the original paper they only decribe search-based (or:
  "on-the-fly") impostor generation
What is Sparse Representation classification?
1. Select random impostors and texts from the candidate author (one or more, here: one)
2. Selected texts are represented in Vector Space Model using features listed above
3. Selected texts (text = column) combined are matrix A
4. Use L_1 homotopy algorithm (sparse coding = forces only few non-zero entries) to find optimal x' s.t. y = Ax' (almost)
5. Compute residuals r_i(y) = |y- Ax'_i| for each class i (impostors vs. candidate texts) where x'_i masks all
non-class members with 0
6. If candidate class = class i with minimal residual: This experiment suggest same authorship
7. Repeat with different sample of impostors and candidate texts
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import re

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LassoLars
# import spams
from sklearn.preprocessing import normalize
from statsmodels.stats.proportion import binom_test

from genai_detection.detectors.components.scorer import ScoreResult
from genai_detection.detectors.impostor import ImpostorDetector


@dataclass
class HBCFeatureConfig:
    """Configuration for homotopy-based document features."""

    include_words_unigrams: bool = True
    include_word_bigrams: bool = True
    include_punctuation: bool = False # Gutierrez et al. (2015) only include for non-English texts
    include_char_trigrams: bool = True
    min_df: int = 1
    max_features: int | None = None
    normalize_rows: bool = True


class HBCFeatureExtractor:
    """
    Feature extractor for Homotopy-based Classification (HBC).

    Features are taken from the paper's vector space representation:
      - Bag of words (counts)
      - Word bigrams (counts)
      - Punctuation counts
      - Letter trigrams (counts)
    """

    def __init__(self, config: HBCFeatureConfig):
        self.config = config
        self._vectorizers: List[tuple[str, CountVectorizer]] = []
        self.vectorizer = SimpleNamespace(vocabulary_={})

    @staticmethod
    def _word_tokens(text: str) -> List[str]:
        # Keep simple word tokens with optional apostrophes.
        return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())

    @classmethod
    def _word_and_bigram_tokens(cls, text: str) -> List[str]:
        tokens = cls._word_tokens(text)
        if not tokens:
            return []
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    @staticmethod
    def _punct_tokens(text: str) -> List[str]:
        return re.findall(r"[!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]", text)

    @staticmethod
    def _char_trigram_tokens(text: str) -> List[str]:
        """
        Extract sliding character trigrams over the sequence of letters only.

        Implementation rationale (strict replication):
        Following Gutierrez et al. (PAN 2015), character features are defined as
        "three consecutive letters". We therefore:
            - lowercase the text,
            - remove all non-letter characters,
            - compute trigrams over the resulting continuous letter stream.

        Importantly, we do NOT preserve whitespace or enforce word boundaries.
        This allows trigrams to cross original word boundaries, which is
        consistent with the standard sliding-window letter n-gram extraction
        commonly used in PAN-style authorship systems of that period.

        No boundary markers are inserted and no per-word splitting is performed.
        """
        letters = re.sub(r"[^A-Za-z]", "", text.lower())
        if len(letters) < 3:
            return []
        return [letters[i : i + 3] for i in range(len(letters) - 2)]

    def _build_vectorizers(self) -> None:
        self._vectorizers = []
        cfg = self.config

        if cfg.include_words_unigrams or cfg.include_word_bigrams:
            analyzer = (
                self._word_and_bigram_tokens
                if cfg.include_word_bigrams
                else self._word_tokens
            )
            self._vectorizers.append(
                (
                    "word:",
                    CountVectorizer(
                        analyzer=analyzer,
                        min_df=cfg.min_df,
                        max_features=cfg.max_features,
                    ),
                )
            )

        if cfg.include_punctuation:
            self._vectorizers.append(
                (
                    "punc:",
                    CountVectorizer(
                        analyzer=self._punct_tokens,
                        min_df=cfg.min_df,
                        max_features=cfg.max_features,
                    ),
                )
            )

        if cfg.include_char_trigrams:
            self._vectorizers.append(
                (
                    "char3:",
                    CountVectorizer(
                        analyzer=self._char_trigram_tokens,
                        min_df=cfg.min_df,
                        max_features=cfg.max_features,
                    ),
                )
            )

    def fit_transform(self, corpus: List[str]) -> sp.csr_matrix:
        if not corpus:
            raise ValueError("The corpus cannot be empty.")

        self._build_vectorizers()

        matrices = []
        combined_vocab: Dict[str, int] = {}
        offset = 0

        for prefix, vectorizer in self._vectorizers:
            mat = vectorizer.fit_transform(corpus)
            matrices.append(mat)

            # Build a combined vocabulary for compatibility with the base API.
            for token, idx in vectorizer.vocabulary_.items():
                combined_vocab[f"{prefix}{token}"] = offset + idx
            offset += len(vectorizer.vocabulary_)

        if not matrices:
            raise ValueError("No features were configured for HBCFeatureExtractor.")

        combined = sp.hstack(matrices, format="csr")
        if self.config.normalize_rows:
            combined = normalize(combined, norm="l2", axis=1, copy=False)

        self.vectorizer.vocabulary_ = combined_vocab
        return combined


class HBCScorer:
    """
    Score pairs using homotopy-style sparse reconstruction (HBC).

    Each round samples a random subset of impostors, reconstructs the disputed
    document using L1-regularized regression, and votes for the identity with
    the smallest reconstruction residual.
    """

    def __init__(
        self,
        rounds: int,
        impostor_keep_ratio: float = 0.5,
        alpha: float = 0.001,
        max_iter: int = 500,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        self.rounds = rounds
        self.impostor_keep_ratio = impostor_keep_ratio
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def _sample_impostors(self, impostors: List[List[float]]) -> List[List[float]]:
        if not impostors:
            return []
        keep = max(1, int(len(impostors) * self.impostor_keep_ratio))
        if keep >= len(impostors):
            return impostors
        idx = self._rng.choice(len(impostors), size=keep, replace=False)
        return [impostors[i] for i in idx]

    @staticmethod
    def _residual(y: np.ndarray, A: np.ndarray, coeffs: np.ndarray, idx: List[int]) -> float:
        if not idx:
            return float("inf")
        masked = np.zeros_like(coeffs)
        masked[idx] = coeffs[idx]
        recon = A @ masked
        return float(np.linalg.norm(y - recon))

    def score_pair(self, pair: Dict[str, Any], vectorizer) -> ScoreResult:
        total_score = 0.0
        disputed, candidate = "left", "right"
        disputed_vec = np.asarray(pair[disputed]["tfidf"], dtype=float)
        # should also be sampled but we only have one candidate text at the time
        candidate_vecs = [pair[candidate]["tfidf"]]

        for _ in range(self.rounds):
            impostors = self._sample_impostors(
                pair[candidate]["impostors_tfidf"]
            )

            if not impostors:
                continue

            # Build dictionary A with candidate docs first, then impostors.
            cols = candidate_vecs + impostors
            A = np.stack(cols, axis=1)

            # Solve y ~ A x with L1 regularization (homotopy-style).
            model = LassoLars(
                alpha=self.alpha,
                fit_intercept=False,
                max_iter=self.max_iter,
                eps=self.tol,
            )
            model.fit(A, disputed_vec)
            coeffs = model.coef_
            # coeffs = spams.lasso(A, disputed_vec, lambda1=0.1)

            # Residuals: candidate identity (all candidate columns) vs each impostor.
            # usually: only one candidate text, but generally more are possible
            candidate_idx = list(range(len(candidate_vecs)))
            r_candidate = self._residual(disputed_vec, A, coeffs, candidate_idx)
            r_impostors = [
                self._residual(
                    disputed_vec,
                    A,
                    coeffs,
                    [len(candidate_vecs) + i],
                )
                for i in range(len(impostors))
            ]

            if r_impostors and r_candidate <= min(r_impostors):
                total_score += 1

        return ScoreResult(score=total_score/self.rounds, p_values={})


class HBCImpostorDetector(ImpostorDetector):
    """
    Impostor detector variant using HBC-style sparse reconstruction scoring.

    This class keeps the full impostor pipeline intact and only swaps the
    feature extractor and scorer to reflect the homotopy-based method.
    """

    def __init__(
        self,
        *args,
        n_impostors: int  = 50,
        # homotopy_alpha: float = 0.001,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, n_impostors=n_impostors)
        impostor_keep_ratio = n_impostors / self.impostor_generator.num_potential_impostors

        feature_config = HBCFeatureConfig(
            include_words_unigrams=True,
            include_word_bigrams=True,
            include_punctuation=True,
            include_char_trigrams=True,
            min_df=1,
            max_features=self.top_n,
            normalize_rows=True,
        )
        self.feature_extractor = HBCFeatureExtractor(config=feature_config)
        self.scorer = HBCScorer(
            rounds=self.rounds,
            impostor_keep_ratio=impostor_keep_ratio,
            alpha=0.001,
        )
        # save outputs to extra ablation output collection
        self.impostor_output_collection = self.mongoDB.impostor_ablation_output_collection


__all__ = ["HBCImpostorDetector"]
