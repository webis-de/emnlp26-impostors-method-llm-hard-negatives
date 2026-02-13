"""
Bootstrap Distance Imposters (BDI) ablation by Nagy (2024).

This variant keeps the existing impostor pipeline and only replaces scoring
with the BDI-style bootstrap distance differences described by Nagy (2024):
- per round:
    - sample features
    - compute distance difference between each impostor and disputed text,
        as well as each candidate (here: one) and disputed text.
    - select closest impostor and closest candidate (here: only one) to disputed text
    - compute d(disputed, impostor) - d(disputed, candidate) (= one point in distribution)
- summarize the resulting distribution by probability mass above zero.

Find other implementations:
- https://github.com/bnagy/ruzicka/blob/main/ruzicka/BDIVerifier.py (13.02.2026)
"""

from __future__ import annotations

import typing as t
from typing import Any, Dict, List

import numpy as np
import scipy as sp

from genai_detection.detectors.components.scorer import ScoreResult, Scorer
from genai_detection.detectors.components.vector_similarity import minmax_similarity
from genai_detection.detectors.impostor import ImpostorDetector


class BDIScorer(Scorer):
    """
    Scorer implementing Bootstrap Distance Imposters (BDI).

    For each round:
    1) sample random feature subset,
    2) compute distances from disputed to all candidates and impostors,
    3) select closest impostor and candidate (minimum distance),
    4) compute diff = d(disputed, closest_impostor) - d(disputed, closest_candidate).

    In this repository's pairwise setup, there is exactly one candidate vector.

    The probability estimate follows Ruzicka's BDI implementation:
    ``proba = (100 - percentileofscore(diffs, 0)) / 100``.
    """

    def __init__(
        self,
        rounds: int = 100,   # Nagy (2024) does not specify beyond "repeat n times", but BDI implementation has 100 nb_bootstrap_iter
        portion_delete: float = 0.67,   # cf. Bootstrap Distance Imposters (BDI) ablation by Nagy (2024)
        similarity_fn=minmax_similarity,    # Nagy (2024) reported minmax and cosine scores
    ):
        super().__init__(
            rounds=rounds,
            portion_delete=portion_delete,
            similarity_fn=similarity_fn,
        )

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Convert similarity output to distance-like form in [0, 1]."""
        return 1.0 - float(self.similarity_fn(a, b))

    @staticmethod
    def _mass_above_zero(diffs: List[float]) -> float:
        """
        Map bootstrap distance differences to a probability-like score.

        This mirrors ``BDIVerifier.predict_proba``:
        ``(100 - scipy.stats.percentileofscore(diffs, 0)) / 100``.

        Args:
            diffs: Bootstrap differences where each element is
                ``d(disputed, impostor) - d(disputed, candidate)``.

        Returns:
            Score in [0, 1]. If no valid differences exist, returns 0.5.
        """

        if not diffs:
            return 0.5
        return float((100 - sp.stats.percentileofscore(diffs, 0)) / 100.0)

    def score_pair(self, pair: Dict[str, Any], vectorizer) -> ScoreResult:
        """
        Score one pair via BDI bootstrapped distance differences.

        Args:
            pair: Pair dictionary containing left/right TF-IDF vectors and
                candidate-side impostor vectors.
            vectorizer: Fitted vectorizer-like object exposing ``vocabulary_``.

        Returns:
            ``ScoreResult`` with BDI score and an empty p-values dict.
        """
        assert (
            vectorizer.vocabulary_ is not None
        ), "Vectorizer must be fitted before scoring."

        feature_count = len(vectorizer.vocabulary_)
        disputed, candidate = "left", "right"
        diffs: List[float] = []

        for _ in range(self.rounds):
            keep_idxs = self._select_random_features(feature_count)
            disputed_vec = self._reduce_vector(pair[disputed]["tfidf"], keep_idxs)
            candidate_vec = self._reduce_vector(pair[candidate]["tfidf"], keep_idxs)

            impostor_vecs = [
                self._reduce_vector(vec, keep_idxs)
                for vec in pair[candidate]["impostors_tfidf"]
            ]
            if not impostor_vecs:
                continue

            closest_impostor_distance = min(
                self._distance(disputed_vec, impostor_vec)
                for impostor_vec in impostor_vecs
            )
            diff = closest_impostor_distance - self._distance(
                disputed_vec, candidate_vec
            )
            diffs.append(float(diff))

        return ScoreResult(score=self._mass_above_zero(diffs), p_values={})


class BDIImpostorDetector(ImpostorDetector):
    """
    Impostor detector ablation implementing Bootstrap Distance Imposters.

    The detector reuses the existing impostor pipeline and only swaps the
    scoring module.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize BDI by replacing only the scoring component.

        All preprocessing, impostor generation, feature extraction, caching,
        and persistence remain inherited from ``ImpostorDetector``.
        """
        super().__init__(*args, **kwargs)
        self.scorer = BDIScorer(
            rounds=self.rounds,
            portion_delete=self.portion_delete,
            similarity_fn=minmax_similarity,
        )
        self.impostor_output_collection = self.mongoDB.impostor_ablation_output_collection

    def _build_impostor_pair_datastructure(
        self, pair: t.Dict[str, str], keys: List[str] = ["right"]
    ) -> tuple[bool, t.Dict[str, str]]:
        """
        Build impostors only for the candidate side ("right"), matching BDI's
        one-way verification setup.
        """
        return super()._build_impostor_pair_datastructure(pair=pair, keys=keys)


__all__ = ["BDIImpostorDetector"]
