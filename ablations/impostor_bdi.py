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

from genai_detection.detectors.components.scorer import ScoreResult, Scorer
from genai_detection.detectors.components.vector_similarity import minmax_similarity
from genai_detection.detectors.impostor import ImpostorDetector


class BDIScorer(Scorer):
    """
    Scorer implementing Bootstrap Distance Imposters.

    For each round:
    1) sample random feature subset,
    2) compute distances from disputed to all candidates and impostors,
    3) select closest impostor and candidate (minimum distance),
    4) compute diff = d(disputed, closest_impostor) - d(disputed, closest_candidate).

    In our scenario, we have only one candidate.

    Final score is the estimated probability mass above zero
    (ties at zero contribute half).
    """

    def __init__(
        self,
        rounds: int = 50,   # Nagy (2024) does not specify beyond "repeat n times"
        portion_delete: float = 0.67,   # cf. Bootstrap Distance Imposters (BDI) ablation by Nagy (2024)
        similarity_fn=minmax_similarity,    # Nagy (2024) reported minmax and cosine scores
    ):
        super().__init__(
            rounds=rounds,
            portion_delete=portion_delete,
            similarity_fn=similarity_fn,
        )

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        # Convert similarity to distance in [0, 1] for minmax/cosine-like metrics.
        return 1.0 - float(self.similarity_fn(a, b))

    @staticmethod
    def _mass_above_zero(diffs: List[float]) -> float:
        if not diffs:
            return 0.5
        arr = np.asarray(diffs, dtype=float)
        gt = np.sum(arr > 0.0)
        eq = np.sum(arr == 0.0)
        return float((gt + 0.5 * eq) / len(arr))

    def score_pair(self, pair: Dict[str, Any], vectorizer) -> ScoreResult:
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

        # Base ImpostorDetector divides returned scores by `self.rounds`.
        # Rescale so the final externally reported score equals BDI probability mass.
        return ScoreResult(score=self._mass_above_zero(diffs) * self.rounds, p_values={})


class BDIImpostorDetector(ImpostorDetector):
    """
    Impostor detector ablation implementing Bootstrap Distance Imposters.

    The detector reuses the existing impostor pipeline and only swaps the
    scoring module.
    """

    def __init__(self, *args, **kwargs):
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
