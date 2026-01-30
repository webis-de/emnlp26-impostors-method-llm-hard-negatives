"""
Slightly modified GI-based impostor method (ASGALF-style scoring).

This module provides a drop-in variant of the existing ImpostorDetector that
only changes the per-round scoring aggregation, as described by
Khonji & Iraqi (2014).

The paper’s larger feature set, body richness feature, and score correction offsets remain unimplemented here
"""
from __future__ import annotations

from typing import Any, Dict

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


class ASGALFImpostorDetector(ImpostorDetector):
    """
    Impostor detector variant using ASGALF-style score aggregation.

    This class keeps the full impostor pipeline intact and only swaps the scorer
    to reflect the ratio-based aggregation in Khonji & Iraqi (2014).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap only the scoring logic; keep the rest of the pipeline unchanged.
        self.scorer = ASGALFScorer(
            rounds=self.rounds,
            portion_delete=self.portion_delete,
            similarity_fn=minmax_similarity,
        )


__all__ = ["ASGALFImpostorDetector"]
