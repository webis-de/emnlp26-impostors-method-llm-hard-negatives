"""
Improved Impostors method (Potha & Stamatatos, 2017).

This module provides a drop-in variant of the existing ImpostorDetector with
two targeted changes:
  1) Impostor selection: keep the most similar impostors to the known document.
  2) Ranking-based scoring: use the rank position of the known document among
     impostors when comparing to the disputed document.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
from statsmodels.stats.proportion import binom_test

from genai_detection.detectors.components.scorer import Scorer
from genai_detection.detectors.components.vector_similarity import minmax_similarity
from genai_detection.detectors.impostor import ImpostorDetector


class Potha2017Scorer(Scorer):
    """
    Scorer implementing the ranking-based aggregation from Potha & Stamatatos (2017).

    For each repetition:
      - select a random subset of features,
      - optionally sample a subset of impostors,
      - compute similarity of impostors to the disputed document,
      - rank candidate similarity among impostors and update the score by 1/pos.
    """

    def __init__(
        self,
        rounds: int,
        portion_delete: float,
        similarity_fn,
        impostors_per_problem: Optional[int] = None,
        impostors_per_round: Optional[int] = None,
    ):
        super().__init__(rounds=rounds, portion_delete=portion_delete, similarity_fn=similarity_fn)
        self.impostors_per_problem = impostors_per_problem
        self.impostors_per_round = impostors_per_round

    def _select_problem_impostors(
        self,
        candidate_vec: List[float],
        impostor_vecs: List[List[float]],
    ) -> List[List[float]]:
        """
        Keep the most similar impostors to the candidate document (min-max similarity).
        """
        if self.impostors_per_problem is None or len(impostor_vecs) <= self.impostors_per_problem:
            return impostor_vecs

        scores = [self.similarity_fn(candidate_vec, vec) for vec in impostor_vecs]
        top_idx = np.argsort(scores)[::-1][: self.impostors_per_problem]
        return [impostor_vecs[i] for i in top_idx]

    def _sample_impostors_for_round(
        self, impostor_vecs: List[List[float]]
    ) -> List[List[float]]:
        """
        Randomly sample impostors for a single repetition (if requested).
        """
        if self.impostors_per_round is None or len(impostor_vecs) <= self.impostors_per_round:
            return impostor_vecs
        return random.sample(impostor_vecs, self.impostors_per_round)

    def score_pair(self, pair: Dict[str, Any], vectorizer) -> [float, Dict]:
        assert (
            vectorizer.vocabulary_ is not None
        ), "Vectorizer must be fitted before scoring."

        feature_count = len(vectorizer.vocabulary_)
        total_score = 0.0
        p_values: Dict[str, float] = {}

        # Iterate over permutations ("left" as disputed, "right" as candidate, and vice versa)
        for j, (disputed, candidate) in enumerate(
            [("left", "right"), ("right", "left")]
        ):
            direction_score = 0.0
            round_hits = 0

            # Select the most similar impostors to the candidate document once per direction.
            problem_impostors = self._select_problem_impostors(
                candidate_vec=pair[candidate]["tfidf"],
                impostor_vecs=pair[candidate]["impostors_tfidf"],
            )

            for _ in range(self.rounds):
                # 1) Select random features
                keep_idxs = self._select_random_features(feature_count)

                # 2) Reduce TF-IDF vectors
                disputed_vec = self._reduce_vector(pair[disputed]["tfidf"], keep_idxs)
                candidate_vec = self._reduce_vector(pair[candidate]["tfidf"], keep_idxs)
                impostor_vecs = [
                    self._reduce_vector(vec, keep_idxs) for vec in problem_impostors
                ]

                # 3) Optionally sample impostors for this round
                round_impostors = self._sample_impostors_for_round(impostor_vecs)

                # 4) Compute similarities (impostor -> disputed only)
                impostor_scores = [
                    self.similarity_fn(disputed_vec, iv) for iv in round_impostors
                ]
                sim_known = self.similarity_fn(disputed_vec, candidate_vec)

                # 5) Rank candidate similarity among impostors (descending)
                pos = 1 + sum(score > sim_known for score in impostor_scores)
                direction_score += 1.0 / (self.rounds * pos)

                # Compatibility counter for downstream binomial p-values
                if impostor_scores:
                    round_hits += sim_known > max(impostor_scores)

            p_feat_name = f"{disputed}_disputed_{candidate}_candidate_uncorrected_p_value"
            effective_impostors = len(problem_impostors)
            if self.impostors_per_round is not None:
                effective_impostors = min(effective_impostors, self.impostors_per_round)
            p_values[p_feat_name] = binom_test(
                count=int(round_hits),
                nobs=self.rounds,
                prop=1 / (1 + effective_impostors),
                alternative="larger",
            )

            total_score += direction_score
            total_score /= j + 1

        return total_score, p_values


class Potha2017ImpostorDetector(ImpostorDetector):
    """
    Impostor detector variant matching Potha & Stamatatos (2017).

    This class keeps the full impostor pipeline intact and only swaps the scoring
    logic to incorporate (1) impostor selection and (2) rank-based aggregation.
    """

    def __init__(
        self,
        *args,
        impostors_per_problem: Optional[int] = None,
        impostors_per_round: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Default to selecting up to the configured number of impostors per problem.
        if impostors_per_problem is None:
            impostors_per_problem = self.n_impostors

        self.scorer = Potha2017Scorer(
            rounds=self.rounds,
            portion_delete=self.portion_delete,
            similarity_fn=minmax_similarity,
            impostors_per_problem=impostors_per_problem,
            impostors_per_round=impostors_per_round,
        )
        # save outputs to extra ablation output collection
        self.impostor_output_collection = self.mongoDB.impostor_ablation_output_collection


__all__ = ["Potha2017ImpostorDetector"]
