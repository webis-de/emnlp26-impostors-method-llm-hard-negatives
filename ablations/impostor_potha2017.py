"""
Improved Impostors method (Potha & Stamatatos, 2017).

This module provides a drop-in variant of the existing ImpostorDetector with
two targeted changes:
  1) Impostor selection: keep the most similar impostors to the known document in terms of min-max similarity.
  2) Only consider candidate + impostor - disputed document (omit role swap)
  3) Ranking-based scoring: use the rank position of the known document among
     impostors when comparing to the disputed document instead of only considering the first rank.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np

from genai_detection.detectors.components.scorer import Scorer
from genai_detection.detectors.components.vector_similarity import minmax_similarity
from genai_detection.detectors.impostor import ImpostorDetector

logger = logging.getLogger(__name__)

class Potha2017Scorer(Scorer):
    """
    Scorer implementing the ranking-based aggregation from Potha & Stamatatos (2017).

    - compute similarity of impostors to the disputed document & keep most similar impostors.
    For each repetition:
      - sample random subset of most similar impostors (cardinality stays the same),
      - select a random subset of features (cardinality stays the same),
      - rank candidate similarity among impostors and update the score by 1/(position * repetition).
    Only consider one direction (i.e., generate impostors of the candidate document) and do not do role swap.
    """

    def __init__(
        self,
        rounds: int=10,
        portion_delete: float=0.5,
        similarity_fn=minmax_similarity,
        impostors_per_problem: Optional[int] = 50,
        impostors_per_round: Optional[int] = 5,
    ):
        """
        :param rounds: Number of repetitions. Paper defaults to impostors_per_problem/5.
        :param portion_delete: Proportion of impostors to be deleted. Paper defaults to 0.5.
        :param similarity_fn: Similarity function. Paper defaults to minmax_similarity.
        :param impostors_per_problem: Number of impostors per problem. Paper does not specify a default value.
        :param impostors_per_round: Number of impostors per round. Paper defaults to impostors_per_problem/10.
        """
        super().__init__(rounds=rounds, portion_delete=portion_delete, similarity_fn=similarity_fn)
        assert impostors_per_problem >= impostors_per_round, (f"# Impostors selected due to max similarity to "
                                                             f"candidate text ({impostors_per_problem}) "
                                                             f"needs to be greater than number chosen"
                                                             f" {impostors_per_round}")
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
        if len(impostor_vecs) <= self.impostors_per_problem:
            return impostor_vecs

        scores = [self.similarity_fn(candidate_vec, vec) for vec in impostor_vecs]
        top_idx = np.argsort(scores)[::-1][: self.impostors_per_problem]
        return [impostor_vecs[i] for i in top_idx]

    def _sample_impostors_for_round(
        self, impostor_vecs: List[List[float]]
    ) -> List[List[float]]:
        """
        Randomly sample impostors for a single repetition (if requested).
        According to the original paper, the number of impostors should remain the same.
        """
        if len(impostor_vecs) <= self.impostors_per_round:
            return impostor_vecs
        return random.sample(impostor_vecs, self.impostors_per_round)

    def score_pair(self, pair: Dict[str, Any], vectorizer) -> [float, Dict]:
        assert (
            vectorizer.vocabulary_ is not None
        ), "Vectorizer must be fitted before scoring."

        feature_count = len(vectorizer.vocabulary_)
        total_score = 0.0

        # Only consider "left" as disputed, "right" as candidate
        disputed, candidate = "left", "right"

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
            logger.info(f"Rounds {self.rounds} impostor scored {pos} impostors ")
            total_score += 1.0 / (self.rounds * pos)

        return total_score, {}


class Potha2017ImpostorDetector(ImpostorDetector):
    """
    Impostor detector variant matching Potha & Stamatatos (2017).

    This class keeps the full impostor pipeline intact and only swaps the scoring
    logic to incorporate (1) impostor selection, (2) rank-based aggregation and (3) one-way comparison.
    """

    def __init__(
        self,
        *args,
        impostors_per_problem: Optional[int] = None,
        impostors_per_round: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        logger.info(f"Initializing Potha2017ImpostorDetector with args: {args}")

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
