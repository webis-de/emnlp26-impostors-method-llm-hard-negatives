import collections
import itertools
from random import sample
from typing import Dict, Any, List

import numpy as np
from statsmodels.stats.proportion import binom_test


class Scorer:
    """
    Performs repeated random-feature deletion scoring (Impostor Method scoring).
    """

    def __init__(self, rounds: int, portion_delete: float, similarity_fn):
        """
        :param rounds: number of random-deletion scoring passes
        :param portion_delete: Portion of features to delete (0–1)
        :param similarity_fn: A function(a: array, b: array) -> float
        """
        self.rounds = rounds
        self.portion_delete = portion_delete
        self.similarity_fn = similarity_fn

    def _select_random_features(self, total_features: int) -> List[int]:
        """Return a random subset of feature indices to keep."""
        keep = int(total_features * (1 - self.portion_delete))
        return sample(range(total_features), keep)

    def _reduce_vector(self, v: List[float], idx: List[int]) -> np.ndarray:
        """Reduce vector to the selected feature indices."""
        return np.asarray(v)[idx]

    def score_pair(self, pair: Dict[str, Any], vectorizer) -> [float, Dict]:
        """
        Compute a score between left/right texts using the impostor method.

        `pair` structure:
        {
            "left":  { "tfidf": [...], "impostors_tfidf": [...] },
            "right": { "tfidf": [...], "impostors_tfidf": [...] },
            ...
        }
        """
        assert (
            vectorizer.vocabulary_ is not None
        ), "Vectorizer must be fitted before scoring."

        feature_count = len(vectorizer.vocabulary_)
        total_score = 0.0
        p_values = collections.defaultdict(float)

        # iterate over the permutations ("left" as disputed, "right" as candidate, and vice versa)
        for j, (disputed, candidate) in enumerate(
            itertools.permutations(pair.keys(), 2)
        ):
            round_score = 0

            for _ in range(self.rounds):
                # 1) Select random features
                keep_idxs = self._select_random_features(feature_count)

                # 2) Reduce TF-IDF vectors
                disputed_vec = self._reduce_vector(pair[disputed]["tfidf"], keep_idxs)
                candidate_vec = self._reduce_vector(pair[candidate]["tfidf"], keep_idxs)
                impostor_vecs = [
                    self._reduce_vector(vec, keep_idxs)
                    for vec in pair[candidate]["impostors_tfidf"]
                ]

                # 3) Compute similarities
                impostor_scores = [
                    self.similarity_fn(disputed_vec, iv) for iv in impostor_vecs
                ]

                best_impostor = max(impostor_scores)
                sim_with_candidate = self.similarity_fn(disputed_vec, candidate_vec)

                # 4) Score increment
                round_score += sim_with_candidate > best_impostor

            # 5) Running mean over permutations
            total_score += round_score
            # right-tail binomial test
            p_feat_name = f"{disputed}_disputed_{candidate}_candidate_uncorrected_p_value"
            pvalue = binom_test(count=int(round_score), nobs=self.rounds, prop=1 / (1 + len(pair[candidate]["impostors_tfidf"])),
                               alternative='larger')
            p_values[p_feat_name] = pvalue
            total_score /= j + 1

        return total_score, p_values
