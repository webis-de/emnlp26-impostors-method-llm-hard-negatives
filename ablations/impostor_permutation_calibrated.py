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

"""Permutation-calibrated ablation of the directional Impostors test.

The standard implementation uses the same repeated random-feature deletion score,
then converts the directional win count to a p-value with a binomial test.  That
binomial step assumes the random-feature rounds are independent Bernoulli trials.
For the Impostors method this assumption is questionable because every round
reuses the same document pair, the same impostor pool, and overlapping feature
subsets.

The ablation below keeps the original scoring event exactly the same: in each
round, the candidate document receives one win if it is more similar to the
disputed document than the best impostor.  Only the p-value calibration changes.
Instead of comparing the observed win count to ``Binomial(rounds, p0)``, it
constructs a pair-local empirical null by permuting which member of the
candidate/impostor pool is treated as the candidate.  This preserves the
document pair, feature extraction, similarity function, impostor texts, and
random-feature deletion mechanism while asking whether the real candidate is
unusually strong relative to exchangeable pseudo-candidates from the same pool.

The detector subclass follows the existing ablation pattern in this repository:
all preprocessing, impostor generation, feature extraction, persistence, and
evaluation code are inherited from ``ImpostorDetector``.  The only swapped
component is ``self.scorer``.  Results are written to the ablation MongoDB
collection because ``self.impostor_output_collection`` is set to
``mongoDB.impostor_ablation_output_collection``.
"""

from __future__ import annotations

import collections
import itertools
import random
from typing import Any, Dict, List

import numpy as np

from genai_detection.detectors.components.scorer import ScoreResult, Scorer
from genai_detection.detectors.components.vector_similarity import minmax_similarity
from genai_detection.detectors.impostor import ImpostorDetector


class PermutationCalibratedScorer(Scorer):
    """Impostor scorer with empirical, pair-local permutation p-values.

    The score returned by this class remains on the original Impostor scale:
    the average number of candidate wins across the two directional tests.  This
    makes the ablation comparable to the standard detector and keeps downstream
    threshold-based evaluation unchanged.

    The p-values are the ablated part.  For each direction, the observed
    candidate win count is compared with null win counts obtained by shuffling
    the real candidate vector together with the candidate-side impostor vectors.
    The first shuffled vector is treated as a pseudo-candidate and the remaining
    vectors are treated as its pseudo-impostors.  This tests whether the real
    candidate is stronger than expected under exchangeability within the local
    candidate/impostor pool without assuming independent feature rounds.
    """

    def __init__(
        self,
        rounds: int = 50,
        portion_delete: float = 0.4,
        similarity_fn=minmax_similarity,
        n_permutations: int = 999,
        random_seed: int | None = None,
    ):
        """Create the permutation-calibrated scorer.

        Args:
            rounds: Number of random-feature deletion rounds per directional
                score, matching the original Impostor implementation.
            portion_delete: Fraction of features deleted in each round.
            similarity_fn: Vector similarity used by the Impostor method.
            n_permutations: Number of pair-local null scores.  The smallest
                possible non-zero p-value is ``1 / (n_permutations + 1)``.
            random_seed: Optional seed for reproducible ablation runs.  When
                omitted, Python's process-level randomness is used, matching the
                existing scorer's stochastic behavior.
        """
        super().__init__(
            rounds=rounds,
            portion_delete=portion_delete,
            similarity_fn=similarity_fn,
        )
        if n_permutations < 1:
            raise ValueError("n_permutations must be at least 1.")
        self.n_permutations = n_permutations
        self._rng = random.Random(random_seed) if random_seed is not None else random

    def _select_random_features(self, total_features: int) -> List[int]:
        """Return a random subset of feature indices to keep.

        The parent scorer uses ``random.sample`` directly.  This override keeps
        the behavior equivalent while allowing the ablation to be seeded without
        changing global randomness in the rest of the pipeline.
        """
        keep = int(total_features * (1 - self.portion_delete))
        return self._rng.sample(range(total_features), keep)

    def _directional_win_count(
        self,
        *,
        disputed_tfidf: List[float],
        candidate_tfidf: List[float],
        impostors_tfidf: List[List[float]],
        feature_count: int,
    ) -> int:
        """Count candidate wins over repeated random-feature subsets.

        This is the original directional Impostor event factored into a helper
        so that both the observed statistic and every permutation-null statistic
        are computed by the exact same code path.
        """
        wins = 0
        for _ in range(self.rounds):
            keep_idxs = self._select_random_features(feature_count)
            disputed_vec = self._reduce_vector(disputed_tfidf, keep_idxs)
            candidate_vec = self._reduce_vector(candidate_tfidf, keep_idxs)
            impostor_vecs = [
                self._reduce_vector(vec, keep_idxs) for vec in impostors_tfidf
            ]

            best_impostor = max(
                self.similarity_fn(disputed_vec, impostor_vec)
                for impostor_vec in impostor_vecs
            )
            candidate_similarity = self.similarity_fn(disputed_vec, candidate_vec)
            wins += candidate_similarity > best_impostor
        return wins

    def _permutation_p_value(
        self,
        *,
        observed_wins: int,
        disputed_tfidf: List[float],
        candidate_tfidf: List[float],
        impostors_tfidf: List[List[float]],
        feature_count: int,
    ) -> float:
        """Estimate the right-tail p-value from pair-local permutations.

        The add-one numerator and denominator are the standard finite-permutation
        correction.  They avoid zero p-values and make the reported p-value
        interpretable as the fraction of null scores at least as extreme as the
        observed score.
        """
        pool = [candidate_tfidf, *impostors_tfidf]
        null_scores_at_least_observed = 0
        for _ in range(self.n_permutations):
            shuffled_pool = list(pool)
            self._rng.shuffle(shuffled_pool)
            pseudo_candidate = shuffled_pool[0]
            pseudo_impostors = shuffled_pool[1:]
            null_wins = self._directional_win_count(
                disputed_tfidf=disputed_tfidf,
                candidate_tfidf=pseudo_candidate,
                impostors_tfidf=pseudo_impostors,
                feature_count=feature_count,
            )
            null_scores_at_least_observed += null_wins >= observed_wins

        return (1 + null_scores_at_least_observed) / (1 + self.n_permutations)

    def score_pair(self, pair: Dict[str, Any], vectorizer) -> ScoreResult:
        """Score a pair and return permutation-calibrated directional p-values."""
        assert (
            vectorizer.vocabulary_ is not None
        ), "Vectorizer must be fitted before scoring."

        feature_count = len(vectorizer.vocabulary_)
        directional_scores: list[int] = []
        p_values = collections.defaultdict(float)

        for disputed, candidate in itertools.permutations(pair.keys(), 2):
            observed_wins = self._directional_win_count(
                disputed_tfidf=pair[disputed]["tfidf"],
                candidate_tfidf=pair[candidate]["tfidf"],
                impostors_tfidf=pair[candidate]["impostors_tfidf"],
                feature_count=feature_count,
            )
            directional_scores.append(observed_wins)
            p_feat_name = (
                f"{disputed}_disputed_{candidate}_candidate_uncorrected_p_value"
            )
            p_values[p_feat_name] = self._permutation_p_value(
                observed_wins=observed_wins,
                disputed_tfidf=pair[disputed]["tfidf"],
                candidate_tfidf=pair[candidate]["tfidf"],
                impostors_tfidf=pair[candidate]["impostors_tfidf"],
                feature_count=feature_count,
            )

        return ScoreResult(
            score=float(np.mean(directional_scores)),
            p_values=p_values,
        )


class PermutationCalibratedImpostorDetector(ImpostorDetector):
    """Impostor detector ablation with permutation-calibrated p-values.

    This class exists to make the calibration experiment selectable as an
    ablation while preserving the original Impostor detector.  It reuses the
    entire ``ImpostorDetector`` pipeline and swaps only the scorer.  Because the
    output collection is set to the ablation collection, inherited persistence
    automatically stores documents with the ``ablation`` field set to this class
    name.
    """

    def __init__(
        self,
        *args,
        n_permutations: int = 999,
        random_seed: int | None = None,
        **kwargs,
    ):
        """Initialize the ablation detector.

        Args:
            *args: Positional arguments forwarded to ``ImpostorDetector``.
            n_permutations: Number of permutation-null scores per direction.
            random_seed: Optional seed for reproducible ablation runs.
            **kwargs: Keyword arguments forwarded to ``ImpostorDetector``.
        """
        super().__init__(*args, **kwargs)
        self.scorer = PermutationCalibratedScorer(
            rounds=self.rounds,
            portion_delete=self.portion_delete,
            similarity_fn=minmax_similarity,
            n_permutations=n_permutations,
            random_seed=random_seed,
        )
        self.impostor_output_collection = (
            self.mongoDB.impostor_ablation_output_collection
        )


__all__ = [
    "PermutationCalibratedImpostorDetector",
    "PermutationCalibratedScorer",
]
