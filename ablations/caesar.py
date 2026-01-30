"""
Caesar: re-implementation of the verification pipeline in
"Authenticating the writings of Julius Caesar" (Kestemont et al., 2016).

This module provides the two verification systems described in the paper:
O1 (first-order, profile-based) and O2 (second-order, General Imposters).
It is designed to highlight where the pipeline differs from the original
GI formulation in Koppel & Winter (2014).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import typing as t

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from genai_detection.detectors.detector_base import DetectorBase

Vector = np.ndarray


@dataclass
class CaesarProblem:
    """
    A single verification problem.

    unknown: the disputed document
    known:   documents by the target author
    impostors: documents by other authors, grouped by author
               (only required for O2)
    """

    unknown: str
    known: t.List[str]
    impostors: t.Optional[t.List[t.List[str]]] = None


class CaesarVectorizer:
    """
    Vectorizes texts using tf, tf-idf, or std VSMs, with unit-norm scaling.

    Feature types follow the paper: word unigrams, character trigrams,
    and character tetragrams, with full (untruncated) vocabularies.
    """

    def __init__(
        self,
        vsm: t.Literal["tf", "tfidf", "std"],
        feature_type: t.Literal["word", "char"],
        char_n: int = 3,
        lowercase: bool = False,
        preprocess_fn: t.Optional[t.Callable[[str], str]] = None,
    ):
        self.vsm = vsm
        self.feature_type = feature_type
        self.char_n = char_n
        self.lowercase = lowercase
        self.preprocess_fn = preprocess_fn or (lambda s: s)

    def _build_vectorizer(self):
        if self.feature_type == "word":
            return CountVectorizer(
                analyzer="word",
                ngram_range=(1, 1),
                lowercase=self.lowercase,
            )
        if self.feature_type == "char":
            return CountVectorizer(
                analyzer="char",
                ngram_range=(self.char_n, self.char_n),
                lowercase=self.lowercase,
            )
        raise ValueError(f"Unknown feature_type: {self.feature_type}")

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def fit_transform(self, texts: t.List[str]) -> np.ndarray:
        texts = [self.preprocess_fn(t) for t in texts]

        if self.vsm == "tfidf":
            vectorizer = TfidfVectorizer(
                analyzer="word" if self.feature_type == "word" else "char",
                ngram_range=(1, 1)
                if self.feature_type == "word"
                else (self.char_n, self.char_n),
                lowercase=self.lowercase,
                norm=None,
                use_idf=True,
                smooth_idf=True,
            )
            matrix = vectorizer.fit_transform(texts).toarray()
            return self._l2_normalize(matrix)

        vectorizer = self._build_vectorizer()
        counts = vectorizer.fit_transform(texts).toarray().astype(np.float64)
        doc_lengths = counts.sum(axis=1, keepdims=True)
        doc_lengths[doc_lengths == 0] = 1.0
        tf = counts / doc_lengths

        if self.vsm == "tf":
            return self._l2_normalize(tf)

        if self.vsm == "std":
            std = tf.std(axis=0, ddof=0)
            std[std == 0] = 1.0
            scaled = tf / std
            return self._l2_normalize(scaled)

        raise ValueError(f"Unknown vsm: {self.vsm}")


def manhattan_distance(a: Vector, b: Vector) -> float:
    return float(np.abs(a - b).sum())


def cosine_distance(a: Vector, b: Vector) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def minmax_similarity(a: Vector, b: Vector) -> float:
    numerator = np.minimum(a, b).sum()
    denominator = np.maximum(a, b).sum()
    return 0.0 if denominator == 0 else float(numerator / denominator)


def minmax_distance(a: Vector, b: Vector) -> float:
    return 1.0 - minmax_similarity(a, b)


def cng_distance(profile: Vector, unknown: Vector) -> float:
    """
    Common n-grams (cng) distance as used in the paper.

    Implementation follows the paper's note: compute the sum only over
    features present in the unknown document.
    """
    if profile.shape != unknown.shape:
        raise ValueError("Vectors must be of the same length.")

    idx = unknown > 0
    if not np.any(idx):
        return 0.0

    a = profile[idx]
    b = unknown[idx]
    denom = a + b
    denom[denom == 0] = 1.0
    frac = (2.0 * (a - b) / denom) ** 2
    return float(frac.sum())


class Caesar(DetectorBase):
    """
    Caesar verification pipeline with O1 and O2 systems.

    O1:
      - build author profile as mean centroid of known documents
      - compute first-order distance to unknown
      - scale using non-zero pairwise distances among known documents,
        then take the positive complement (1 - scaled)

    O2:
      - General Imposters with random feature subsets and imposter profiles
      - use impostor author profiles (mean centroids) for comparability
    """

    def __init__(
        self,
        system: t.Literal["O1", "O2"] = "O2",
        vsm: t.Literal["tf", "tfidf", "std"] = "tf",
        feature_type: t.Literal["word", "char"] = "char",
        char_n: int = 3,
        distance: t.Literal["cng", "manhattan", "cosine", "minmax"] = "minmax",
        rounds: int = 100,
        feature_subsample_ratio: float = 0.5,
        impostor_subsample_ratio: float = 0.5,
        threshold: float = 0.5,
        lowercase: bool = False,
        seed: t.Optional[int] = None,
        preprocess_fn: t.Optional[t.Callable[[str], str]] = None,
    ):
        super().__init__()
        self.system = system
        self.rounds = rounds
        self.feature_subsample_ratio = feature_subsample_ratio
        self.impostor_subsample_ratio = impostor_subsample_ratio
        self.threshold = threshold
        self.random = random.Random(seed)

        self.vectorizer = CaesarVectorizer(
            vsm=vsm,
            feature_type=feature_type,
            char_n=char_n,
            lowercase=lowercase,
            preprocess_fn=preprocess_fn,
        )
        self.distance_fn = self._select_distance(distance)

    @staticmethod
    def _select_distance(name: str) -> t.Callable[[Vector, Vector], float]:
        if name == "cng":
            return cng_distance
        if name == "manhattan":
            return manhattan_distance
        if name == "cosine":
            return cosine_distance
        if name == "minmax":
            return minmax_distance
        raise ValueError(f"Unknown distance metric: {name}")

    def _author_profile(self, vectors: np.ndarray) -> Vector:
        return vectors.mean(axis=0)

    def _scale_distance_to_probability(
        self, distance: float, known_vectors: np.ndarray
    ) -> float:
        """
        Scale using non-zero pairwise distances between known documents,
        then take the positive complement (1 - scaled).
        """
        n = known_vectors.shape[0]
        if n < 2:
            return 0.5

        scores = []
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance_fn(known_vectors[i], known_vectors[j])
                if d > 0:
                    scores.append(d)

        if not scores:
            return 0.5

        d_min = min(scores)
        d_max = max(scores)
        if math.isclose(d_min, d_max):
            scaled = 1.0
        else:
            scaled = (distance - d_min) / (d_max - d_min)
        scaled = max(0.0, min(1.0, scaled))
        return 1.0 - scaled

    def _o1_score(self, problem: CaesarProblem) -> float:
        texts = [problem.unknown] + problem.known
        matrix = self.vectorizer.fit_transform(texts)
        unknown_vec = matrix[0]
        known_vecs = matrix[1:]

        profile = self._author_profile(known_vecs)
        dist = self.distance_fn(profile, unknown_vec)
        return self._scale_distance_to_probability(dist, known_vecs)

    def _o2_score(self, problem: CaesarProblem) -> float:
        if not problem.impostors:
            raise ValueError("O2 requires impostor author documents.")

        # Build a single VSM across unknown, known, and all impostor docs.
        flat_impostors = [doc for author_docs in problem.impostors for doc in author_docs]
        texts = [problem.unknown] + problem.known + flat_impostors
        matrix = self.vectorizer.fit_transform(texts)

        unknown_vec = matrix[0]
        known_vecs = matrix[1 : 1 + len(problem.known)]

        # Build impostor author profiles (mean centroids).
        impostor_profiles = []
        idx = 1 + len(problem.known)
        for author_docs in problem.impostors:
            size = len(author_docs)
            author_vecs = matrix[idx : idx + size]
            impostor_profiles.append(self._author_profile(author_vecs))
            idx += size

        feature_count = unknown_vec.shape[0]
        n_feat_keep = max(1, int(feature_count * self.feature_subsample_ratio))

        imp_count = len(impostor_profiles)
        n_imp_keep = max(1, int(imp_count * self.impostor_subsample_ratio))

        wins = 0
        for _ in range(self.rounds):
            # Random feature subset.
            feat_idx = self.random.sample(range(feature_count), n_feat_keep)

            u = unknown_vec[feat_idx]
            t_vecs = known_vecs[:, feat_idx]
            imp_vecs = [p[feat_idx] for p in impostor_profiles]

            # Random subset of impostor profiles.
            imp_subset = self.random.sample(imp_vecs, n_imp_keep)

            min_target = min(self.distance_fn(u, t) for t in t_vecs)
            min_imp = min(self.distance_fn(u, imp) for imp in imp_subset)

            wins += min_target < min_imp

        return wins / self.rounds

    def score_problem(self, problem: CaesarProblem) -> float:
        if self.system == "O1":
            return self._o1_score(problem)
        if self.system == "O2":
            return self._o2_score(problem)
        raise ValueError(f"Unknown system: {self.system}")

    def _get_score_impl(self, problems: t.Iterable[CaesarProblem]) -> t.List[float]:
        return [self.score_problem(p) for p in problems]

    def get_score(
        self, problems: t.Union[CaesarProblem, t.Iterable[CaesarProblem]]
    ) -> t.List[float]:
        if isinstance(problems, CaesarProblem):
            return [self.score_problem(problems)]
        return self._get_score_impl(list(problems))

    def get_prediction(
        self, problems: t.Union[CaesarProblem, t.Iterable[CaesarProblem]]
    ) -> t.List[bool]:
        scores = self.get_score(problems)
        return [score >= self.threshold for score in scores]


__all__ = ["Caesar", "CaesarProblem"]
