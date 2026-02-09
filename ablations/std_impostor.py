"""
Caesar (O2-only by Kestemont et al., 2016) variant aligned with the ImpostorDetector pipeline.

This keeps the original Impostor logic intact and only swaps the feature
representation to TF-STD (Burrows, 2002) with Caesar-style word/char n-grams.
"""
from __future__ import annotations

import typing as t
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from genai_detection.detectors.components.vector_similarity import (cosine_similarity, )
from genai_detection.detectors.impostor import ImpostorDetector


class CaesarTFStdFeatureExtractor:
    """
    Feature extractor adapter for the Impostor pipeline using TF-STD.

    It mirrors the Caesar TF-STD representation but exposes `fit_transform`
    and `vectorizer.vocabulary_` to match the ImpostorDetector interface.
    """

    def __init__(
        self,
        feature_type: t.Literal["word", "char"] = "char",
        char_n: int = 3,
        lowercase: bool = False,
        preprocess_fn: t.Optional[t.Callable[[str], str]] = None,
    ):
        """
        Configure the TF-STD feature extractor.

        Inputs:
            feature_type: "word" for unigrams or "char" for fixed-length chars.
            char_n: Character n-gram size (only used when feature_type="char").
            lowercase: Whether to lowercase before vectorization.
            preprocess_fn: Optional preprocessing function applied to each text.
        """
        self.feature_type = feature_type
        self.char_n = char_n
        self.lowercase = lowercase
        self.preprocess_fn = preprocess_fn or (lambda s: s)
        self.vectorizer = SimpleNamespace(vocabulary_={})
        self._count_vectorizer: CountVectorizer | None = None

    def _build_vectorizer(self) -> CountVectorizer:
        """Build a CountVectorizer for word or fixed-length char n-grams."""
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
        """L2-normalize rows of a dense matrix, guarding division by zero."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def fit_transform(self, texts: t.List[str]) -> sp.csr_matrix:
        """
        Fit TF-STD features and return a sparse matrix.

        Inputs:
            texts: List of raw documents.

        Returns:
            CSR matrix (documents x features) with TF-STD features.
        """
        texts = [self.preprocess_fn(t) for t in texts]
        self._count_vectorizer = self._build_vectorizer()
        counts = self._count_vectorizer.fit_transform(texts).toarray().astype(np.float64)

        doc_lengths = counts.sum(axis=1, keepdims=True)
        doc_lengths[doc_lengths == 0] = 1.0
        tf = counts / doc_lengths

        std = tf.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        scaled = tf / std
        normalized = self._l2_normalize(scaled)

        self.vectorizer.vocabulary_ = dict(self._count_vectorizer.vocabulary_)
        return sp.csr_matrix(normalized)


class StdImpostor(ImpostorDetector):
    """
    Caesar Paper O2 variant that reuses the ImpostorDetector pipeline.

    Only differences from the original impostor approach:
      - TF-STD representation with word/char n-grams.
    """

    def __init__(
        self,
        *args,
        feature_type: t.Literal["word", "char"] = "char",
        char_n: int = 3,
        lowercase: bool = False,
        preprocess_fn: t.Optional[t.Callable[[str], str]] = None,
        **kwargs,
    ):
        """
        Initialize the Caesar O2 detector.

        Inputs:
            *args, **kwargs: Passed through to ImpostorDetector.
            feature_type: "word" or "char" n-grams for TF-STD.
            char_n: Character n-gram size for char features.
            lowercase: Whether to lowercase before vectorization.
            preprocess_fn: Optional preprocessing applied to each text.
        """
        super().__init__(*args, **kwargs)

        # Swap in TF-STD Caesar features.
        self.feature_extractor = CaesarTFStdFeatureExtractor(
            feature_type=feature_type,
            char_n=char_n,
            lowercase=lowercase,
            preprocess_fn=preprocess_fn,
        )
        # Ensure the length-matching tokenizer aligns with the feature type.
        self.pair_processor.tokenizer = self._std_tokenizer(feature_type, char_n)
        self.scorer.similarity_fn = cosine_similarity
        # Store outputs in ablation collection.
        self.impostor_output_collection = self.mongoDB.impostor_ablation_output_collection

    @staticmethod
    def _std_tokenizer(
        feature_type: t.Literal["word", "char"],
        char_n: int,
    ) -> t.Callable[[str], t.List[str]]:
        """Tokenizer for length matching in PairPreprocessor."""
        if feature_type == "word":
            return lambda text: text.split()
        if feature_type == "char":
            return lambda text: [text[i : i + char_n] for i in range(max(0, len(text) - char_n + 1))]
        raise ValueError(f"Unknown feature_type: {feature_type}")


__all__ = ["StdImpostor"]
