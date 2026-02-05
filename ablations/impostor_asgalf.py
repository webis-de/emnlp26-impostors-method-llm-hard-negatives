"""
Slightly modified GI-based impostor method (ASGALF-style scoring).

This module provides a drop-in variant of the existing ImpostorDetector that
only changes the per-round scoring aggregation, as described by
Khonji & Iraqi (2014).

The paper's larger impostor selection grouped on languages remain unimplemented here.
"""
from __future__ import annotations

import itertools
import logging
import re
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import nltk
import numpy as np

import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from genai_detection.detectors.components.scorer import Scorer, ScoreResult
from genai_detection.detectors.components.vector_similarity import (
    extended_minmax_similarity, )
from genai_detection.detectors.impostor import ImpostorDetector

logger = logging.getLogger(__name__)


class NltkPosTagger:
    """
    Lazy POS tagger wrapper to avoid downloads at import time.

    This class checks for the tagger resource only when tagging is requested.
    """

    _resource_candidates = (
        "taggers/averaged_perceptron_tagger_eng",
        "taggers/averaged_perceptron_tagger",
    )

    @classmethod
    def _ensure_available(cls) -> None:
        for resource in cls._resource_candidates:
            try:
                nltk.data.find(resource)
                return
            except LookupError:
                continue
        raise RuntimeError(
            "Missing NLTK tagger data. Install one of: "
            "'averaged_perceptron_tagger_eng' or 'averaged_perceptron_tagger'."
        )

    @classmethod
    def tag(cls, tokens: List[str]) -> List[str]:
        """
        Tag tokens with POS labels using NLTK.

        Inputs:
            tokens: List of word tokens.

        Returns:
            List of POS tags aligned with tokens.
        """
        cls._ensure_available()
        from nltk import pos_tag

        return [tag for _word, tag in pos_tag(tokens)]


class ASGALFTokenizer:
    """
    Tokenization utilities used by ASGALF feature extraction.

    This wrapper centralizes token, shape, and POS tagging logic and isolates
    NLTK usage from the feature extractor.
    """

    @staticmethod
    def word_tokens(text: str) -> List[str]:
        """
        Tokenize into word-like units (letters/digits with optional apostrophes).

        Inputs:
            text: Raw document text.

        Returns:
            List of token strings.
        """
        # Wordpunct tokenization without external resources.
        return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)

    @staticmethod
    def word_shapes(tokens: List[str]) -> List[str]:
        """
        Convert tokens into simplified shape patterns (C/c/N).

        Inputs:
            tokens: List of word tokens.

        Returns:
            List of shape strings aligned to tokens.
        """
        shapes = []
        for token in tokens:
            # Uppercase -> C, lowercase -> c, digits -> N.
            t1 = re.sub("[A-Z]", "C", token)
            t2 = re.sub("[a-z]", "c", t1)
            t3 = re.sub("[0-9]", "N", t2)
            shapes.append(t3)
        return shapes

    @staticmethod
    def pos_tags(tokens: List[str]) -> List[str]:
        """
        Compute POS tags for tokens using NLTK.

        Inputs:
            tokens: List of word tokens.

        Returns:
            List of POS tag strings.

        Raises:
            RuntimeError: If required NLTK tagger data is missing.
        """
        return NltkPosTagger.tag(tokens)

class ASGALFScorer(Scorer):
    """
    Scorer implementing the extended min-max score from Khonji & Iraqi (2014).

    - Scorer that replaces the binary win-count with a ratio-based score.
        The ratio follows the "slightly modified" GI-based verification method:
            score += sim(v1, v2)^2 / (sim(v1, x1) * sim(v2, x2))
        where x1 and x2 are the most similar impostors to v1 and v2, respectively.
    - more features
    """

    def score_pair(self, pair: Dict[str, Any], vectorizer) -> ScoreResult:
        """
        Score a single pair using ASGALF aggregation.

        Inputs:
            pair: Mapping with "left"/"right" entries containing TF-IDF vectors and impostors.
            vectorizer: Fitted vectorizer-like object with `vocabulary_`.

        Returns:
            Tuple[float, Dict]: (aggregate score, empty p-values dict).
        """
        assert (
            vectorizer.vocabulary_ is not None
        ), "Vectorizer must be fitted before scoring."

        feature_count = len(vectorizer.vocabulary_)
        total_score = 0.0

        # Iterate over both directions: disputed/candidate and candidate/disputed.
        for j, (disputed, candidate) in enumerate(
            itertools.permutations(pair.keys(), 2)
        ):
            round_score = 0
            for _ in range(self.rounds):
                # 1) Select random features to keep this round.
                keep_idxs = self._select_random_features(feature_count)

                # 2) Reduce TF-IDF vectors to selected feature subset.
                disputed_vec = self._reduce_vector(pair[disputed]["tfidf"], keep_idxs)
                candidate_vec = self._reduce_vector(pair[candidate]["tfidf"], keep_idxs)
                candidate_impostor_vecs = [
                    self._reduce_vector(vec, keep_idxs)
                    for vec in pair[candidate]["impostors_tfidf"]
                ]

                # 3) Compute ASGALF similarity for this round and accumulate.
                round_score += self.similarity_fn(disputed_vec, candidate_vec, candidate_impostor_vecs)
            total_score += round_score
            total_score /= j + 1

        empty_pvals: Dict[str, float] = {}
        return ScoreResult(score=total_score, p_values=empty_pvals)


class ASGALFFeatureExtractor:
    """
    Feature extractor for ASGALF (Khonji & Iraqi, 2014).

    Features:
      - Letter, word, function word, word shape, POS tag, and POS-word n-grams (n in 1..10).
      - Body richness = unique words / total words.

    Frequency filtering:
      - Keep features that appear at least `min_token_count` times in any single document.
    """

    def __init__(
        self,
        ngram_max: int = 10,
        ngram_min: int = 1,
        min_token_count: int = 5,
        function_words: Iterable[str] | None = None,
        tokenizer: ASGALFTokenizer | None = None,
    ):
        """
        Initialize feature extraction configuration.

        Inputs:
            ngram_max: Maximum n-gram size used for feature generation.
            ngram_min: Minimum n-gram size used for feature generation.
            min_token_count: Minimum per-document token count to keep a feature.
            function_words: Optional custom function-word list, defaults to sklearn's ENGLISH_STOP_WORDS.
            tokenizer: Optional tokenizer implementation for tokens, shapes, and POS tags.

        Side effects:
            Initializes an internal TF-IDF vectorizer placeholder and vocabulary.
        """
        if ngram_min <= 0 or ngram_min > ngram_max:
            raise ValueError(
                f"Invalid n-gram bounds: ngram_min={ngram_min}, ngram_max={ngram_max}. "
                "Expected 0 < ngram_min <= ngram_max."
            )
        self.ngram_max = ngram_max
        self.ngram_min = ngram_min
        self.min_token_count = min_token_count
        self.function_words = (
            {w.lower() for w in function_words}
            if function_words is not None
            else set(ENGLISH_STOP_WORDS)
        )
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self.vectorizer = SimpleNamespace(vocabulary_={})
        self.tokenizer = tokenizer or ASGALFTokenizer()

    @staticmethod
    def _letters_only(text: str) -> str:
        """
        Strip everything except ASCII letters.

        Inputs:
            text: Raw document text.

        Returns:
            String containing only letters A-Z/a-z.
        """
        return re.sub(r"[^A-Za-z]", "", text)

    @staticmethod
    def _word_tokens(text: str) -> List[str]:
        """
        Tokenize into word-like units (letters/digits with optional apostrophes).

        Inputs:
            text: Raw document text.

        Returns:
            List of token strings.
        """
        # Deprecated: kept for backward compatibility; use ASGALFTokenizer instead.
        return ASGALFTokenizer.word_tokens(text)

    @staticmethod
    def _word_shapes(tokens: List[str]) -> List[str]:
        """
        Convert tokens into simplified shape patterns (C/c/N).

        Inputs:
            tokens: List of word tokens.

        Returns:
            List of shape strings aligned to tokens.
        """
        # Deprecated: kept for backward compatibility; use ASGALFTokenizer instead.
        return ASGALFTokenizer.word_shapes(tokens)

    @staticmethod
    def _pos_tags(tokens: List[str]) -> List[str]:
        """
        Compute POS tags for tokens using NLTK.

        Inputs:
            tokens: List of word tokens.

        Returns:
            List of POS tag strings.
        """
        # Deprecated: kept for backward compatibility; use ASGALFTokenizer instead.
        return ASGALFTokenizer.pos_tags(tokens)

    @staticmethod
    def _ngrams(tokens: List[str], n: int) -> List[str]:
        """
        Build contiguous n-grams from a token sequence.

        Inputs:
            tokens: List of token strings.
            n: N-gram size.

        Returns:
            List of n-gram strings (joined by underscores).
        """
        if n <= 0 or len(tokens) < n:
            return []
        return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _analyze(self, text: str) -> List[str]:
        """
        Generate ASGALF feature tokens for a single document.

        Inputs:
            text: Raw document text.

        Returns:
            List of feature strings for TF-IDF.
        """
        tokens = self.tokenizer.word_tokens(text)
        features: List[str] = []

        # char n-grams.
        letters = self._letters_only(text)
        for n in range(self.ngram_min, self.ngram_max + 1):
            for i in range(0, max(0, len(letters) - n + 1)):
                features.append(f"{letters[i:i+n]}")

        # Word n-grams.
        features.extend(self._add_ngrams(tokens=tokens))

        # Function word n-grams.
        func_tokens = [t.lower() for t in tokens if t.lower() in self.function_words]
        features.extend(self._add_ngrams(tokens=func_tokens))

        # Word shape n-grams.
        shapes = self.tokenizer.word_shapes(tokens)
        features.extend(self._add_ngrams(tokens=shapes))

        # POS tag n-grams.
        pos_tags = self.tokenizer.pos_tags(tokens)
        features.extend(self._add_ngrams(tokens=pos_tags))

        # POS-word n-grams.
        pos_words = [f"{w}-{t}" for w, t in zip(tokens, pos_tags)]
        features.extend(self._add_ngrams(tokens=pos_words))

        return features

    def _add_ngrams(self, tokens: list[str]) -> list[str]:
        """
        Build n-gram features (from ngram_min..ngram_max) for the token sequence.

        Inputs:
            tokens: Token sequence to generate n-grams from.

        Returns:
            List of n-gram feature strings.
        """
        ngrams: list[str] = []
        for n in range(self.ngram_min, self.ngram_max + 1):
            for gram in self._ngrams(tokens, n):
                ngrams.append(f"{gram}")
        return ngrams

    def _build_vocabulary(self, corpus: List[str]) -> Dict[str, int]:
        """
        Build a vocabulary of features that meet the minimum per-document count.

        Inputs:
            corpus: List of raw documents.

        Returns:
            Mapping from feature string to column index.
        """
        max_counts: Dict[str, int] = {}
        for doc in corpus:
            counts: Dict[str, int] = {}
            for token in self._analyze(doc):
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                prev = max_counts.get(token, 0)
                if count > prev:
                    max_counts[token] = count

        tokens = [t for t, c in max_counts.items() if c >= self.min_token_count]
        tokens.sort()
        return {token: idx for idx, token in enumerate(tokens)}

    def _body_richness(self, corpus: List[str]) -> np.ndarray:
        """
        Compute body richness (unique/total tokens) for each document.

        Inputs:
            corpus: List of raw documents.

        Returns:
            1D numpy array of body richness values, aligned with corpus order.
        """
        values = []
        for doc in corpus:
            tokens = self.tokenizer.word_tokens(doc)
            if not tokens:
                values.append(0.0)
            else:
                values.append(len(set(tokens)) / float(len(tokens)))
        return np.asarray(values, dtype=float)

    def fit_transform(self, corpus: List[str]) -> sp.csr_matrix:
        """
        Fit TF-IDF on ASGALF features and return the combined feature matrix.

        Inputs:
            corpus: List of raw documents.

        Returns:
            scipy.sparse.csr_matrix with TF-IDF features plus body richness.
        """
        if not corpus:
            raise ValueError("The corpus cannot be empty.")

        vocabulary = self._build_vocabulary(corpus)
        self._tfidf_vectorizer = TfidfVectorizer(
            analyzer=self._analyze,
            vocabulary=vocabulary,
        )
        tfidf = self._tfidf_vectorizer.fit_transform(corpus)

        body_richness = self._body_richness(corpus)
        body_richness_col = sp.csr_matrix(body_richness.reshape(-1, 1))
        combined = sp.hstack([tfidf, body_richness_col], format="csr")

        # Extend vocabulary with body richness feature for downstream indexing.
        extended_vocab = dict(vocabulary)
        extended_vocab["body_richness"] = len(vocabulary)
        self.vectorizer.vocabulary_ = extended_vocab

        return combined


class ASGALFImpostorDetector(ImpostorDetector):
    """
    Impostor detector variant using ASGALF-style score aggregation.

    This class keeps the full impostor pipeline intact and only swaps the scorer
    to reflect the ratio-based aggregation in Khonji & Iraqi (2014).
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the ASGALF ablation detector.

        Inputs:
            *args, **kwargs: Passed through to ImpostorDetector.

        Side effects:
            Swaps the feature extractor and scorer, and redirects output collection.
        """
        super().__init__(*args, **kwargs)
        # Swap only the scoring logic and feature inventory; keep the rest unchanged.
        self.feature_extractor = ASGALFFeatureExtractor(
            ngram_max=10,
            min_token_count=5,
        )
        self.scorer = ASGALFScorer(
            rounds=self.rounds,
            portion_delete=self.portion_delete,
            similarity_fn=extended_minmax_similarity,
        )
        # save outputs to extra ablation output collection
        self.impostor_output_collection = self.mongoDB.impostor_ablation_output_collection


__all__ = ["ASGALFImpostorDetector"]
