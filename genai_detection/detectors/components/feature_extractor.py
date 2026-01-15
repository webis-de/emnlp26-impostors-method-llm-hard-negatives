from functools import partial
from typing import List

import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
import re


class TfidfFeatureExtractor:
    def __init__(self, top_n_freq_words: int, ngram_n: int = 4, min_df: int = 1):
        # Koppel et al. (2014): space-free character 4-grams.
        # do not use built-in analyzer="char_wb", bc for input "hello to" it returns: "hello": " hel", "hell", "ello", "llo ", " to ",
        # but we want only: "hell", "ello", "to ", i.e., dropping " hel" and "llo "
        self.vectorizer = TfidfVectorizer(
            analyzer=partial(self._space_free_char_ngrams, n=ngram_n),
            min_df=min_df,
            max_features=top_n_freq_words,
        )

    @staticmethod
    def _space_free_char_ngrams(text: str, n: int = 4) -> List[str]:
        """
        Space-free character n-grams used by Koppel et al. (2014).
        A space-free n-gram is a sequence of n characters with no whitespace in it,
        plus sequences shorter than n padded with spaces.
        """
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        # tokens with length >= n
        n_grams = [
            text[i : i + n]
            for i in range(0, len(text) - n + 1)
            if " " not in text[i : i + n]
        ]
        # tokens with n - 2 <= length < n
        for token in text.split():
            if (len(token) < n) and ((n - 2) <= len(token)):
                n_grams.append(token + " " * (n - len(token)))
        return n_grams

    def fit_transform(self, corpus: List[str]) -> scipy.sparse.csr_matrix:
        """
        Returns a numpy array with shape (len(corpus), len(feature_extractor.vectorizer.vocabulary_))
        :param corpus: List of strings; each string is a document.
        :return: The sparse matrix with shape (len(corpus), len(feature_extractor.vectorizer.vocabulary_)) with tfidf features.
        """
        assert len(corpus) > 0, "The corpus cannot be empty."
        assert all(
            [type(doc) == str for doc in corpus]
        ), "The corpus cannot contain non-string."
        return self.vectorizer.fit_transform(corpus)
