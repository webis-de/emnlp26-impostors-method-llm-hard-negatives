import re
from more_itertools import ichunked
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from genai_detection.config import CONFIG
from genai_detection.detectors.detector_base import DetectorBase
from collections import Counter, defaultdict
from datasets import load_from_disk
import heapq
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class ImposterBase(DetectorBase):
    def __init__(self):
        """
        Initialize the ImposterBase detector.
        Both the Imposter approach and its supervised and unsupervised baselines
        extend this base class.
        """
        super().__init__()

    @staticmethod
    def tokenize_char_ngrams(
        text: str, n: int = 4, normalize_ws: bool = True, space_free: bool = True
    ):
        """
        Tokenize input text into character n-grams.
        Koppel et Al. (2014) use space-free character 4-grams tfidf values to represent each document as a numerical vector.
        A space-free n-grams is a (1) sequence of n characters without any whitespace in it, (2) a sequence of <= n characters surrounded by spaces.

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
        Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.

        :param text: input text
        :param n: n-gram order
        :param normalize_ws: collapse whitespace before tokenization
        :return: list of n-gram tokens
        """
        # remove first and last whitespace
        text = text.strip()
        if normalize_ws:
            text = re.sub(r"\s+", " ", text)
        if space_free:
            # of size n without spaces
            n_grams = [
                text[i : i + n]
                for i in range(0, len(text) - n + 1)
                if " " not in text[i : i + n]
            ]
            # add m-grams with spaces, where m < n
            # TODO: should be >2 whitespaces to pad be allowed? I don't think so, produces: 'to  ', '  to'
            for token in text.split():
                if (len(token) < n) and ((n - 2) <= len(token)):
                    # add all n-grams options with spaces
                    # TODO: results in most common ngrams ('the ', 22), (' the', 22), 'to  ': 10, ' to ': 10, '  to': 10,
                    n_grams.extend(
                        " " * i + token + " " * (n - len(token) - i)
                        for i in range(n - len(token) + 1)
                    )
            return n_grams

        else:
            return [text[i : i + n] for i in range(0, len(text) - n + 1)]

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Calculate cosine similarity between two vectors.
        Koppel et Al. (2014) have use cosine similarity as a baseline.
        """
        return (
            cosine_similarity(vec1, vec2).flatten()[0]
            if vec1 is not None and vec2 is not None
            else 0.0
        )

    def minmax_similarity(self, vec1, vec2):
        """
        Calculate min-max similarity between two vectors in TFIDF format.
        Koppel et Al. (2014) use min-max similarity.
        """
        if vec1 is None or vec2 is None:
            return 0.0
        assert len(vec1) == len(vec2), "Vectors must be of the same length."
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        numerator = np.minimum(vec1, vec2).sum()
        denominator = np.maximum(vec1, vec2).sum()
        return 0.0 if denominator == 0 else numerator / denominator


class ImposterBaselineBase(ImposterBase):
    """
    Base class for Imposter Baseline detectors.
    This class provides the basic structure for scoring and prediction methods.
    """

    def __init__(self):
        super().__init__()
        self.dataset = load_from_disk(
            Path(__file__).resolve().parents[2] / CONFIG.CROSS_GENRE
        )["train"].to_pandas()
        self._vectorizer = TfidfVectorizer(
            vocabulary=self.get_top_tokens(), input="content", dtype=np.float32
        ).fit(
            [
                " ".join(self.tokenize_char_ngrams(t))
                for t in self.dataset["disputed_text"].tolist()
            ]
        )

    def get_top_tokens(self, max_tokens: int = 100000):
        """
        Get the top tokens from the corpus.

        :param max_tokens: The maximum number of tokens to return.
        :return: A list of the top tokens.
        """
        all_texts = self.dataset["disputed_text"].tolist()
        tokens = [self.tokenize_char_ngrams(text, 4) for text in all_texts]
        flat_list = [item for sublist in tokens for item in sublist]
        freqs = Counter(flat_list)
        freqs = Counter({k: v for k, v in freqs.items() if v > 1})
        return heapq.nlargest(max_tokens, tokens, key=lambda x: freqs[x])

    def get_tfidf_vector_for_text(self, text: str):
        """
        Get the TF-IDF vector for a given text.

        :param text: The input text to vectorize.
        :return: A TF-IDF vector as a NumPy array.
        """
        ngrams = self.tokenize_char_ngrams(text, 4)
        tfidf_matrix = self._vectorizer.fit_transform(
            [" ".join(ngrams)]
        )  # format (n_samples=1, n_features=self.top_n)

        return tfidf_matrix.toarray()
