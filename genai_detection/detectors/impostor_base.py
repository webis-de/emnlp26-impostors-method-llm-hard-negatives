# Copyright 2024 Klara M. Gutekunst, Webis
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
import logging
import re
from collections import Counter
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from genai_detection.config import CONFIG
from genai_detection.detectors.detector_base import DetectorBase

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )

class ImpostorBase(DetectorBase):
    def __init__(self):
        """
        Initialize the impostorBase detector.
        Both the impostor approach and its supervised and unsupervised baselines
        extend this base class.
        """
        super().__init__()

    @staticmethod
    def tokenize_char_ngrams(
        text: str, n: int = 4, normalize_ws: bool = True, space_free: bool = True
    ):
        """
        Tokenize input text into character n-grams.
        Koppel et al. (2014) use space-free character 4-grams tfidf values to represent each document as a numerical vector.
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
        try:
            text = text.strip()
        except AttributeError as e:
            logging.warning(f"Input text is not a string. Text: {text}...\nError: {e}")
            raise e
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
            # >2 whitespaces in character sequence are not paded, those are excluded. Otherwise, we include: 'to  ', '  to'
            for token in text.split():
                if (len(token) < n) and ((n - 2) <= len(token)):
                    n_grams.append(token + " " * (n - len(token)))
            return n_grams

        else:
            return [text[i : i + n] for i in range(0, len(text) - n + 1)]

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Calculate cosine similarity between two vectors.
        Koppel et al. (2014) have use cosine similarity as a baseline.
        """
        return (
            cosine_similarity(vec1, vec2).flatten()[0]
            if vec1 is not None and vec2 is not None
            else 0.0
        )

    def minmax_similarity(self, vec1, vec2):
        """
        Calculate min-max similarity between two vectors in TFIDF format.
        Koppel et al. (2014) use min-max similarity.
        """
        if vec1 is None or vec2 is None:
            return 0.0
        assert len(vec1) == len(vec2), "Vectors must be of the same length."
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        numerator = np.minimum(vec1, vec2).sum()
        denominator = np.maximum(vec1, vec2).sum()
        return 0.0 if denominator == 0 else numerator / denominator


class ImpostorBaselineBase(ImpostorBase):
    """
    Base class for impostor Baseline detectors.
    This class provides the basic structure for scoring and prediction methods.
    """

    def __init__(self, dataset_name: str = CONFIG.STUDENT_ESSAYS):
        super().__init__()
        dataset = (
            CONFIG.PATH2STUDENT_ESSAYS
            if dataset_name == CONFIG.STUDENT_ESSAYS
            else CONFIG.PATH2BLOG
        )
        self.dataset = load_from_disk(Path(__file__).resolve().parents[2] / dataset)[
            "train"
        ].to_pandas()
        self.dataset[["disputed_text", "candidate_text"]] = pd.DataFrame(
            self.dataset["pair"].tolist(), index=self.dataset.index
        )
        logging.info("Obtained dataset.")
        self._vectorizer = TfidfVectorizer(
            vocabulary=self.get_top_tokens(), input="content", dtype=np.float32
        ).fit(
            [
                " ".join(self.tokenize_char_ngrams(t))
                for t in self.dataset["disputed_text"].tolist()
            ]
        )
        logging.info("Fitted vectorizer.")

    def get_top_tokens(self, max_tokens: int = 100000):
        """
        Get the top tokens from the corpus.

        :param max_tokens: The maximum number of tokens to return.
        :return: A list of the top tokens.
        """
        logging.info("%s", self.dataset.columns)
        all_texts = self.dataset["disputed_text"].tolist()
        tokens = [self.tokenize_char_ngrams(text, 4) for text in all_texts]
        flat_list = [item for sublist in tokens for item in sublist]
        freqs = Counter(flat_list)
        freqs = Counter({k: v for k, v in freqs.items() if v > 1})
        return list(map(itemgetter(0), freqs.most_common(max_tokens)))

    def get_tfidf_vector_for_text(self, text: str):
        """
        Get the TF-IDF vector for a given text.

        :param text: The input text to vectorize.
        :return: A TF-IDF vector as a NumPy array.
        """
        ngrams = self.tokenize_char_ngrams(text, 4)
        tfidf_matrix = self._vectorizer.fit_transform(
            [" ".join(ngrams)]
        )  # format: (n_samples=1, n_features=self.top_n)

        return tfidf_matrix.toarray()
