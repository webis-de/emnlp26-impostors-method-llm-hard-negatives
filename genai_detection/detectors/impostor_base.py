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
import os
import re
from collections import Counter
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from genai_detection.config import CONFIG
from genai_detection.detectors.detector_base import DetectorBase
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

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
        self.mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))

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


class ImpostorBaselineBase(ImpostorBase):
    """
    Base class for impostor Baseline detectors.
    This class provides the basic structure for scoring and prediction methods.
    """

    def __init__(self, dataset_name: str = CONFIG.STUDENT_ESSAYS):
        super().__init__()
        # get all original texts from mongodb collection whose ID is not in test pairs mongodb collection
        self.dataset_name = dataset_name
        train_dataset_generator = self.mongoDB.get_training_data_from_original_texts(dataset_name=dataset_name)
        self.train_dataset = pd.DataFrame(list(train_dataset_generator))
        self.train_dataset["left_text"] = self.train_dataset["left_id"].apply(
            lambda x: self.mongoDB.get_text_or_id_from_orginal_collection(
                text=None, text_id=x
            )
        )
        self.train_dataset["right_text"] = self.train_dataset["right_id"].apply(
            lambda x: self.mongoDB.get_text_or_id_from_orginal_collection(
                text=None, text_id=x
            )
        )
        logging.info(f"Training dataset ready (in-memory).")

        def preprocessed_texts():
            # generator is exhausted after computing the vocabulary, and loading whole data into memory is not a good idea
            for pair in self.train_dataset:
                yield " ".join(self.tokenize_char_ngrams(pair["left_text"]))
                yield " ".join(self.tokenize_char_ngrams(pair["right_text"]))

        self._vectorizer = TfidfVectorizer(
            vocabulary=self.get_top_tokens(), input="content", dtype=np.float32).fit(preprocessed_texts())
        logging.info("Fitted vectorizer.")

    def get_top_tokens(self, max_tokens: int = 100000):
        """
        Get the top tokens from the corpus.

        :param max_tokens: The maximum number of tokens to return.
        :return: A list of the top tokens.
        """
        freqs = Counter()

        # Iterate over generator and tokenize on the fly
        for doc in self.train_dataset_generator:
            text = doc["text"]
            tokens = self.tokenize_char_ngrams(text, n=4)  # adjust n if needed
            freqs.update(tokens)

        freqs = Counter({k: v for k, v in freqs.items() if v > 1})
        return [itemgetter(0)(item) for item in freqs.most_common(max_tokens)]#list(map(itemgetter(0), freqs.most_common(max_tokens)))

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
