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
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from genai_detection.config import CONFIG
from genai_detection.detectors.detector_base import DetectorBase
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

class ImpostorBase(DetectorBase):
    def __init__(self):
        """
        Initialize the impostorBase detector.
        Both the impostor approach and its supervised and unsupervised baselines
        extend this base class.
        """
        super().__init__()
        self.mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))

    # TODO: obsolete? TFIDF vectorizer has this built-in
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
        self.train_dataset = pd.DataFrame(self.mongoDB.get_training_data_from_original_texts(dataset_name=self.dataset_name))
        print("Number of training pairs %d (in-memory)", self.train_dataset.shape[0])

        # texts = pd.concat(
        #     [
        #         self.train_dataset["left_text"],
        #         self.train_dataset["right_text"],
        #     ],
        #     ignore_index=True,
        # )
        texts = pd.DataFrame(self.mongoDB.original_collection.find({"dataset":self.dataset_name},projection={
            '_id': False,"text":True}))
        # print("Basis for fitting tfidf vectorizer", len(texts))
        # print(texts["text"].iloc[0][:500])

        self._vectorizer = TfidfVectorizer(
            max_features=100000,
            dtype=np.float32,
            ngram_range=(4, 4), analyzer="char_wb", min_df=2
        )

        self._vectorizer = self._vectorizer.fit(texts["text"])
        # vector = self._vectorizer.transform([texts["text"].iloc[0]]).toarray()[0]
        # print(vector, np.count_nonzero(vector))
        print("Fitted vectorizer.")

    def get_tfidf_vector_for_text(self, texts: List[str]):
        """
        Get the TF-IDF vector for a list of texts.

        :param texts: The input texts to vectorize.
        :return: A TF-IDF vector as a NumPy array.
        """
        if not isinstance(texts, list):
            # pandas Series or generator
            texts = list(texts)
        assert texts[0] != texts[1], "Texts are identical"
        tfidf_matrix = self._vectorizer.transform(texts)  # format: (n_samples=1, n_features=self.top_n)
        n_zero = [np.count_nonzero(vector) == 0 for vector in tfidf_matrix.toarray()]
        assert sum(n_zero) == 0, f"Found all-zero tfidf vector, number of non-zero tfidf vector: {sum(n_zero)}"

        return tfidf_matrix.toarray()
