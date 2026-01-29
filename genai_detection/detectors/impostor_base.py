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
from typing import List

import numpy as np
import pandas as pd

from genai_detection.config import CONFIG
from genai_detection.detectors.components.feature_extractor import TfidfFeatureExtractor
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

class ImpostorBaselineBase(ImpostorBase):
    """
    Base class for impostor Baseline detectors.
    This class provides the basic structure for scoring and prediction methods.
    """

    def __init__(self, dataset_name: str = CONFIG.STUDENT_ESSAYS, in_args=None, not_in_args=None):
        super().__init__()
        # get all original texts from mongodb collection whose ID is not in test pairs mongodb collection
        self.dataset_name = dataset_name
        logger.info(f"About to use in_args: {in_args} and not_in_args: {not_in_args}")
        if in_args is None and not_in_args is None:
            self.train_dataset = pd.DataFrame(self.mongoDB.get_training_data_from_original_texts(
                dataset_name=self.dataset_name)
            )
            logger.info(f"Did not use in_args and not_in_args")

        else:
            self.train_dataset = pd.DataFrame(
                self.mongoDB.get_all_data_but_certain_from_original_texts(
                    in_args=in_args,
                    not_in_args=not_in_args
                )
            )
            logger.info(f"Used in_args: {in_args} and not_in_args: {not_in_args}")

        logger.info("Number of training pairs %d (in-memory)", self.train_dataset.shape[0])
        texts = pd.DataFrame(self.mongoDB.original_collection.find({"dataset_name":self.dataset_name},projection={
            '_id': False,"text":True}))
        
        tfidf_extractor = TfidfFeatureExtractor(top_n_freq_words=100000, ngram_n=4, min_df=2)

        self._vectorizer = tfidf_extractor.vectorizer

        self._vectorizer = self._vectorizer.fit(texts["text"])
        logger.info("Fitted vectorizer.")

    def get_tfidf_vector_for_text(self, texts: List[str]):
        """
        Get the TF-IDF vector for a list of texts.

        :param texts: The input texts to vectorize.
        :return: A TF-IDF vector as a NumPy array.
        """
        if not isinstance(texts, list):
            # pandas Series or generator
            texts = list(texts)
        if texts[0] == texts[1]:
            logger.warning("Texts are identical %s", texts[0])
        tfidf_matrix = self._vectorizer.transform(texts)  # format: (n_samples=1, n_features=self.top_n)
        n_zero = [np.count_nonzero(vector) == 0 for vector in tfidf_matrix.toarray()]
        assert sum(n_zero) == 0, f"Found all-zero tfidf vector, number of non-zero tfidf vector: {sum(n_zero)}"

        return tfidf_matrix.toarray()
