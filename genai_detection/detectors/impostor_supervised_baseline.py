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
import pickle
import typing as t
from pathlib import Path

import numpy as np
import torch
from more_itertools import ichunked
from sklearn.svm import LinearSVC

from genai_detection.config import CONFIG
from genai_detection.detectors.impostor_base import ImpostorBaselineBase

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )

class SupervisedImpostorBaseline(ImpostorBaselineBase):
    """
    Supervised Impostor Baseline detector class.

    This class extends the DetectorBase and implements a supervised baseline for the Impostor method by Koopel et al. (2014).
    A document pair (i.e. disputed document and candidate author document) is represented as a vector of TF-IDF features.
    The frequenies are calculated based on the 100,000 most frequent space-free character 4-grams in the corpus.
    The representation of a text pair is the element-wise difference of the two TF-IDF vectors.
    Classification is carried out using a trained model, like a linear SVM.
    Koppel et al. (2014) have achieved (with a linear SVM) a maximum of an accuracy of 79.8% test set.

    References:
    ===========
    Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
    Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.
    """

    def __init__(self, dataset_name: str = CONFIG.STUDENT_ESSAYS):
        """
        Initialize the Supervised Impostor Baseline detector.
        """
        super().__init__(dataset_name=dataset_name)
        self.model = self.get_trained_linear_svc()

    def _get_score_impl(
        self, text: t.Iterable[str]
    ) -> t.Union[torch.Tensor, np.ndarray, t.Iterable[float]]:
        """
        Scoring implementation for the Supervised Impostor Baseline detector.

        :param text: An iterable of strings (texts) to score.
        :return: A list of scores for each text.
        """
        if isinstance(text, str):
            text = [text]

        scores_per_pair = (
            []
        )  # id is the index of the pair (i.e., length is half of the input text list)
        pairs = list(ichunked(text, 2)) if type(text[0]) == str else text
        for text_pair in pairs:
            vectors = [self.get_tfidf_vector_for_text(t) for t in text_pair]
            assert len(vectors) == 2, "Input text must be a list of pairs of texts."
            # get score for positive class: https://scikit-learn.org/stable/modules/svm.html#classification (06.08.2025)
            scores_per_pair.append(
                self.model.decision_function(abs(vectors[0] - vectors[1]))
            )
        return np.array(scores_per_pair)

    def get_prediction(self, text: t.Iterable[str]) -> t.List[bool]:
        """
        Predict if the input text(s) were written by a the same author.

        :param text: input text or batch of input texts
        :return: boolean classifications of whether inputs are likely same author.
        """
        if isinstance(text, str):
            text = [text]

        scores_per_pair = (
            []
        )  # id is index of pair (i.e, length is half of the input text list)
        pairs = list(ichunked(text, 2)) if type(text[0]) == str else text
        for text_pair in pairs:
            vectors = [self.get_tfidf_vector_for_text(t) for t in text_pair]
            assert len(vectors) == 2, "Input text must be a list of pairs of texts."
            scores_per_pair.append(self.model.predict(abs(vectors[0] - vectors[1])))
        return np.array(scores_per_pair)

    def get_trained_linear_svc(self, save_model: bool = False) -> LinearSVC:
        """
        Load a pre-trained LinearSVC model for the Supervised Impostor Baseline detector.
        If there is no pretrained model, train one.
        :param save_model: whether to save the trained model (given we do not want data leakage, this should be False to ensure trained is not trained on test data)
        :return: trained LinearSVC model
        """
        path2model = (
            Path(__file__).resolve().parents[2] / "models" / f"impostor_{self.dataset_name}_svc_model.pkl"
        )
        path2model.parent.mkdir(parents=True, exist_ok=True)
        if path2model.exists():
            logging.info(f"Loading trained LinearSVC model from {path2model}")
            return pickle.load(path2model)
        else:
            model = LinearSVC()
            self.train_dataset = (
                self.mongoDB.get_training_data_from_original_texts(
                    dataset_name=self.dataset_name
                )
            )
            disputed_texts = self.train_dataset["left_text"]
            candidate_texts = self.train_dataset["right_text"]
            disputed_vectors = self._vectorizer.transform(
                [" ".join(self.tokenize_char_ngrams(t)) for t in disputed_texts]
            )
            candidate_vectors = self._vectorizer.transform(
                [" ".join(self.tokenize_char_ngrams(t)) for t in candidate_texts]
            )
            # Calculate element-wise difference
            X = abs(disputed_vectors - candidate_vectors)
            # y = train_dataset["same"].astype(int).values
            y = np.array(self.train_dataset["same"], dtype=int)
            # Train model
            model.fit(X, y)
            if save_model:
                # pickle.dump(model, open(path2model, "wb"))
                with open(path2model, "wb") as f:
                    pickle.dump(model, f)
                    logging.info(f"Saved trained LinearSVC model to {path2model}")
            return model
