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

    def __init__(self, dataset_name: str = CONFIG.STUDENT_ESSAYS, left_input=None, right_input=None,
                 additional_in_args=None):
        """
        Initialize the Supervised Impostor Baseline detector.
        """
        if additional_in_args is None:
            additional_in_args = []
        if left_input is None and right_input is None:
            in_args, not_in_args = None, None
        else:
            left_id = left_input["left_id"]
            right_id = right_input["right_id"]
            in_args = {
                "dataset_name": dataset_name,
            }
            if additional_in_args:
                in_args.update({f"left_{arg}": left_input[f"left_{arg}"] for arg in additional_in_args})
                in_args.update({f"right_{arg}": right_input[f"right_{arg}"] for arg in additional_in_args})
            not_in_args = {
                "left_id": [left_id, right_id],
                "right_id": [left_id, right_id],
            }
        super().__init__(dataset_name=dataset_name, in_args=in_args, not_in_args=not_in_args)
        self.model = self.get_trained_linear_svc()

    def _process_pairs_with_func(
        self, text: t.Iterable[str], score_fn: t.Callable[[np.ndarray], t.Any]
    ) -> t.List[t.Any]:
        """
        Process text pairs and apply a scoring function to the difference of TF-IDF vectors.

        :param text: Iterable of strings or iterable of pairs of strings
        :param score_fn: Function to compute score from vector difference
        :return: List of results from score_fn
        """
        if isinstance(text, str):
            raise ValueError("Expected iterable of texts, got a single string")

        if (
            isinstance(text, (list, tuple))
            and len(text) == 2
            and all(isinstance(t, str) for t in text)
        ):
            pairs = [text]  # already one pair
        else:
            pairs = list(ichunked(text, 2))

        results = []
        for chunk in pairs:
            text_pair = list(chunk)
            # print("textst", text_pair)
            vectors = self.get_tfidf_vector_for_text(text_pair)
            # print(vectors.shape)
            # print("Non-zeros text 0:", np.count_nonzero(vectors[0]))
            # print("Non-zeros text 1:", np.count_nonzero(vectors[1]))

            # print("TFIDF shape of pair: %s", vectors.shape)
            assert len(vectors) == 2, "Each pair must contain exactly two texts."

            diff = abs(vectors[0] - vectors[1])
            diff = diff.reshape(1, -1)  # shape: (1, n_features)
            results.append(score_fn(diff))

        return results

    def _get_score_impl(
        self, text: t.Iterable[str]
    ) -> t.Union[torch.Tensor, np.ndarray, t.Iterable[float]]:
        """
        Scoring implementation for the Supervised Impostor Baseline detector.

        :param text: An iterable of strings (texts) to score.
        :return: A list of scores for each text.
        """
        scores = self._process_pairs_with_func(text, self.model.decision_function)
        return np.array(scores)

    def get_prediction(self, text: t.Iterable[str]) -> t.List[bool]:
        """
        Predict if the input text(s) were written by the same author.

        :param text: input text or batch of input texts
        :return: boolean classifications of whether inputs are likely same author.
        """
        return self._process_pairs_with_func(text, self.model.predict)

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
            print(f"Loading trained LinearSVC model from {path2model}")
            return pickle.load(path2model)
        else:
            # avoid OOM by subsampling (# positive = # negative samples) training data if too large
            if self.train_dataset.shape[0] > 5000:
                labels = np.array(self.train_dataset["same"], dtype=bool)
                pos_idx = np.flatnonzero(labels)
                neg_idx = np.flatnonzero(~labels)
                if pos_idx.size == 0 or neg_idx.size == 0:
                    raise ValueError("y contains only True or only False values.")
                target_per_class = min(pos_idx.size, neg_idx.size, 2500)
                rng = np.random.default_rng(0)
                pos_sample = rng.choice(pos_idx, size=target_per_class, replace=False)
                neg_sample = rng.choice(neg_idx, size=target_per_class, replace=False)
                sample_idx = np.concatenate((pos_sample, neg_sample))
                self.train_dataset = self.train_dataset.iloc[sample_idx]
            model = LinearSVC()
            disputed_texts = self.train_dataset["left_text"]
            candidate_texts = self.train_dataset["right_text"]
            logger.info(f"Training {len(candidate_texts)} candidate texts")
            disputed_vectors = self.get_tfidf_vector_for_text(texts=disputed_texts)
            candidate_vectors = self.get_tfidf_vector_for_text(texts=candidate_texts)

            # Calculate element-wise difference
            X = np.abs(disputed_vectors - candidate_vectors)
            # y = train_dataset["same"].astype(int).values
            y = np.array(self.train_dataset["same"], dtype=int)
            if np.all(y == 0) or np.all(y == 1):
                raise ValueError("y contains only True or only False values.")
            logger.info(f"Training {len(y)} labels")
            # Train model
            model.fit(X, y)
            logger.info(f"Saving trained LinearSVC model to {path2model}")
            if save_model:
                # pickle.dump(model, open(path2model, "wb"))
                with open(path2model, "wb") as f:
                    pickle.dump(model, f)
                    logger.info(f"Saved trained LinearSVC model to {path2model}")
            return model
