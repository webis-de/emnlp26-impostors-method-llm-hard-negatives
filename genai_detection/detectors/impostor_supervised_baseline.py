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

import os
from pathlib import Path
from more_itertools import ichunked
import numpy as np
from datasets import load_from_disk
import pandas as pd

import torch
from genai_detection.config import CONFIG
from genai_detection.detectors.impostor_base import ImpostorBaselineBase
import typing as t
from sklearn.svm import LinearSVC
import pickle


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
        )  # id is index of pair (i.e, length is half of the input text list)
        pairs = list(ichunked(vectors, 2)) if type(text[0]) == str else text
        for text_pair in pairs:
            vectors = [self.get_tfidf_vector_for_text(t) for t in text_pair]
            assert len(vectors) == 2, "Input text must be a list of pairs of texts."
            # get score for potive class: https://scikit-learn.org/stable/modules/svm.html#classification (06.08.2025)
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
            Path(__file__).resolve().parents[2] / "models" / "impostor_svc_model.pkl"
        )
        if path2model.exists():
            return pickle.load(path2model)
        else:
            path2model.parent.mkdir(parents=True, exist_ok=True)
            model = LinearSVC()
            # Load training data
            disputed_texts = self.dataset[["disputed_text", "candidate_text", "same"]]
            # sample texts
            frac = 0.6
            training_data = disputed_texts[disputed_texts["same"]].sample(
                frac=frac, random_state=42
            )
            training_data = pd.concat(
                [
                    training_data,
                    disputed_texts[~disputed_texts["same"]].sample(
                        frac=frac, random_state=42
                    ),
                ]
            )
            # Vectorize texts
            disputed_texts = training_data["disputed_text"].tolist()
            candidate_texts = training_data["candidate_text"].tolist()
            disputed_vectors = self._vectorizer.transform(
                [" ".join(self.tokenize_char_ngrams(t)) for t in disputed_texts]
            )
            candidate_vectors = self._vectorizer.transform(
                [" ".join(self.tokenize_char_ngrams(t)) for t in candidate_texts]
            )
            # Calculate element-wise difference
            X = abs(disputed_vectors - candidate_vectors)
            y = training_data["same"].astype(int).values
            # Train model
            model.fit(X, y)
            if save_model:
                pickle.dump(model, open(path2model, "wb"))
            return model


if __name__ == "__main__":
    # Example usage
    detector = SupervisedImpostorBaseline()
    dataset = load_from_disk(
        Path(__file__).resolve().parents[2] / CONFIG.PATH2CROSS_GENRE
    )["test"].to_pandas()[["disputed_text", "candidate_text", "same"]]
    print("TWCHCVH", dataset.iloc[0])
    sample_texts = [
        [dataset.iloc[i]["disputed_text"], dataset.iloc[i]["candidate_text"]]
        for i in range(len(dataset))
    ]
    print("Number of texts: ", len(sample_texts))
    sample_texts = [item for sublist in sample_texts for item in sublist]
    scores = detector.get_score(sample_texts)
    predictions = detector.get_prediction(sample_texts)
    print("Scores:", scores)
    print("Predictions:", predictions)
    print("Ground Truth: ", dataset["same"])

    # debug
    dataset = load_from_disk(
        Path(__file__).resolve().parents[2] / CONFIG.PATH2CROSS_GENRE
    )["train"].to_pandas()[["disputed_text", "candidate_text", "same"]]
    print(
        "Aggregated number of same/different predictions:",
        dataset["same"].value_counts(),
    )
