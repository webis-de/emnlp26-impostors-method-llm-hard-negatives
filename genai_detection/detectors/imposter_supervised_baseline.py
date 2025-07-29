import os
from pathlib import Path
from more_itertools import ichunked
import numpy as np
from datasets import load_from_disk
import pandas as pd

import torch
from genai_detection.config import CONFIG
from genai_detection.detectors.imposter_base import ImposterBaselineBase
import typing as t
from sklearn.svm import LinearSVC
import pickle


class SupervisedImposterBaseline(ImposterBaselineBase):
    """
    Supervised Imposter Baseline detector class.

    This class extends the DetectorBase and implements a supervised baseline for the Imposter method by Koopel et. Al. (2014).
    A document pair (i.e. disputed document and candidate author document) is represented as a vector of TF-IDF features.
    The frequenies are calculated based on the 100,000 most frequent space-free character 4-grams in the corpus.
    The representation of a text pair is the element-wise difference of the two TF-IDF vectors.
    Classification is carried out using a trained model, like a linear SVM.
    Koppel et Al. (2014) have achieved (with a linear SVM) a maximum of an accuracy of 79.8% test set.

    References:
    ===========
    Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
    Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.
    """

    def __init__(self):
        """
        Initialize the Supervised Imposter Baseline detector.
        """
        super().__init__()
        self.model = self.get_trained_linear_svc()

    def _get_score_impl(
        self, text: t.Iterable[str]
    ) -> t.Union[torch.Tensor, np.ndarray, t.Iterable[float]]:
        """
        Scoring implementation for the Supervised Imposter Baseline detector.

        :param text: An iterable of strings (texts) to score.
        :return: A list of scores for each text.
        """
        if isinstance(text, str):
            text = [text]

        vectors = [self.get_tfidf_vector_for_text(t) for t in text]
        scores_per_pair = (
            []
        )  # id is index of pair (i.e, length is half of the input text list)
        for i, vecs in enumerate(ichunked(vectors, 2)):
            vecs = list(
                vecs
            )  # generator object is not subscriptable, so convert to list
            assert len(vecs) == 2, "Input text must be a list of pairs of texts."
            scores_per_pair.append(self.model.predict(abs(vecs[0] - vecs[1])))

        return np.array(scores_per_pair)

    def get_prediction(self, text: t.Iterable[str]) -> t.List[bool]:
        """
        Predict if the input text(s) were written by a the same author TODO: machine.

        :param text: input text or batch of input texts
        :return: boolean classifications of whether inputs are likely same author TODO: machine-generated
        """
        scores = self.get_score(text)
        return scores

    def get_trained_linear_svc(self):
        """
        Load a pre-trained LinearSVC model for the Supervised Imposter Baseline detector.
        If there is no pretrained model, train one.
        """
        path2model = (
            Path(__file__).resolve().parents[2] / "models" / "imposter_svc_model.pkl"
        )
        if path2model.exists():
            return pickle.load(path2model)
        else:
            os.mkdir(path2model.parent, exist_ok=True)
            model = LinearSVC()
            # Load training data
            dataset = load_from_disk(
                Path(__file__).resolve().parents[2] / CONFIG.CROSS_GENRE
            )["train"].to_pandas()
            disputed_texts = dataset["disputed_text", "candidate_text", "same"]
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
            # Save model
            pickle.dump(model, open(path2model, "wb"))
            return model
