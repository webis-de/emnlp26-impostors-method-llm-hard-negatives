from more_itertools import ichunked
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from genai_detection.config import CONFIG
from genai_detection.detectors.impostor_base import ImpostorBaselineBase
import typing as t


class UnSupervisedImpostorBaseline(ImpostorBaselineBase):
    """
    Unsupervised Impostor Baseline detector class.

    This class extends the DetectorBase and implements an unsupervised baseline for the Impostor method by Koopel et al. (2014).
    A document pair (i.e. disputed document and candidate author document) is represented as a vector of TF-IDF features.
    The frequenies are calculated based on the 100,000 most frequent space-free character 4-grams in the corpus.
    Similarities are than calculated using cosine similarity or min-max similarity.
    Classification is carried out using a threshold on the similarity score.
    Koppel et al. (2014) have achieved (with the best threshold) a maximum of an accuracy of 70.6% using cosine similarity
    and a maximum an accuracy of 74.2% using min-max simialrity on a deployment set.

    References:
    ===========
    Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
    Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.
    """

    def __init__(
        self,
        use_cosine_simiarity: bool = True,
        dataset_name: str = CONFIG.STUDENT_ESSAYS,
    ):
        """
        Initialize the Unsupervised Impostor Baseline detector.
        """
        super().__init__(dataset_name=dataset_name)
        self.threshold = (
            0.5  # Default threshold, can be adjusted based on validation set
        )
        self.cosine = use_cosine_simiarity

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
            if self.cosine:
                scores_per_pair.append(self.cosine_similarity(vectors[0], vectors[1]))
            else:
                scores_per_pair.append(self.minmax_similarity(vectors[0], vectors[1]))

        return np.array(scores_per_pair)

    def get_prediction(self, text: t.Iterable[str]) -> t.List[bool]:
        """
        Predict if the input text(s) were written by a the same author TODO: machine.

        :param text: input text or batch of input texts
        :return: boolean classifications of whether inputs are likely same author TODO: machine-generated
        """
        scores = self.get_score(text)
        return [score > self.threshold for score in scores]


if __name__ == "__main__":
    # Example usage
    detector = UnSupervisedImpostorBaseline()
    sample_texts = [
        "This is a sample text for testing.",
        "This is another sample text for testing.",
        "This text is different from the others.",
        "Yet another text to test the detector.",
    ]
    scores = detector.get_score(sample_texts)
    predictions = detector.get_prediction(sample_texts)
    print("Scores:", scores)
    print("Predictions:", predictions)
