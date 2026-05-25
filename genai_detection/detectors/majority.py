# Copyright 2026 Klara M. Gutekunst, Webis
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
import sys
import typing as t
from pathlib import Path
from typing import Iterable, List

import nltk
import numpy as np
import numpy.typing as npt
import torch

from genai_detection.detectors.detector_base import DetectorBase
from genai_detection.detectors.impostor import ImpostorDetector
from genai_detection.detectors.ppmd import PPMdDetector
from genai_detection.paraphrasing.one_step_paraphrasers import T5ChatGPTParaphraser

nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

__all__ = ["ImpostorDetector", "PPMdDetector"]


class MajorityDetector(DetectorBase):
    def __init__(self, Detector):
        super().__init__()
        self.detector = Detector()
        self.paraphraser = T5ChatGPTParaphraser()

    def _split_text_into_chunks(self, text: str, n: int = 4) -> List[str]:
        """
        Split text into <=n chunks, each containing approximately the same number of words.
        The text is first tokenized into sentences, then grouped into chunks until the total word count
        is approximately even across the n parts.
        :param text: Input text to be split into chunks.
        :param n: Number of chunks to split the text into. If n is greater than the number of sentences, each sentence will be a chunk.
        :return: List of text chunks.
        """
        # Step 1: Tokenize into sentences
        sentences = sent_tokenize(text)

        # Step 2: Group sentences until total word count is ~ even across n parts
        total_words = sum(len(word_tokenize(sent)) for sent in sentences)
        target_words_per_chunk = total_words / n

        chunks = []
        current_chunk = []
        current_word_count = 0

        for sent in sentences:
            sent_words = word_tokenize(sent)
            current_chunk.append(sent)
            current_word_count += len(sent_words)

            if current_word_count >= target_words_per_chunk and len(chunks) < n - 1:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

        # Append remaining sentences to the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _get_score_impl(
        self, text: Iterable[str]
    ) -> t.Union[torch.Tensor, np.ndarray, t.Iterable[float]]:
        """
        Calculate the majority score for the given text(s).
        The score per text is the average of the scores from the detector for each chunk of text.

        :param text: input text(s) to score, if list we batch of texts.
        :return: Majority score for the input text(s).
        """
        if isinstance(text, str):
            text = [text]
        chunks = [self._split_text_into_chunks(t) for t in text]
        score_per_text = []
        for text_chunks in chunks:
            if isinstance(self.detector, (ImpostorDetector, PPMdDetector)):
                # generate artificial paraphrase candidates
                prompt = (
                    "Paraphrase the text above and output only the paraphrased version."
                )
                paraphrases = [
                    self.paraphraser.paraphrase(text=t, prompt=prompt)[0]
                    for t in text_chunks
                ]
                inputs = [
                    [original, candidate]
                    for original, candidate in zip(text_chunks, paraphrases)
                ]
            else:
                inputs = text_chunks

            scores = [self.detector.get_score(c, normalize=False) for c in inputs]
            score_per_text.append(np.mean(scores) if scores else 0.0)

        return score_per_text

    def get_score(
        self, text: t.Union[str, t.Iterable[str]], normalize: bool = False
    ) -> t.Union[np.float32, np.ndarray, npt.NDArray[np.float32]]:
        """
        Return scores indicating the probability of the input text(s) being machine-generated.

        :param text: Input text or iterable of texts.
        :param normalize: Whether to normalize the scores.
        :return: Majority score as a float or numpy array.
        """
        return np.array(self._get_score_impl(text), dtype=np.float32)

    def _predict_impl(
        self, text: t.Iterable[str], threshold: float = 0.5
    ) -> t.Union[torch.Tensor, np.ndarray, t.Iterable[bool]]:
        """
        Predict if the input text(s) were written by a machine.
        The prediction is based on the majority score.

        :param text: Input text or iterable of texts.
        :return: Boolean predictions indicating whether the input is likely machine-generated.
        """
        scores = self._get_score_impl(text)
        return [score > threshold for score in scores]


if __name__ == "__main__":
    # Example usage
    detector = MajorityDetector(Detector=PPMdDetector)  # ImpostorDetector)
    path2datasets = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "datasets"
        / "custom_texts"
    )
    file_names = [
        "cnn_230625",  # USA attacks Iran
        "cnn_040725",  # Dalai Lama
    ]
    original_texts = [
        open(path2datasets / f"{file_name}.txt").read() for file_name in file_names
    ]
    scores = detector.get_score(original_texts, normalize=True)
    predictions = detector.predict(original_texts)

    print("Scores:", scores)
    print("Predictions:", predictions)
