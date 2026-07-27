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
import re
from typing import List, Optional

import numpy as np

from genai_detection.impostor_generators.ImpostorGenerator import BaseImpostorGenerator


class RandomWordImpostorGenerator(BaseImpostorGenerator):
    """
    Generate non-semantic impostors by sampling English words from an empirical
    unigram frequency distribution.

    The distribution is intentionally content-free: it follows corpus-informed
    English word frequencies via wordfreq, but does not model syntax, coherence,
    topic, or author-specific style. This is useful as an ablation for testing
    whether the impostor method benefits from text existence/coherence or merely
    from plausible marginal word frequencies.

    References:
    ===========
    Zipf, George Kingsley. Human Behavior and the Principle of Least Effort. 1949.

    Speer, Robyn, Joshua Chin, and Catherine Havasi. ConceptNet 5.5:
    An Open Multilingual Graph of General Knowledge. AAAI, 2017.
    wordfreq uses language-specific word frequencies derived from multiple
    corpora and resources.
    """

    def __init__(
        self,
        n_impostors: int,
        top_n_freq_words: int,
        vocabulary_size: int = 50000,
        distribution_temperature: float = 1.0,
        sentence_mean_tokens: int = 22,
        random_seed: Optional[int] = None,
    ):
        super().__init__(n_impostors=n_impostors, top_n_freq_words=top_n_freq_words)
        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least 2.")
        if distribution_temperature <= 0:
            raise ValueError("distribution_temperature must be positive.")
        if sentence_mean_tokens < 1:
            raise ValueError("sentence_mean_tokens must be at least 1.")

        self.vocabulary_size = vocabulary_size
        self.distribution_temperature = distribution_temperature
        self.sentence_mean_tokens = sentence_mean_tokens
        self.rng = np.random.default_rng(random_seed)
        self._vocabulary = None
        self._probabilities = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z']+", text)

    def _load_word_distribution(self):
        if self._vocabulary is not None and self._probabilities is not None:
            return self._vocabulary, self._probabilities

        try:
            from wordfreq import top_n_list, word_frequency
        except ImportError as exc:
            raise ImportError(
                "RandomWordImpostorGenerator requires the optional dependency "
                "'wordfreq'. Install project dependencies with Poetry before "
                "using impostor_technique='random_words'."
            ) from exc

        candidates = top_n_list("en", self.vocabulary_size)
        vocabulary = [
            word for word in candidates if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", word)
        ]
        if len(vocabulary) < 2:
            raise ValueError(
                "wordfreq did not return enough English alphabetic tokens for sampling."
            )

        probabilities = np.array(
            [word_frequency(word, "en", minimum=0.0) for word in vocabulary],
            dtype=float,
        )
        probabilities = np.power(probabilities, 1.0 / self.distribution_temperature)
        probabilities_sum = probabilities.sum()
        if probabilities_sum <= 0:
            raise ValueError("wordfreq returned an all-zero English word distribution.")

        self._vocabulary = np.array(vocabulary, dtype=object)
        self._probabilities = probabilities / probabilities_sum
        return self._vocabulary, self._probabilities

    def _sample_impostor(self, n_tokens: int) -> str:
        vocabulary, probabilities = self._load_word_distribution()
        words = self.rng.choice(
            vocabulary, size=n_tokens, replace=True, p=probabilities
        )
        return self._format_as_pseudo_document(words.tolist())

    def _format_as_pseudo_document(self, words: List[str]) -> str:
        sentences = []
        pos = 0
        while pos < len(words):
            sentence_len = max(1, int(self.rng.poisson(self.sentence_mean_tokens)))
            sentence_words = words[pos : pos + sentence_len]
            if not sentence_words:
                break
            sentence = " ".join(sentence_words)
            sentences.append(sentence[:1].upper() + sentence[1:] + ".")
            pos += sentence_len
        return " ".join(sentences)

    def generate_impostors(self, text: Optional[str], text_id=None) -> List[str]:
        if isinstance(text, tuple):
            text = text[0]
        if text is None:
            raise ValueError("RandomWordImpostorGenerator requires input text.")

        n_tokens = len(self._tokenize(text))
        if n_tokens == 0:
            raise ValueError(
                "RandomWordImpostorGenerator requires non-empty input text."
            )

        return [self._sample_impostor(n_tokens) for _ in range(self.n_impostors)]
