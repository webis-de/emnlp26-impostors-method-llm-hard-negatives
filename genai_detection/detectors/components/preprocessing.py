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
import logging
import re
from random import sample, choices
from typing import List, Iterable

from bson import ObjectId
from more_itertools import ichunked
from nltk import SnowballStemmer

from genai_detection.detectors.detector_base import DetectorBase

logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def bootstrap_tokens(tokens: List[str], n_tokens: int = 500):
        """
        Samples `n_tokens` from the input token sequence using bootstrapping. If the desired number of tokens
        exceeds the size of the input sequence, sampling continues with replacement. This strategy reflects
        the procedure outlined in Bevendorff et al. (2019).

        :param tokens: Sequence of tokens
        :param n_tokens: Number of tokens to sample from input sequence, defaults to 500
        :return: List of sampled tokens

        References:
        ===========
        Janek Bevendorff, Benno Stein, Matthias Hagen, and Martin Potthast. 2019. Generalizing Unmasking for Short Texts. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 654–659, Minneapolis, Minnesota. Association for Computational Linguistics.
        """
        if not tokens:
            raise ValueError("Cannot bootstrap tokens from an empty sequence.")
        tokens = list(tokens)  # mutable copy
        sampled = sample(tokens, min(n_tokens, len(tokens)))  # without replacement
        remaining = max(0, n_tokens - len(tokens))
        sampled.extend(choices(tokens, k=remaining))  # with replacement

        return sampled

    @staticmethod
    def tokenize_whitespace(text: str, normalize_ws: bool = True):
        """
        Tokenize input text by any whitespace character (including \n \r \t \f and spaces).
        Kocher et al. (2015) use isolated words without stemming but with punctuation symbols.

        References:
        ===========
        Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.

        :param text: Input text
        :param normalize_ws: Collapse whitespace before tokenization
        :return: List of tokens
        """
        if normalize_ws:
            text = re.sub(r"\s+", " ", text)
        return text.split()

    @staticmethod
    def normalize_text(text: str):
        """
        Normalize input text by lowercasing and stemming.
        Koppel et al. (2014) do (explicitly) not normalize text pairs, but without normalization, the results are terrible.
        Kocher et al. (2015) use isolated words without stemming but with punctuation symbols.
        """
        stemmer = SnowballStemmer("english")
        return " ".join(stemmer.stem(w) for w in text.lower().split())

    def upsample_to_min_n_tokens(self, text: str, min_n_tokens: int, upsample: bool):
        whitespace_tokens = self.tokenize_whitespace(text)
        if (len(whitespace_tokens) < min_n_tokens) and upsample:
            return " ".join(
                self.bootstrap_tokens(whitespace_tokens, n_tokens=min_n_tokens)
            )
        return text

    def obtain_text_and_id(self, value):
        try:
            ObjectId(value)
            return self.mongoDB.get_text_or_id_from_orginal_collection(
                text_id=value, text=None
            )
        except Exception as e:
            return self.mongoDB.get_text_or_id_from_orginal_collection(
                text=value, text_id=None
            )


class PairPreprocessor:
    def __init__(
        self, mongoDB, tokenizer, min_n_tokens: int = 500, upsample: bool = False
    ):
        self.mongoDB = mongoDB
        self.text_preprocessor = Preprocessor()
        self.tokenizer = tokenizer
        self.min_n_tokens = min_n_tokens
        self.upsample = upsample
        self.detector_base = DetectorBase()

    # -----------------------------------------------------------
    # 1. Validate & turn input into iterable
    # -----------------------------------------------------------
    def turn_input_iterable(self, text):
        assert isinstance(
            text, Iterable
        ), f"Input text must be iterable. But got: {type(text)}"
        return list(text)

    # -----------------------------------------------------------
    # 2. Convert IDs to text or other way around
    # -----------------------------------------------------------
    def obtain_texts_and_idx_from_pair(self, texts):
        texts = list(texts)
        assert len(texts) == 2, "Input must contain exactly two text entries."

        def resolve(value: str):
            try:
                ObjectId(value)
                return self.mongoDB.get_text_or_id_from_orginal_collection(
                    text_id=ObjectId(value), text=None
                )
            except Exception as e:
                return self.mongoDB.get_text_or_id_from_orginal_collection(
                    text=value, text_id=None
                )

        left_text, left_id = resolve(texts[0])
        right_text, right_id = resolve(texts[1])

        return left_text, right_text, left_id, right_id

    # -----------------------------------------------------------
    # 3. Validate lengths and optionally upsample
    # -----------------------------------------------------------
    def ensure_min_lengths(self, text_left: str, text_right: str) -> bool:
        """
        Indicates that text pair should be skipped if shorter than certain threshold.
        :param text_left: String which is left input text.
        :param text_right: String which is right input text.
        :return: Indication whether input should be skipped.
        """
        w_left = self.text_preprocessor.tokenize_whitespace(text_left)
        w_right = self.text_preprocessor.tokenize_whitespace(text_right)

        # Case: too short AND upsample disabled → skip
        return (
            len(w_left) < self.min_n_tokens or len(w_right) < self.min_n_tokens
        ) and not self.upsample

    # -----------------------------------------------------------
    # 4. Preprocess + tokenize
    # -----------------------------------------------------------
    def _preprocess_and_tokenize_single(self, text: str):
        return self.tokenizer(self.detector_base.preprocess_text(text=text))

    def preprocess_and_tokenize(self, left: str, right: str):
        left_tokens = self._preprocess_and_tokenize_single(left)
        right_tokens = self._preprocess_and_tokenize_single(right)
        return left_tokens, right_tokens

    # -----------------------------------------------------------
    # 5. Make left/right have same token length
    # -----------------------------------------------------------
    def match_lengths(self, left_toks, right_toks):
        max_allowed = min(len(left_toks), len(right_toks))
        if len(left_toks) > max_allowed:
            left_toks[:] = left_toks[:max_allowed]
        if len(right_toks) > max_allowed:
            right_toks[:] = right_toks[:max_allowed]
        return left_toks, right_toks

    # -----------------------------------------------------------
    # PUBLIC MAIN ENTRY POINT
    # -----------------------------------------------------------
    def filter_pairs(self, text_list: Iterable[str]):
        logger.info(f"Filtering {len(text_list)} texts.")
        text_list = self.turn_input_iterable(text_list)

        filtered = []
        for t in ichunked(text_list, 2):
            original_left, original_right, id_left, id_right = (
                self.obtain_texts_and_idx_from_pair(t)
            )

            # upsample if set to True (i.e., different to the original texts)
            skip = self.ensure_min_lengths(original_left, original_right)
            if skip:
                logger.error(
                    f"Skipping texts {id_left} and {id_right}, because they are not long enough."
                )
                continue

            # left, right are of at least minimal required length
            filtered.append(
                {
                    "left": {"id": id_left, "original_text": original_left},
                    "right": {"id": id_right, "original_text": original_right},
                }
            )

        return filtered
