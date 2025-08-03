from collections import Counter, defaultdict
from more_itertools import ichunked
import itertools
import json
import os
from pathlib import Path
from random import choices, sample
import re
import sys
from typing import Iterable, List, Literal
import heapq
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import typing as t
from collections import Counter

import numpy as np
from nltk import ngrams
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from genai_detection.detectors.impostor_base import ImpostorBase
from genai_detection.impostor_generators import ImpostorGenerator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from genai_detection.config import CONFIG

__all__ = ["ImpostorDetector"]


class ImpostorDetector(ImpostorBase):
    """
    The Impostor method extends the ngram-unmasking method.
    It uses saves the most similar author to the disputed text for each of multiple random feature selection rounds,
    where the disputed text is compared not only to the candidate text, but alos to a set of impostor texts.
    The final prediction is made based on of how often an author is predicted after each feature-elimination step.

    The input is a list of texts where text ``i`` and text ``i+1`` belong to a pair.
    The output for one document pair is a score for the disputed text and the candidate text (i.e. author).

    References:
    ===========
    Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
    Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.

    Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.
    """

    def __init__(
        self,
        rounds=100,
        top_n=100000,
        portion_delete=0.5,
        tokenizer=None,
        shared_vocab_only=True,
        tfidf_freqs=True,
        n_impostors=25,
        threshold=0.1,
        impostor_technique: Literal[
            "llm",
            "text_len",
            "on-the-fly",
            "blogs",
            "fixed",
            "content",
            "naive_llm",
            "non_naive_llm",
        ] = "llm",
        path2imp: str = CONFIG.PATH2BLOG,  # PATH2GENERIC_ON_FLY_IMP,  # path to impostor file, where fixed impostors are saved or where to save generated impostors
        real_time_generation: bool = False,  # whether to generate impostors in real-time or use pre-generated ones
        min_n_tokens: int = 500,  # minimum number of tokens to consider input sequence valid, defaults to 500
        upsample: bool = True,  # whether to upsample short texts (default: True, i.e. upsample) or skip them
    ):
        """
        :param rounds: number of random feature selection rounds, Koppel et al. (2014) use 100
        :param top_n: number of top space-free character 4-grams to consider, Koppel et al. (2014) use 100,000
        :param portion_delete: portion of features to eliminate in each round (reset in each round); Koppel et al. (2014) use 50% of features
        :param tokenizer: custom tokenizer function (must accept exactly one parameter, defaults to space-free character 4-grams cf. Koppel et al. (2014))
        :param shared_vocab_only: restrict analysis to shared vocabulary across pairs of texts (Koppel et al. (2014): all texts in the corpus, i.e. shared)
        :param tfidf_freqs: use tfidf term frequencies (Koppel et al. (2014) use tfidf)
        :param n_impostors: number of impostors to use for each candidate TODO: allow specification type of LLM impostors; Koppel et al. (2014) use 25 impostors
        :param threshold: threshold for the minimum similarity score to consider two texts same-author, TODO: not used yet, Koppel et al. (2014) use 0.1
        :param impostor_technique: which technique to use to generate impostors. Options are:
            - "llm": use LLMs to generate impostors to control both topic and genre
            - "naive_llm": use a naive LLM approach to generate impostors
            - "text_len": generate impostors of similar length from a predefined dataset (our baseline w/o reference, default)
            - "fixed": use a fixed set of impostors (Koppel et. A. (2014), not implemented yet), impostors are not related to the input text
            - "on-the-fly": generate same-topic impostors on-the-fly (Koppel et al. (2014), not implemented yet)
            - "blogs": use blogs to obtain same genre impostors (Koppel et al. (2014), not implemented yet)
        :param path2imp: path to the impostor file, where fixed impostors are saved or where to save generated impostors
        :param real_time_generation: whether to generate impostors in real-time or use pre-generated ones (default: False, i.e. use pre-generated impostors)
        :param min_n_tokens: minimum number of tokens to consider input sequence valid, defaults to 500 (Bevendorff et al. (2019): 500 words)
        :param upsample: whether to upsample short texts (default: True, i.e. upsample acc. to Bevendorff (2019)) or skip them (Bevendorff et al. (2019)/ Koppel et al (2014) at 500 words)
        """

        self.rounds = rounds
        self.top_n = top_n
        self.shared_vocab_only = shared_vocab_only
        self.portion_delete = portion_delete
        self.n_impostors = n_impostors
        self.tfidf_freqs = tfidf_freqs
        self.tokenizer = tokenizer or self.tokenize_char_ngrams
        self.threshold = threshold
        self.path2imp = path2imp
        self.real_time_generation = real_time_generation
        self.min_n_tokens = min_n_tokens
        self.upsample = upsample
        self.impostor_technique = impostor_technique
        self._training_mode = True  # set to True if you are in training mode, False for validation of model

        if impostor_technique == "llm":
            self.impostor_generator = ImpostorGenerator.LLMImpostorGenerator(
                n_impostors=self.n_impostors
            )
        elif impostor_technique == "naive_llm":
            self.impostor_generator = ImpostorGenerator.NaiveLLMImpostorGenerator(
                n_impostors=self.n_impostors
            )
        elif impostor_technique == "non_naive_llm":
            self.impostor_generator = ImpostorGenerator.NonNaiveLLMImpostorGenerator(
                n_impostors=self.n_impostors
            )
        elif impostor_technique == "fixed":
            self.impostor_generator = ImpostorGenerator.FixedImpostorGenerator(
                n_impostors=self.n_impostors,
                split="test" if self._training_mode else "train",
            )
        elif impostor_technique == "on-the-fly":
            self.impostor_generator = ImpostorGenerator.GoogleSearchImpostorGenerator(
                api_key=CONFIG.SERPAPI_KEY
            )
        elif impostor_technique == "blogs":
            self.impostor_generator = ImpostorGenerator.BlogImpostorGenerator(
                n_impostors=self.n_impostors,
                split="test" if self._training_mode else "train",
            )
        elif impostor_technique == "content":
            self.impostor_generator = ImpostorGenerator.ContentImpostorGenerator(
                n_impostors=self.n_impostors
            )
        else:
            self.impostor_generator = ImpostorGenerator.TextLenImpostorGenerator(
                n_impostors=self.n_impostors
            )

    def set_training_mode(self, training_mode: bool):
        """
        Set the training mode for the detector.
        If training_mode is True, the detector will use the test split of the impostor generator.
        If training_mode is False, the detector will use the train split of the impostor generator.
        This will reduce the risk of texts from the actual author among the impostors (i.e. actual positives among the hard negatives).
        :param training_mode: True if in training mode, False otherwise.
        """
        self._training_mode = training_mode
        if self.impostor_technique == "fixed":
            self.impostor_generator = ImpostorGenerator.FixedImpostorGenerator(
                n_impostors=self.n_impostors,
                split="test" if self._training_mode else "train",
            )
        elif self.impostor_technique == "blogs":
            self.impostor_generator = ImpostorGenerator.BlogImpostorGenerator(
                n_impostors=self.n_impostors,
                split="test" if self._training_mode else "train",
            )

    @staticmethod
    def bootstrap_tokens(tokens, n_tokens: int = 500):
        """
         Samples `n_tokens` from the input token sequence using bootstrapping. If the desired number of tokens
        exceeds the size of the input sequence, sampling continues with replacement. This strategy reflects
        the procedure outlined in Bevendorff et al. (2019).

        :param tokens: sequence of tokens
        :param n_tokens: number of tokens to sample from input sequence, defaults to 500
        :return: list of sampled tokens

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

    def _get_score_impl(
        self, text: Iterable[str]
    ) -> t.Union[torch.Tensor, np.ndarray, t.Iterable[float]]:
        """
        Called by get_score from detecor_base parent class to compute the score for the input text(s).

        Get scores for text pairs. A higher score indicates that the input text pair is more likely to be authored by the same author.
        The algorithm stems from Koppel et al. (2014)[, where some details are adapted from Kocher et al. (2015)]:
        Each text from the pair is the disputed text and the candidate text once.
        For the candidate text, a set of impostors is generated.
        For each round, a portion of features is randomly deleted, and the most similar candidate text is determined.
        The final score is the number of rounds where the candidate text was the most similar to the disputed text.
        The final score for a pair is the average of the scores for both directions (disputed text vs. candidate text and vice versa).

        While Koppel et al. (2014) use (1) a fixed set of impostor documents without realtion to document pair,
        (2) on-the-fly generated same content impostor via Google search,
        (3) Blogs to obtain same genre impostors, and Kocher et al. (2015) use (4) a set of impostor documents based on the number of documents written by the author,
        we define different techniques to generate impostors, which can be specified via the `technique` parameter in the `_get_impostors` method:
        We currently support:
        (1) `text_len`: generate impostors of similar length from a predefined dataset (default, see `_get_impostors` method).
        (2) `llm`: use LLMs to generate impostors, extension of Koppel et al. (2014).
        (3) `n_docs`: generate impostors based on the number of documents written by the author (TODO: not implemented yet), cf. Kocher et al. (2015).

        TODO: If the score is above a certain threshold, the input text is classified as same-author, which is not implemented yet/ not the purpose of this method.

        Koppel et al. (2014) exclude texts shorter than 500 words.
        Kocher et al. (2015) exclude words appearing only once to prevent overfitting to words occuring only once.
        Koppel et al. (2014) select m most similar impostors in terms of min-max similarity as impostor candidates and then,
        randomly select n actual impostors among potential impostors (because it has proven superior to using the top n impostors).
        They claim the approach is not sensitive to the choice of m and n.
        Koppel et al. (2014) compare using min-max and cosine simialrity.

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
        Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.

        Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.

        :param text: input text or batch of input texts
        :return: score indicating whether the input text is machine-generated, i.e. close 1 means machine-generated, close 0 means human-written
        """
        text = list(text)  # convert to tuple to list

        scores_per_pair = defaultdict(
            int
        )  # id is index of pair (i.e, length is half of the input text list)
        for i, t in enumerate(ichunked(text, 2)):
            t = list(t)  # generator object is not subscriptable, so convert to list
            assert len(t) == 2, "Input text must be a list of pairs of texts."
            text_left, text_right = t[0], t[1]
            len_ws_token_left = len(self.tokenize_whitespace(text_left))
            len_ws_token_right = len(self.tokenize_whitespace(text_right))

            # check text length, if too short, i.e. less than 500 `words` (acc. to Koppel et al. (2014) -> invalid; acc. to Bevendorff (2019) -> upsample)
            if (
                len_ws_token_left + len_ws_token_right < 2 * self.min_n_tokens
            ) and not self.upsample:  # skip
                print(
                    f"Skipping text pair: Left: {text_left}, Right: {text_right} (too short, {len_ws_token_left + len_ws_token_right} tokens < {2 * self.min_n_tokens})"
                )
                continue
            if len_ws_token_left == 0 or len_ws_token_right == 0:
                print(
                    f"Skipping empty text pair: Left: {text_left}, Right: {text_right}"
                )
                continue  # skip empty texts

            # upsample short texts to the minimum number of tokens
            if len_ws_token_left < self.min_n_tokens:
                text_left = " ".join(
                    self.bootstrap_tokens(
                        self.tokenize_whitespace(text_left), n_tokens=self.min_n_tokens
                    )
                )
            if len_ws_token_right < self.min_n_tokens:
                text_right = " ".join(
                    self.bootstrap_tokens(
                        self.tokenize_whitespace(text_right), n_tokens=self.min_n_tokens
                    )
                )

            # TODO: preprocessing: remove punctuation, lowercasing, remove html tags (e.g., <nl>), etc.?
            # Koppel et al. (2014) do not normalize text pairs, but without normalization, the results are terrible.
            # Does not make sense, bc 	idiosyncrasies of authors are not captured when using stemmed text.
            # Kontrolliere Situation

            # Koppel et al. (2014) use documents of length 500 words exactly -> we DON'T crop at min_n_tokens to keep more information
            # preprocess_text omits all layour/ structural information to keep only style
            tokens_left = self.tokenizer(self.preprocess_text(text_left))
            tokens_right = self.tokenizer(self.preprocess_text(text_right))

            # ensure both texts have same length (control confounder text length): min length of both
            max_len_allowed = min(len(tokens_left), len(tokens_right))
            for tokens in [tokens_left, tokens_right]:
                if len(tokens) > max_len_allowed:
                    tokens[:] = tokens[
                        :max_len_allowed
                    ]  # inplace crop, changes also other references to the same list

            if len(tokens_left) == 0 or len(tokens_right) == 0:
                print(
                    "Skipping empty text pair: Left: {}, Right: {}".format(
                        text_left, text_right
                    )
                )
                continue

            # frequencies as Counter (subclass of defaultdict(int))
            freqs_left = Counter(tokens_left)
            freqs_right = Counter(tokens_right)

            # Kocher et al. (2015) exclude words appearing only once
            freqs_left = Counter({k: v for k, v in freqs_left.items() if v > 1})
            freqs_right = Counter({k: v for k, v in freqs_right.items() if v > 1})

            if self.shared_vocab_only:  # TODO: over all corpus documents
                shared_tokens = freqs_left.keys() & freqs_right.keys()
            else:
                shared_tokens = freqs_left.keys() | freqs_right.keys()

            # TODO: Use complete corpus for Student Essays
            top_tokens = heapq.nlargest(
                self.top_n, shared_tokens, key=lambda x: freqs_left[x] + freqs_right[x]
            )

            # TODO: muss man TFIDF gemeinsam (left, right, impostors) berechnen, wegen Dataset Normalierung?
            x_left = self.tokens_to_matrix(tokens_left, top_tokens)
            x_right = self.tokens_to_matrix(tokens_right, top_tokens)

            store = {
                # TODO: if I knew >=1 author, i could choose impostors based on similar number of documents written (like paper)
                "left": {
                    "tfidf": x_left,
                    "tokens": tokens_left,
                    "text": text_left,
                    "author": "unknown",
                },
                "right": {
                    "tfidf": x_right,
                    "tokens": tokens_right,
                    "text": text_right,
                    "author": "unknown",
                },
            }

            # two iterations, generating impostors for each candidate once
            for j, (disputed, candidate) in enumerate(
                itertools.permutations(list(store.keys()), 2)
            ):
                scores_over_different_rounds = 0
                # get impostors for the candidate text, NOT the disputed text
                impostor_candidates = self.impostor_generator.generate_impostors(
                    store[candidate]["text"],
                    real_time_generation=self.real_time_generation,
                    path2imp=self.path2imp,
                )

                tmp_store = {
                    impostor_name: {
                        "tfidf": self.tokens_to_matrix(
                            self.tokenizer(impostor_text), top_tokens
                        ),
                        "text": impostor_text,
                        "tokens": self.tokenizer(impostor_text),
                    }
                    for impostor_name, impostor_text in impostor_candidates.items()
                }
                tmp_store[candidate] = store[candidate]  # add actual candidate

                # for different rounds, randomly delete a portion of features (reset in each round)
                for _ in range(self.rounds):
                    # feature selection: randomly delete a portion of features
                    rand_feat_to_keep_ids = sample(
                        range(len(top_tokens)),
                        int(len(top_tokens) * (1 - self.portion_delete)),
                    )
                    scores = {
                        c: self.minmax_similarity(
                            store[disputed]["tfidf"][
                                :, rand_feat_to_keep_ids
                            ],  # disputed text
                            tmp_store[c]["tfidf"][:, rand_feat_to_keep_ids],
                        )
                        for c in list(tmp_store.keys())
                    }
                    # increase score if the most similar candidate is the actual candidate
                    max_similar_candidate = max(scores, key=scores.get)
                    scores_over_different_rounds += max_similar_candidate == candidate
                # average after second loop
                scores_per_pair[i] += scores_over_different_rounds
                scores_per_pair[i] /= j + 1

        # one elmenent = averaged score of X,Y and Y,X pair (score=number of rounds where the candidate was the most similar)
        # TODO: threshold is in [0,1], maybe normalize by rounds?
        # return list(scores_per_pair.values())
        return [v / self.rounds for v in scores_per_pair.values()]

    def normalize_text(self, text):
        """
        Normalize input text by lowercasing and stemming.
        Koppel et al. (2014) do (explicitly) not normalize text pairs, but without normalization, the results are terrible.
        Kocher et al. (2015) use isolated words without stemming but with punctuation symbols.
        """
        stemmer = SnowballStemmer("english")
        return " ".join(stemmer.stem(w) for w in text.lower().split())

    def get_prediction(self, text: Iterable[str]) -> List[bool]:
        """
        Predict if the input text(s) were written by a the same author TODO: machine.

        :param text: input text or batch of input texts
        :return: boolean classifications of whether inputs are likely same author TODO: machine-generated
        """
        scores = self.get_score(text)
        return [score > self.threshold for score in scores]

    def tokens_to_matrix(self, tokens, top_token_list):
        """
        Transform list of tokens into matrix of term tfidf values of the top tokens.
        Koppel et al. (2014) use space-free character 4-grams tfidf values to represent each document as a numerical vector.

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.

        :param tokens: list of input tokens (e.g., space-free character 4-grams)
        :param top_token_list: list of top tokens to include in the matrix
        :return: Numpy array of term tfidf values, `shape = (len(tokens), len(top_token_list))`
        """
        if len(top_token_list) == 0:
            return np.zeros(
                (1, self.top_n), dtype=np.float32
            )  # return empty matrix if no top tokens

        # avoid fitting a new vectorizer every time (costly)
        if not hasattr(self, "_vectorizer") or self._vectorizer_vocab != top_token_list:
            self._vectorizer = TfidfVectorizer(
                vocabulary=top_token_list,
                input="content",
                dtype=np.float32,
                lowercase=False,
            )
            self._vectorizer_vocab = top_token_list

        tfidf_matrix = self._vectorizer.fit_transform(
            [" ".join(tokens)]
        )  # format (n_samples=1, n_features=self.top_n)

        assert tfidf_matrix.shape[1] == len(
            top_token_list
        ), "TFIDF matrix shape mismatch with top token list length."
        assert (
            tfidf_matrix.shape[0] == 1
        ), "TFIDF matrix should have one row for the single input document."

        return tfidf_matrix.toarray()

    @staticmethod
    def tokenize_whitespace(text: str, normalize_ws: bool = True):
        """
        Tokenize input text by any whitespace character (including \n \r \t \f and spaces).
        Kocher et al. (2015) use isolated words without stemming but with punctuation symbols.

        References:
        ===========
        Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.

        :param text: input text
        :param normalize_ws: collapse whitespace before tokenization
        :return: list of tokens
        """
        if normalize_ws:
            text = re.sub(r"\s+", " ", text)
        return text.split()
