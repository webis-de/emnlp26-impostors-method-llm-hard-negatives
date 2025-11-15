import heapq
import itertools
import json
import os
import pathlib
import re
import sys
import typing as t
from collections import defaultdict, Counter
from random import choices, sample
from typing import Iterable, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from more_itertools import ichunked
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from genai_detection.detectors.impostor_base import ImpostorBase
from genai_detection.impostor_generators import MirrorMinds_generator
from genai_detection.impostor_generators.content_impostor_generator import ContentImpostorGenerator
from genai_detection.impostor_generators.fixed_impostor_generator import (
    FixedImpostorGenerator,
    BlogImpostorGenerator,
)
from genai_detection.impostor_generators.google_search_impostor_generator import GoogleSearchImpostorGenerator
from genai_detection.impostor_generators.naive_impostor_generator import NaiveImpostorGenerator
from genai_detection.impostor_generators.text_length_impostor_generator import TextLenImpostorGenerator
from genai_detection.impostor_generators.translation_impostor_generator import TranslationImpostorGenerator
from genai_detection.impostor_generators.two_step_impostor_generator import TwoStepImpostorGenerator
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from genai_detection.config import CONFIG

__all__ = ["ImpostorDetector"]


class ImpostorDetector(ImpostorBase):
    """
    The Impostor method extends the ngram-unmasking method.
    It uses saves the most similar author to the disputed text for each of multiple random feature selection rounds,
    where the disputed text is compared not only to the candidate text, but also to a set of impostor texts.
    The final prediction is made based on of how often an author is predicted after each feature-elimination step.

    The input is a list of texts where the text ``i`` and the text ``i+1`` belong to a pair.
    The output for one document pair is a score (indicating the same authorship) for the disputed text and the candidate
    text (i.e., author).

    References:
    ===========
    Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
    Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.

    Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.
    """

    def __init__(
        self,
        rounds:int=100,
        top_n:int=100000,
        portion_delete:float=0.5,
        tokenizer=None,
        shared_vocab_only:bool=True,
        tfidf_freqs:bool=True,
        n_impostors:int=25,
        threshold:float=0.1,
        impostor_technique: Literal[
            "translation",
            "text_len",
            "on-the-fly",
            "blogs",
            "fixed",
            "content",
            "naive_llm",
            "two_step_llm",
            "mirror_minds",
        ] = "two_step_llm",
        path2imp: str = CONFIG.PATH2BLOG,  # PATH2GENERIC_ON_FLY_IMP,  # path to impostor file, where fixed impostors are saved or where to save generated impostors
        real_time_generation: bool = False,  # whether to generate impostors in real-time or use pre-generated ones
        min_n_tokens: int = 500,  # minimum number of tokens to consider input sequence valid, defaults to 500
        upsample: bool = True,  # whether to upsample short texts (default: True, i.e. upsample) or skip them
    ):
        """
        :param rounds: Number of random feature selection rounds, Koppel et al. (2014) use 100
        :param top_n: Number of top space-free character 4-grams to consider, Koppel et al. (2014) use 100,000
        :param portion_delete: Portion of features to eliminate in each round (reset in each round); Koppel et al. (
        2014) use 50% of features
        :param tokenizer: Custom tokenizer function (must accept exactly one parameter, defaults to space-free
        character 4-grams cf. Koppel et al. (2014))
        :param shared_vocab_only: Restrict analysis to shared vocabulary across pairs of texts (Koppel et al. (2014): all texts in the corpus, i.e. shared)
        :param tfidf_freqs: Use tfidf term frequencies (Koppel et al. (2014) use tfidf)
        :param n_impostors: Number of impostors to use for each candidate; Koppel et al. (2014) use 25 impostors
        :param threshold: Threshold for the minimum similarity score to consider two texts same-author, Koppel et al. (2014) use 0.1
        :param impostor_technique: Technique to use to generate impostors. Options are:
            - "translation": use LLMs to generate impostors (i.e., English to x and x to English)
            - "naive_llm": use a naive LLM approach to generate impostors
            - "two_step_llm": use two-step LLM approach to generate impostors (i.e., first extracting information and then, generating paraphrases based on these information)
            - "text_len": generate impostors of similar length from a predefined dataset (our baseline w/o reference, default)
            - "fixed": use a fixed set of impostors (Koppel et. A. (2014), not implemented yet), impostors are not related to the input text
            - "on-the-fly": generate same-topic impostors on-the-fly (Koppel et al. (2014), not implemented yet)
            - "blogs": use blogs to obtain same genre impostors (Koppel et al. (2014), not implemented yet)
        :param path2imp: Path to the impostor directory, where fixed impostors are saved or where to save newly
        generated impostors
        :param real_time_generation: Whether to generate impostors in real-time or use pre-generated ones (default:
        False, i.e. use pre-generated impostors)
        :param min_n_tokens: Minimum number of tokens to consider input sequence valid, defaults to 500 (Bevendorff
        et al. (2019): 500 words)
        :param upsample: Whether to upsample short texts (default: True, i.e. upsample acc. to Bevendorff (2019)) or
        skip them (Bevendorff et al. (2019)/ Koppel et al (2014) at 500 words)
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
        self.mongoDB = ParaphraseMongoDB()

        if impostor_technique == "two_step_llm":
            self.impostor_generator = TwoStepImpostorGenerator(
                n_impostors=self.n_impostors
            )
        elif impostor_technique == "naive_llm":
            self.impostor_generator = NaiveImpostorGenerator(
                n_impostors=self.n_impostors
            )
        elif impostor_technique == "translation":
            self.impostor_generator = TranslationImpostorGenerator(
                n_impostors=self.n_impostors
            )
        elif impostor_technique == "fixed":
            self.impostor_generator = FixedImpostorGenerator(
                n_impostors=self.n_impostors,
                split="test" if self._training_mode else "train",
                path2imp=self.path2imp,
            )
        elif impostor_technique == "on-the-fly":
            self.impostor_generator = GoogleSearchImpostorGenerator(
                api_key=CONFIG.SERPAPI_KEY,
                num_queries=max(
                    1, int(self.n_impostors / 25)
                ),  # 25 responses per query
                real_time_generation=real_time_generation
            )
        elif impostor_technique == "blogs":
            self.impostor_generator = BlogImpostorGenerator(
                n_impostors=self.n_impostors,
                split="test" if self._training_mode else "train"
            )
        elif impostor_technique == "content":
            self.impostor_generator = ContentImpostorGenerator(
                n_impostors=self.n_impostors,
                path2imp=path2imp
            )
        elif impostor_technique == "mirror_minds":
            self.impostor_generator = MirrorMinds_generator.MirrorMindsGenerator(
                n_impostors=self.n_impostors
            )
        elif impostor_technique == "text_len":
            self.impostor_generator = TextLenImpostorGenerator(
                n_impostors=self.n_impostors,
                path2imp=path2imp
            )
        else:
            raise NotImplementedError

    def set_treshold(self, threshold: float):
        """
        Set the threshold for the minimum similarity score to consider two texts same-author.
        :param threshold: Threshold value
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in [0, 1].")
        self.threshold = threshold

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
            self.impostor_generator = FixedImpostorGenerator(
                n_impostors=self.n_impostors,
                split="test" if self._training_mode else "train",
                path2imp=self.path2imp,
            )
        elif self.impostor_technique == "blogs":
            self.impostor_generator = BlogImpostorGenerator(
                n_impostors=self.n_impostors,
                split="test" if self._training_mode else "train",
            )

    @staticmethod
    def bootstrap_tokens(tokens, n_tokens: int = 500):
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

        Koppel et al. (2014) exclude texts shorter than 500 words.
        Kocher et al. (2015) exclude words appearing only once to prevent overfitting to words occuring only once.
        Koppel et al. (2014) select m most similar impostors in terms of min-max similarity as impostor candidates and then,
        randomly select n actual impostors among potential impostors (because it has proven superior to using the top n impostors).
        They claim the approach is not sensitive to the choice of m and n.
        Koppel et al. (2014) compare using min-max and cosine similarity.

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
        Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.

        Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.

        :param text: input text or batch of input texts
        :return: score indicating whether the input text is machine-generated, i.e. close 1 means machine-generated, close 0 means human-written
        """
        assert isinstance(
            text, Iterable
        ), "Input text must be iterable. But got: {}".format(
            type(text), text[0:10] if isinstance(text, (str, list)) else text
        )
        text = list(text)  # convert to tuple to list

        scores_per_pair = defaultdict(
            int
        )  # id is the index of the pair (i.e., length is half of the input text list)

        for i, t in enumerate(ichunked(text, 2)):
            t = list(t)  # generator object is not subscriptable, so convert to list
            assert len(t) == 2, "Input text must be a list of pairs of texts."
            if not len(t[0]) + len(t[1]) > 500: # text input is only text_id in mongodb
                text_left, text_id_left = self.mongoDB.get_text_or_id_from_orginal_collection(text=None, text_id=t[0])
                text_right, text_id_right = (
                    self.mongoDB.get_text_or_id_from_orginal_collection(
                        text=None, text_id=t[1]
                    )
                )
            else:
                text_left, text_right = t[0], t[1]
                # TODO: what do i do than?
                text_id_left = 0
                text_id_right = 1
            len_ws_token_left = len(self.tokenize_whitespace(text_left))
            len_ws_token_right = len(self.tokenize_whitespace(text_right))

            # check text length, if too short, i.e., less than 500 `words` (acc. to Koppel et al. (2014) -> invalid;
            # acc. to Bevendorff (2019) -> upsample)
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

            # Control situation via preprocessing: remove genre artifacts, remove html tags (e.g., <nl>), etc.
            # Koppel et al. (2014) do not normalize text pairs.
            # preprocess_text omits all layout/ structural information to keep only style
            # Koppel et al. (2014) use documents of length 500 words exactly -> we DON'T crop at min_n_tokens to keep more information
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
            # freqs_left = Counter(tokens_left)
            # freqs_right = Counter(tokens_right)

            # Kocher et al. (2015) exclude words appearing only once
            # freqs_left = Counter({k: v for k, v in freqs_left.items() if v > 1})
            # freqs_right = Counter({k: v for k, v in freqs_right.items() if v > 1})

            # TODO: generate candidate set first for proper tfidf representation
            # --- 1) Generate impostors & validate ----------------------------------------
            # Check if the impostor generator supports "generate_impostors_by_text_id"
            if hasattr(
                self.impostor_generator, "generate_impostors_by_text_id"
            ) and callable(self.impostor_generator.generate_impostors_by_text_id):
                print("Success: Generating impostors by text ID")

                impostors_of_left = (
                    self.impostor_generator.generate_impostors_by_text_id(
                        text_id=text_id_left
                    )
                )
                impostors_of_right = (
                    self.impostor_generator.generate_impostors_by_text_id(
                        text_id=text_id_right
                    )
                )

            else:
                impostors_of_left = self.impostor_generator.generate_impostors(
                    text=text_left
                )
                impostors_of_right = self.impostor_generator.generate_impostors(
                    text_right
                )
            if not isinstance(impostors_of_left, list) or len(impostors_of_left) < 2:
                raise ValueError(
                    "Left impostor generator must return a list with at least 2 impostors."
                )
            if not isinstance(impostors_of_right, list) or len(impostors_of_right) < 2:
                raise ValueError(
                    "Right impostor generator must return a list with at least 2 impostors."
                )
            # --- 2) Build corpus for TFIDF -----------------------------------------------
            def preprocess_for_tfidf(text: str) -> str:
                tokens = self.tokenizer(self.preprocess_text(text))
                return " ".join(tokens)

            corpus = [text_left, text_right] + impostors_of_left + impostors_of_right

            # char_wb: n-grams only from text inside word boundaries; n-grams at the edges of words are padded with space.
            # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html (14.11.2025)
            # min_df: Kocher et al. (2015) exclude words appearing only once
            vectorizer = TfidfVectorizer(ngram_range=(4, 4), analyzer="char_wb", min_df=2)
            X = vectorizer.fit_transform(corpus)

            def dense_vector(row):
                """Helper to convert sparse TF-IDF row to dense list."""
                return row.toarray().flatten().tolist()
            # --- 3) Slice TF-IDF vectors cleanly -----------------------------------------

            idx_left = 0
            idx_right = 1
            idx_left_impostors_start = 2
            idx_left_impostors_end = 2 + len(impostors_of_left)
            idx_right_impostors_start = idx_left_impostors_end
            idx_right_impostors_end = idx_right_impostors_start + len(
                impostors_of_right
            )

            left_tfidf = dense_vector(X[idx_left])
            right_tfidf = dense_vector(X[idx_right])

            left_impostors_tfidf = [
                dense_vector(X[i])
                for i in range(idx_left_impostors_start, idx_left_impostors_end)
            ]

            right_impostors_tfidf = [
                dense_vector(X[i])
                for i in range(idx_right_impostors_start, idx_right_impostors_end)
            ]

            # --- 4) Final store structure ------------------------------------------------

            store = {
                "left": {
                    "tfidf": left_tfidf,
                    "tokens": tokens_left,
                    "text": text_left,
                    "author": "unknown",
                    "impostors": impostors_of_left,
                    "impostors_tfidf": left_impostors_tfidf,
                },
                "right": {
                    "tfidf": right_tfidf,
                    "tokens": tokens_right,
                    "text": text_right,
                    "author": "unknown",
                    "impostors": impostors_of_right,
                    "impostors_tfidf": right_impostors_tfidf,
                },
            }

            for j, (disputed, candidate) in enumerate(
                itertools.permutations(list(store.keys()), 2)
            ):
                scores_over_different_rounds = 0
                # for different rounds, randomly delete a portion of features (reset in each round)
                assert (
                    vectorizer.vocabulary_ is not None
                ), "TFIDF Vectorizer vocabulary is not set. Please ensure that the vectorizer is fitted before calling _get_score_impl."
                for _ in range(self.rounds):
                    # feature selection: randomly delete a portion of features
                    rand_feat_to_keep_ids = sample(
                        range(len(vectorizer.vocabulary_)),
                        int(len(vectorizer.vocabulary_) * (1 - self.portion_delete)),
                    )
                    reduced_disputed_tfidf = np.array(store[disputed]["tfidf"])[rand_feat_to_keep_ids]
                    impostor_scores = [self.minmax_similarity(reduced_disputed_tfidf,  # disputed text
                            np.array(imp_tfidf)[rand_feat_to_keep_ids],
                        ) for imp_tfidf in store[candidate]["impostors_tfidf"]]

                    # increase score if the most similar candidate is the actual candidate
                    max_similar_imp_val = max(impostor_scores)
                    disputed_candidate_sim_val = self.minmax_similarity(reduced_disputed_tfidf,  # disputed text
                            np.array(store[candidate]["tfidf"])[rand_feat_to_keep_ids],
                        )
                    scores_over_different_rounds += (disputed_candidate_sim_val > max_similar_imp_val)
                # average after second loop
                scores_per_pair[i] += scores_over_different_rounds
                scores_per_pair[i] /= (j + 1)

        # TODO: save score to mongodb collection

        # one element = averaged score of X,Y and Y,X pair (score=number of rounds where the candidate was the most similar)
        # threshold is in [0,1], hence: normalized by rounds
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
        Predict if the input text(s) were written by a the same author

        :param text: input text or batch of input texts
        :return: boolean classifications of whether inputs are likely same author
        """
        scores = self.get_score(text)
        return [score > self.threshold for score in scores]

    def tokens_to_matrix(self, tokens:List[str], path2imp: Optional[pathlib.Path] = None):
        """
        Transform a list of tokens into a matrix of term-tfidf-values of the top tokens.
        Koppel et al. (2014) use space-free character 4-grams tfidf values to represent each document as a numerical vector.

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.

        :param tokens: list of input tokens (e.g., space-free character 4-grams)
        :param path2imp: Path to impostors.
        :return: Numpy array of term tfidf values, `shape = (len(tokens), len(top_token_list))`
        """
        if not hasattr(self, "_vectorizer") or not hasattr(
            self._vectorizer, "vocabulary_"
        ):
            self._update_vectorizer_if_necessary(
                path2imp=path2imp,
                input_tokens=tokens,
            )
        try:
            tfidf_matrix = self._vectorizer.transform([" ".join(tokens)])
        except Exception as e:
            print(f"Error transforming tokens to matrix: {e}.")
            # avoid fitting a new vectorizer every time (costly)
            self._update_vectorizer_if_necessary(
                path2imp=path2imp,
                input_tokens=tokens,
            )
        try:
            tfidf_matrix = self._vectorizer.transform([" ".join(tokens)])
        except Exception as e:
            raise Exception("SECOND Error transforming tokens to matrix: {}".format(e))

        return tfidf_matrix.toarray()

    def _update_vectorizer_if_necessary(self, path2imp:pathlib.Path, input_tokens):  # top_token_list
        train_data = None
        candidate_texts = [" ".join(input_tokens)]
        # TODO: Fit on complete set of candidates and disputed document
        print(
            f"Fitting TFIDF vectorizer on input tokens or candidate texts from {path2imp}."
        )
        if path2imp and path2imp.exists():
            split = "train" if self._training_mode else "test"
            if path2imp.suffix == ".json":
                print(f"Loading impostor data from JSON file: {path2imp}")
                with open(path2imp, "r") as f:
                    # dumps have structure {outer_key: {inner_key: value}} -> load as dict and then reshape to long format
                    raw = json.load(f)
                    train_data = pd.DataFrame.from_dict(raw, orient="index")
                    train_data = train_data.reset_index().melt(
                        id_vars="index", var_name="inner_key", value_name="value"
                    )
                    train_data = train_data.dropna(subset=["value"])  # drop missing
                    train_data["pair"] = train_data.apply(
                        lambda row: [row["index"], row["value"]], axis=1
                    )
                    assert isinstance(
                        train_data, pd.DataFrame
                    ), f"Expected train_data to be a pandas DataFrame, but got {type(train_data)}."
                    if split in train_data:
                        train_data = train_data[split]
            else:
                train_data = load_from_disk(path2imp)[split].to_pandas()
            assert isinstance(
                train_data, pd.DataFrame
            ), f"Expected train_data to be a pandas DataFrame, but got {type(train_data)}."
            if not train_data is None and not train_data.empty:
                candidate_texts = []
                for _, row in train_data.iterrows():
                    entry = row.to_dict()
                    assert isinstance(
                        entry, dict
                    ), f"Each entry in the dataset must be a dictionary (tokens_to_matrix). But is {type(entry)}/entry:{entry}/row:{row}."
                    pair = entry.get("pair", [])
                    candidate_texts.extend(pair)
        else:
            print(
                f"Warning: path2imp {path2imp} does not exist. Using only input tokens for vectorizer fitting."
            )
        assert isinstance(
            candidate_texts, list
        ), f"Candidate texts must be a list, but got {type(candidate_texts)}."
        assert len(candidate_texts) > 0, f"No candidate texts found in {path2imp}."
        tokens = [
            token
            for t in candidate_texts
            for token in self.tokenizer(self.preprocess_text(t))
        ]
        freqs = Counter(tokens)
        freqs = Counter({k: v for k, v in freqs.items() if v > 1})
        self._vectorizer_vocab = heapq.nlargest(
            self.top_n, list(freqs.keys()), key=lambda x: freqs[x]
        )
        self._vectorizer = TfidfVectorizer(
            vocabulary=self._vectorizer_vocab,
            input="content",
            dtype=np.float32,
            lowercase=False,  # do not lowercase all, treat tokens as case-sensitive
        )

        self._vectorizer = self._vectorizer.fit(candidate_texts)
        if (
            not hasattr(self._vectorizer, "vocabulary_")
            or self._vectorizer.vocabulary_ is None
        ):
            raise RuntimeError("Vectorizer fitting failed: vocabulary is empty.")

        return self._vectorizer

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


if __name__ == "__main__":
    doc_pairs = ["68f50029edacdf3d5c0279e8", "68f50029edacdf3d5c0279ea"]
    imp = ImpostorDetector(impostor_technique="two_step_llm", n_impostors=4)
    res = imp.get_score(text=doc_pairs, normalize=True)
    print(res)
