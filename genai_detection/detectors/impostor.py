import logging
import os
import sys
import typing as t
from typing import Iterable, List, Literal

import numpy as np
import torch

from genai_detection.detectors.components.feature_extractor import TfidfFeatureExtractor
from genai_detection.detectors.components.impostor_factory import create_impostor_generator
from genai_detection.detectors.components.preprocessing import Preprocessor, PairPreprocessor
from genai_detection.detectors.components.scorer import Scorer
from genai_detection.detectors.impostor_base import ImpostorBase
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from genai_detection.config import CONFIG

__all__ = ["ImpostorDetector"]

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )


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
            "on-the-fly",   # Startpage by default
            "on_the_fly_chatnoir",
            "on_the_fly_serpapi",
            "on_the_fly_startpage",
            "blogs",
            "fixed",
            "content",
            "naive_llm",
            "two_step_llm",
            "mirror_minds",
        ] = "two_step_llm",
        path2imp: str = CONFIG.PATH2BLOG,  # PATH2GENERIC_ON_FLY_IMP,  # path to impostor file, where fixed impostors are saved or where to save generated impostors
        min_n_tokens: int = 500,  # minimum number of tokens to consider input sequence valid, defaults to 500
        upsample: bool = True,  # whether to upsample short texts (default: True, i.e., upsample) or skip them
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
            - "on_the_fly": generate same-topic impostors on-the-fly (Koppel et al. (2014))
            - "on_the_fly_<approach>": Where <approach> is either  "chatnoir", "serpapi", "startpage"; i.e., different implementations of the same approach.
            - "blogs": use blogs to obtain same genre impostors (Koppel et al. (2014), not implemented yet)
        :param path2imp: Path to the impostor directory, where fixed impostors are saved or where to save newly
        generated impostors
        :param min_n_tokens: Minimum number of tokens to consider input sequence valid, defaults to 500 (Bevendorff
        et al. (2019): 500 words)
        :param upsample: Whether to upsample short texts (default: True, i.e. upsample acc. to Bevendorff (2019)) or
        skip them (Bevendorff et al. (2019)/ Koppel et al (2014) at 500 words)
        """
        super().__init__()

        self.rounds = rounds
        self.top_n = top_n
        self.shared_vocab_only = shared_vocab_only
        self.portion_delete = portion_delete
        self.n_impostors = n_impostors
        self.tfidf_freqs = tfidf_freqs
        self.tokenizer = tokenizer or self.tokenize_char_ngrams
        self.threshold = threshold
        self.path2imp = path2imp
        self.min_n_tokens = min_n_tokens
        self.upsample = upsample
        self.impostor_technique = impostor_technique
        self._training_mode = True  # set to True if you are in training mode, False for validation of model
        self.mongoDB = ParaphraseMongoDB()

        self.impostor_generator = create_impostor_generator(
            impostor_technique=impostor_technique, n_impostors=self.n_impostors, path2imp=self.path2imp,
        )
        self.text_preprocessor = Preprocessor()
        self.pair_processor = PairPreprocessor(mongoDB=self.mongoDB, tokenizer=self.tokenizer, min_n_tokens=self.min_n_tokens, upsample=self.upsample)
        self.scorer = Scorer(rounds=self.rounds, portion_delete=self.portion_delete, similarity_fn=self.minmax_similarity)

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
        self.impostor_generator = create_impostor_generator(
            impostor_technique=self.impostor_technique,
            n_impostors=self.n_impostors,
            split="test" if self._training_mode else "train",
            path2imp=self.path2imp
        )

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
        final_scores = []
        for pair in self.pair_processor.preprocess_pairs(text_list=text):
            # --- 1) Generate impostors & validate ----------------------------------------
            # Impostors are generated based on the original, not processed (i.e., upsampled), text, to keep semantic content
            # Hence, impostors will be as short as original text
            # Check if the impostor generator supports "generate_impostors_by_text_id"
            if hasattr(
                self.impostor_generator, "generate_impostors_by_text_id"
            ) and callable(self.impostor_generator.generate_impostors_by_text_id):
                impostors_of_left = (
                    self.impostor_generator.generate_impostors_by_text_id(
                        text_id=pair["left"]["id"]
                    )
                )
                logging.info(
                    f"Obtained impostors by text ID for left text with text ID {pair['left']['id']}. Type of "
                    f"impostors is {type(impostors_of_left)}"
                )
                impostors_of_right = (
                    self.impostor_generator.generate_impostors_by_text_id(
                        text_id=pair["right"]["id"]
                    )
                )
                logging.info(f"Obtained impostors by text ID for right text with text ID {pair['right']['id']}. Type "
                             f"of impostors is {type(impostors_of_right)}")

            else:
                impostors_of_left = self.impostor_generator.generate_impostors(
                    text=pair["left"]["original_text"]
                )
                logging.info(f"Obtained impostors by text for left text.")
                impostors_of_right = self.impostor_generator.generate_impostors(
                    text=pair["right"]["original_text"]
                )
                logging.info(f"Obtained impostors by text for right text.")
            if not isinstance(impostors_of_right, list) or len(impostors_of_right) < 2:
                raise ValueError(
                    f"Right impostor generator must return a list with at least 2 impostors. Is list {isinstance(impostors_of_right, list)} with {len(impostors_of_right)} impostors."
                )
            if not isinstance(impostors_of_left, list) or len(impostors_of_left) < 2:
                raise ValueError(
                    f"Left impostor generator must return a list with at least 2 impostors. Is list {isinstance(impostors_of_left, list)} with {len(impostors_of_left)} impostors."
                )
            pair["left"]["impostors"] = impostors_of_left
            pair["left"]["processed_impostors"] = [self.text_preprocessor.upsample_to_min_n_tokens(text=imp, min_n_tokens=self.min_n_tokens, upsample=self.upsample) for imp in impostors_of_left]
            logging.info(f"Processed left impostors.")

            pair["right"]["impostors"] = impostors_of_right
            pair["right"]["processed_impostors"] = [
                self.text_preprocessor.upsample_to_min_n_tokens(
                    text=imp, min_n_tokens=self.min_n_tokens, upsample=self.upsample
                )
                for imp in impostors_of_right
            ]
            logging.info(f"Processed right impostors.")
            # --- 2) Build corpus for TFIDF -----------------------------------------------
            # Compute TFIDF based on the processed text, which is upsampled if upsample is set to true and the original (preprocessed) text otherwise
            corpus = [pair["left"]["processed_text"], pair["right"]["processed_text"]] +  pair["left"]["processed_impostors"] +  pair["right"]["processed_impostors"]

            feature_extractor = TfidfFeatureExtractor()
            X = feature_extractor.fit_transform(corpus)
            logging.info(f"Feature extractor (i.e., TFIDF) fit-transform done.")

            # --- 3) Slice TF-IDF vectors cleanly -----------------------------------------
            def dense_vector(row):
                """Helper to convert sparse TF-IDF row to a dense list."""
                return row.toarray().flatten().tolist()

            idx_left = 0
            idx_right = 1
            idx_left_impostors_start = 2
            idx_left_impostors_end = 2 + len(impostors_of_left)
            idx_right_impostors_start = idx_left_impostors_end
            idx_right_impostors_end = idx_right_impostors_start + len(
                impostors_of_right
            )

            pair["left"]["tfidf"] = dense_vector(X[idx_left])
            pair["right"]["tfidf"] = dense_vector(X[idx_right])

            left_impostors_tfidf = [
                dense_vector(X[i])
                for i in range(idx_left_impostors_start, idx_left_impostors_end)
            ]

            right_impostors_tfidf = [
                dense_vector(X[i])
                for i in range(idx_right_impostors_start, idx_right_impostors_end)
            ]
            pair["left"]["impostors_tfidf"] = left_impostors_tfidf
            pair["right"]["impostors_tfidf"] = right_impostors_tfidf
            logging.info("Obtained TFIDF vector for candidate and disputed text, as well as impostors.")

            # --- 4) Final store structure ------------------------------------------------
            # TFIDF is too big to be saved (BSON error during mongodb upload)
            document2insert = {
                f"{old_key}_{new_key}": pair[old_key][new_key]
                for old_key in ["left", "right"]
                for new_key in pair[old_key]
                if new_key != "impostors_tfidf"
            }
            document2insert["scores_over_different_rounds"] = self.scorer.score_pair(pair=pair, vectorizer=feature_extractor.vectorizer)
            logging.info(f"Obtained final score of {document2insert['scores_over_different_rounds']} for text input pair.")
            try:
                self.mongoDB.insert_document(collection=self.mongoDB.impostor_output_collection, insert_data=document2insert)
            except Exception as e:
                logging.error(f"Failed to insert document: {document2insert}\n\n{e}")
            logging.info(f"Inserted score into MongoDB collection {CONFIG.MONGO_IMPOSTOR_OUTPUT_COLLECTION}.")
            final_scores.append(document2insert["scores_over_different_rounds"])
            logging.info(f"Finished computing score for texts with ID {pair['left']['id']} and ID {pair['right']['id']}.")

        # one element = averaged score of X,Y and Y,X pair (score=number of rounds where the candidate was the most similar)
        # threshold is in [0,1], hence: normalized by rounds
        return [v / self.rounds for v in final_scores]

    def get_prediction(self, text: Iterable[str]) -> List[bool]:
        """
        Predict if the input texts were written by the same author

        :param text: input text or batch of input texts
        :return: boolean classifications of whether inputs are likely the same author
        """
        scores = self.get_score(text)
        return [score > self.threshold for score in scores]


if __name__ == "__main__":
    # generating 1 x 50 impostors takes around 40 minutes using openai.
    # ..dd: Ass4 and author TDH426, ..4f: Ass3 and author TDH426, ...e8: Ass4 and author ASR497
    doc_pairs = ["68f50029edacdf3d5c0279dd", #"68f50029edacdf3d5c027d4f"]#, "68f50029edacdf3d5c0279e8"]#,
    # "68f50029edacdf3d5c0279eb"]
    # "68f50029edacdf3d5c0279d9"]
    #              "68f50029edacdf3d5c0279ec"]
                 "68f50029edacdf3d5c0279e0"]

    # "translation",
    # "text_len",
    # "on-the-fly", # startpage by default
    # "on_the_fly_chatnoir",
    # "on_the_fly_serpapi",
    # "on_the_fly_startpage",
    # "blogs",
    # "fixed",
    # "content",
    # "naive_llm",
    # "two_step_llm",
    # "mirror_minds",
    imp = ImpostorDetector(impostor_technique="translation", n_impostors=2)
    res = imp.get_score(text=doc_pairs, normalize=True)
    logging.info(res)
