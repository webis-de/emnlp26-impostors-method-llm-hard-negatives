import datetime
import logging
import os
import sys
import typing as t
from typing import Any, Iterable, List, Literal

import numpy as np
import torch
from bson import ObjectId
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import binom_test, proportion_effectsize

from genai_detection.detectors.components.feature_extractor import TfidfFeatureExtractor
from genai_detection.detectors.components.impostor_factory import create_impostor_generator
from genai_detection.detectors.components.preprocessing import PairPreprocessor, Preprocessor
from genai_detection.detectors.components.scorer import Scorer, ScoreResult
from genai_detection.detectors.components.vector_similarity import minmax_similarity
from genai_detection.detectors.impostor_base import ImpostorBase
from genai_detection.impostor_generators.search_generator_base import SearchImpostorGeneratorBase

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from genai_detection.config import CONFIG

__all__ = ["ImpostorDetector"]

logger = logging.getLogger(__name__)


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
        rounds: int = 100,
        top_n: int = 100000,
        portion_delete: float = 0.5,
        n_impostors: int = 25,
        threshold: float = 0.1,
        impostor_technique: Literal[
            "translation",
            "on-the-fly",  # Startpage by default
            "on_the_fly_chatnoir",
            "on_the_fly_serpapi",
            "on_the_fly_startpage",
            "in_domain",
            "one_step_llm",
            "two_step_llm",
            "mirror_minds",
        ] = "two_step_llm",
        dataset_name: str = "student_essays",
        min_n_tokens: int = 500,  # minimum number of tokens to consider input sequence valid, defaults to 500
        upsample: bool = True,  # whether to upsample short texts (default: True, i.e., upsample) or skip them
    ):
        """
        :param rounds: Number of random feature selection rounds, Koppel et al. (2014) use 100
        :param top_n: Number of top space-free character 4-grams to consider, Koppel et al. (2014) use 100,000
        :param portion_delete: Portion of features to eliminate in each round (reset in each round); Koppel et al. (
        2014) use 50% of features
        :param n_impostors: Number of impostors to use for each candidate; Koppel et al. (2014) use 25 impostors
        :param threshold: Threshold for the minimum similarity score to consider two texts same-author, Koppel et al. (2014) use 0.1
        :param impostor_technique: Technique to use to generate impostors. Options are:
            - "translation": use LLMs to generate impostors (i.e., English to x and x to English)
            - "one_step_llm": use a naive LLM approach to generate impostors
            - "two_step_llm": use two-step LLM approach to generate impostors (i.e., first extracting information and then, generating paraphrases based on these information)
            - "in_domain": use a fixed set of in-domain impostors (Koppel et. A. (2014), not implemented yet),
            impostors are not related to the input text
            - "on_the_fly": generate same-topic impostors on-the-fly (Koppel et al. (2014))
            - "on_the_fly_<approach>": Where <approach> is either  "chatnoir", "serpapi", "startpage"; i.e., different implementations of the same approach.
        :param dataset_name: Name of dataset, where fixed impostors are sampled from (in mongoDB original_text collection)
        :param min_n_tokens: Minimum number of tokens to consider the input sequence valid, defaults to 500 (Bevendorff
        et al. (2019): 500 words)
        :param upsample: Whether to upsample short texts (default: True, i.e., upsample acc. to Bevendorff (2019)) or
        skip them (Bevendorff et al. (2019)/ Koppel et al. (2014) at 500 words)
        """
        super().__init__()

        self.rounds = rounds
        self.top_n = top_n
        self.portion_delete = portion_delete
        self.n_impostors = n_impostors
        self.threshold = threshold
        self.dataset_name = dataset_name
        self.min_n_tokens = min_n_tokens
        self.upsample = upsample
        self.impostor_technique = impostor_technique

        self.impostor_generator = create_impostor_generator(
            impostor_technique=impostor_technique,
            n_impostors=self.n_impostors,
            dataset_name=self.dataset_name,
            top_n_freq_words=self.top_n,
        )
        self.text_preprocessor = Preprocessor()
        self.feature_extractor = TfidfFeatureExtractor(top_n_freq_words=self.top_n)
        self.pair_processor = PairPreprocessor(
            mongoDB=self.mongoDB,
            tokenizer=self.feature_extractor._space_free_char_ngrams,
            min_n_tokens=self.min_n_tokens,
            upsample=self.upsample,
        )
        self.scorer = Scorer(
            rounds=self.rounds,
            portion_delete=self.portion_delete,
            similarity_fn=minmax_similarity,
        )
        self.significance_level = 0.05
        self.impostor_output_collection = self.mongoDB.impostor_output_collection

    def set_treshold(self, threshold: float):
        """
        Set the threshold for the minimum similarity score to consider two texts same-author.
        :param threshold: Threshold value
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be in [0, 1].")
        self.threshold = threshold

    def _generate_impostors_for_single_input(
        self, input_dict: t.Dict[str, str]
    ) -> List[str]:
        """

        :param input_dict: Dictionary of input text with keys "id"
        :return: impostors
        """
        # --- 1) Generate impostors & validate ----------------------------------------
        # Impostors are generated based on the original, not processed (i.e., upsampled), text, to keep semantic content
        # Hence, impostors will be as short as original text
        # Check if the impostor generator supports "generate_impostors_by_text_id"
        if hasattr(
            self.impostor_generator, "generate_impostors_by_text_id"
        ) and callable(self.impostor_generator.generate_impostors_by_text_id):
            # check if result has already been computed and stored in the mongoDB collection
            assert (
                "id" in input_dict.keys()
            ), f"The 'id' key must be provided in input_dict. Only found {input_dict.keys()}."
            text_id = input_dict["id"]
            assert isinstance(
                text_id, ObjectId
            ), f"input IDs must be ObjectID, but are {type(text_id)}."
            impostors = self.impostor_generator.generate_impostors_by_text_id(
                text_id=text_id
            )
        else:
            assert (
                "original_text" in input_dict.keys()
            ), f"The 'original_text' key must be provided in input_dict. Only found {input_dict.keys()}."
            impostors = self.impostor_generator.generate_impostors(
                text=input_dict["original_text"]
            )

        if not isinstance(impostors, list) or len(impostors) < 2:
            raise ValueError(
                f"Impostor generator must return a list with at least 2 impostors. Is list {isinstance(impostors, list)} with {len(impostors)} impostors."
            )

        return impostors

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
        # consider only pairs long enough & obtain texts for IDs; no preprocessing so far
        search_args = {
            "impostor_generation_technique": self.impostor_technique,
            "n_impostors": self.n_impostors,
            "n_potential_impostors": self.impostor_generator.num_potential_impostors,
        }
        if (
            self.impostor_output_collection.name
            == CONFIG.MONGO_IMPOSTOR_ABLATION_OUTPUT_COLLECTION
        ):
            # ablations inherit from this method and need to specify their approach type
            search_args["ablation"] = self.__class__.__name__
        for pair in self.pair_processor.filter_pairs(text_list=text):
            # query for existing score in mongodb collection
            search_args.update({
                "left_id": ObjectId(pair["left"]["id"]),
                "right_id": ObjectId(pair["right"]["id"]),
            })
            cursor = list(
                self.mongoDB.find_document_by_multiple_fields(
                    collection=self.impostor_output_collection,
                    search_args=search_args,
                )
            )
            if len(cursor) > 0:
                final_scores.append(cursor[0]["scores_over_different_rounds"])
                # logger.info(
                #     f"Found pre-computed scores for {pair['left']['id']}, {pair['right']['id']} in mongoDB collection. Using pre-computed scores.")
                continue

            generation_failed, pair = self._build_impostor_pair_datastructure(pair=pair)

            if generation_failed:
                final_scores.append(0.5) 
                continue  # skip to next pair and save nothing to mongoDB

            # --- 2) Build corpus for TFIDF -----------------------------------------------
            # Compute TFIDF based on the processed text, which is upsampled (and preprocessed) if upsample is set to true
            # and the original preprocessed text otherwise
            corpus = []

            def upsample_then_preprocess(input_text: str) -> str:
                upsampled_text = self.text_preprocessor.upsample_to_min_n_tokens(
                    text=input_text,
                    min_n_tokens=self.min_n_tokens,
                    upsample=self.upsample,
                )
                return self.preprocess_text(text=upsampled_text)

            index_map = {}  # save indices parallel to corpus construction
            for side in ["left", "right"]:
                # preprocess original texts and impostors: (1) upsampling, (2) self.preprocess_text
                pair[side]["processed_text"] = upsample_then_preprocess(
                    input_text=pair[side]["original_text"]
                )
                index_map[f"{side}_text"] = len(corpus)
                corpus.append(pair[side]["processed_text"])

                if "impostors" in pair[side]:
                    pair[side]["processed_impostors"] = [
                        upsample_then_preprocess(input_text=imp)
                        for imp in pair[side]["impostors"]
                    ]
                    index_map[f"{side}_impostors"] = list(
                        range(
                            len(corpus),
                            len(corpus) + len(pair[side]["processed_impostors"]),
                        )
                    )
                    corpus.extend(pair[side]["processed_impostors"])

                logger.info(
                    "Preprocessed %s impostors and input texts (first (optionally) upsampled, then preprocessed) and added preprocessed text to TFIDF corpus.",
                    side,
                )

            assert (
                len(corpus) > 0
            ), f"Length of TFIDF corpus is {len(corpus)}, no preprocessed texts or preprocessed impostors in the corpus."

            X = self.feature_extractor.fit_transform(corpus)
            logger.info(f"Feature extractor (i.e., TFIDF) fit-transform done.")

            # --- 3) Slice TF-IDF vectors cleanly -----------------------------------------
            def dense_vector(row):
                """Helper to convert sparse TF-IDF row to a dense list."""
                return row.toarray().flatten().tolist()

            for side in ["left", "right"]:
                pair[side]["tfidf"] = dense_vector(X[index_map[f"{side}_text"]])
                if f"{side}_impostors" in index_map:
                    pair[side]["impostors_tfidf"] = [
                        dense_vector(X[i]) for i in index_map[f"{side}_impostors"]
                    ]

            logger.info(
                "Obtained TFIDF vector for candidate and disputed text, as well as impostors."
            )

            # --- 4) Final store structure ------------------------------------------------
            # TFIDF is too big to be saved (BSON error during mongodb upload)
            assert isinstance(pair["left"], dict) and isinstance(pair["right"], dict)
            document2insert = {
                f"{old_key}_{new_key}": value
                for old_key in ["left", "right"]
                for new_key, value in pair[old_key].items()
                if new_key not in {
                    "tfidf",
                    "processed_text",
                    "tokens",
                    "processed_impostors",
                    "impostors",
                    "original_text",
                    "impostors_tfidf",
                }
            }
            score_result = self.scorer.score_pair(
                pair=pair, vectorizer=self.feature_extractor.vectorizer
            )
            if not isinstance(score_result, ScoreResult):
                raise TypeError(
                    f"score_pair must return ScoreResult, got {type(score_result)}."
                )
            assert 0 <= score_result.score <= self.rounds, f"Score must be in [0,{self.rounds}], got {score_result.score}."
            document2insert["scores_over_different_rounds"] = score_result.score
            p_values = score_result.p_values
            if p_values:    # ablations inherit from this method, but do not return p-values
                document2insert = self._handle_statistical_test(document2insert, p_values, pair)

            document2insert["impostor_generation_technique"] = self.impostor_technique
            if isinstance(self.impostor_generator, SearchImpostorGeneratorBase):
                document2insert["impostor_generation_technique"] = "on_the_fly"
                document2insert["retrieval_index"] = self.impostor_generator.index_name
            document2insert["dataset_name"] = (
                self.dataset_name or self.impostor_generator.dataset_name
            )
            document2insert["n_impostors"] = self.n_impostors
            document2insert["n_potential_impostors"] = (
                self.impostor_generator.num_potential_impostors
            )
            if self.impostor_output_collection.name == CONFIG.MONGO_IMPOSTOR_ABLATION_OUTPUT_COLLECTION:
                # ablations inherit from this method and need to specify their approach type
                document2insert["ablation"] = (
                    self.__class__.__name__
                )
                logger.info(f"Ablation {self.__class__.__name__} will be inserted into MongoDB collection.")
            document2insert["created_at"] = datetime.datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
            logger.info(
                f"Obtained final score of {document2insert['scores_over_different_rounds']} for text input "
                f"pair. About to insert instance with following keys: {document2insert.keys()}."
            )
            try:
                self.mongoDB.insert_document(
                    collection=self.impostor_output_collection,
                    insert_data=document2insert,
                )
                logger.info(
                    "Inserted score into MongoDB collection %s.",
                    self.impostor_output_collection.name,
                )
            except Exception as e:
                logger.error(f"Failed to insert document: {document2insert}\n\n{e}")

            final_scores.append(document2insert["scores_over_different_rounds"])
            logger.info(
                f"Finished computing score for texts with ID {pair['left']['id']} and ID {pair['right']['id']}."
            )

        # one element = averaged score of X,Y and Y,X pair (score=number of rounds where the candidate was the most similar)
        # threshold is in [0,1], hence: normalized by rounds
        return [v / self.rounds for v in final_scores]

    def _build_impostor_pair_datastructure(self, pair: t.Dict[str, str], keys: List[str] = ["left", "right"]) -> (
                bool, t.Dict[str, str]):
        generation_failed = False
        for key in keys:
            try:
                pair[key]["impostors"] = self._generate_impostors_for_single_input(input_dict=pair[key])
                logger.info(
                        f"Obtained {len(pair[key]['impostors'])} impostors for {key} input text. Type of impostors is {type(pair[key]['impostors'])}.")
            except Exception as e:
                logger.error(
                        f"Failed to generate impostors for text with ID {pair[key]['id']} and text: {pair[key]['original_text'][:200]} -> Return 0.5. Error: {e}")
                generation_failed = True
                break  # exit for loop over sides
        return generation_failed, pair

    def _handle_statistical_test(self, document2insert: dict[str, Any], p_values, pair):
        # aggregated score over different rounds (due to overlap in vocabularies, scores are not independent over different rounds and this test thus, lacks correctness)
        document2insert["uncorr_p_val_over_different_rounds"] = binom_test(
                count=2 * document2insert["scores_over_different_rounds"], nobs=self.rounds * 2,
                prop=1 / (1 + len(pair["left"]["impostors_tfidf"])), alternative="larger", )
        document2insert["corr_pred_over_different_rounds"] = bool(
                document2insert["uncorr_p_val_over_different_rounds"] < 2 * self.significance_level)

        # compare corrected p-value (times 2, since two tests) to alpha for statistical significance
        # https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html#statsmodels.stats.multitest.multipletests (09.01.2026)
        rejects, pvals_corrected, _, alphacBonf = multipletests(pvals=list(p_values.values()),
                alpha=self.significance_level, method="bonferroni", )
        logger.info(f"Corrected p-values: {pvals_corrected} and uncorrected p-values: {p_values.values()}, "
                    f"left_id: {pair['left']['id']} and right_id: {pair['right']['id']}")

        # reject null hypothesis means texts were written by same author
        preds = {f"{key}_pred": bool(reject) for key, reject in zip(p_values.keys(), rejects)}
        corrected_pval = {key.replace("uncorrected", "corrected"): c_pval for key, c_pval in
                zip(p_values.keys(), pvals_corrected)}

        # compute Cohen's H based on corrected p-values
        effect_size = {f"effect_size_{'_'.join(k.split('_')[:3])}": proportion_effectsize(prop1=corr_pval,
                prop2=1 / (1 + len(pair[k.split("_")[2]]["impostors_tfidf"])), method="normal", ) for k, corr_pval in
                zip(p_values.keys(), pvals_corrected)}

        # Update the document dictionary
        document2insert.update(preds)
        document2insert.update(p_values)
        document2insert.update(effect_size)
        document2insert.update(corrected_pval)
        document2insert.update({"bonferroni_corrected_alph": alphacBonf})
        return document2insert

    def get_prediction(self, text: Iterable[str]) -> List[bool]:
        """
        Predict if the input texts were written by the same author

        :param text: input text or batch of input texts
        :return: boolean classifications of whether inputs are likely the same author
        """
        scores = self.get_score(text)
        return [score > self.threshold for score in scores]
