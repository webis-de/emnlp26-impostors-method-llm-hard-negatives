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

import difflib
from collections import defaultdict
from pathlib import Path

import evaluate
import gensim.downloader
import nltk
import numpy as np
import pandas as pd
from nltk.translate import bleu_score, meteor_score
from sentence_transformers import SentenceTransformer
from word_mover_distance import model  # https://pypi.org/project/word-mover-distance/

from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.paraphrasing.one_step_paraphrasers import *
from genai_detection.experiments.paraphrases.paraphrase_data_loader import ParaphraseDataLoader
from genai_detection.experiments.paraphrases.paraphrase_plots import ParaphrasePlotter
from genai_detection.paraphrasing.two_step_paraphrasers import *
from genai_detection.util import preprocess_text as _preprocess_text

nltk.download("wordnet")  # necessary for METEOR score
logger = logging.getLogger(__name__)

class WMDReadyKeyedVectors:
    def __init__(self, keyed_vectors):
        self.model = keyed_vectors

    def __getitem__(self, key):
        return self.model[key]

    def __contains__(self, key):
        return key in self.model

    def keys(self):
        return self.model.key_to_index.keys()


class ParaphrasingEvaluator:
    def __init__(
        self,
        ground_truth: Optional[dict[str, Any]] = None,
        data_category: Optional[str] = None,
        original_file_name: Optional[str] = None,
    ):
        """
        Initializes the ParaphrasingEvaluator with the given paraphrasers and prompts.
        :param ground_truth: Optional ground truth data to compare against the generated paraphrases.
        :param data_category: Optional category of the data being evaluated, used for logging and saving results.
        :param original_file_name: Optional name of the original file, used for logging and saving results.
        """
        self.mongodb = ParaphraseMongoDB()
        # self.original_file_name = (
        #     str(original_file_name).replace("/", "_").replace(".", "_")
        #     if original_file_name
        #     else "unknown"
        # )

        self.rouge_score = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.sbert_model = None
        try:
            self.sbert_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )  # for cosine similarity
        except Exception as e:
            raise RuntimeError("Failed to load SentenceTransformer model: {e}") from e
        # https://pypi.org/project/word-mover-distance/ Word Mover's Distance (WMD)
        logging.info("Loading pre-trained word vectors for WMD...")
        tries = 0
        self.pretr_word_model = None
        while not self.pretr_word_model and tries < 10:
            try:
                self.pretr_word_model = WMDReadyKeyedVectors(
                    gensim.downloader.load("glove-twitter-25")
                )
            except Exception as e:
                logging.warning(f"Failed to load gensim glove-twitter-25. Retrying...\n{e}")
                tries += 1
        self.wmd_model = model.WordEmbedding(model=self.pretr_word_model)
        self.base_dirs = {
            "blog": Path(__file__).resolve().parents[2] / "data/datasets/Blog_corpus/",
            "gutenberg": Path(__file__).resolve().parents[2]
            / "data/datasets/gutenberg/",
            "student_essays": Path(__file__).resolve().parents[2]
            / "data/datasets/student_essays/Intro2006/",
        }
        self.ground_truth = ground_truth or {}
        self.data_category = data_category or "unknown"
        self.paraphrases_save_base_path = (
            Path(__file__).resolve().parents[2]
            / CONFIG.SAVE_PATH
            / "paraphrasing"
            / "paraphrase_evaluation"
            / self.data_category.replace(" ", "_").replace("/", "_")
        )
        logger.info(f"Paraphrasing evaluation will be saved to {self.paraphrases_save_base_path}")
        self.paraphrases_save_base_path.mkdir(parents=True, exist_ok=True)
        self.plotter = ParaphrasePlotter(save_base_path=self.paraphrases_save_base_path)
        self.data_loader = ParaphraseDataLoader(
            mongodb=self.mongodb, base_dirs=self.base_dirs
        )

    @staticmethod
    def _degree_of_similarity(a: str, b: str) -> float:
        """Calculate similarity ratio between two strings (case-insensitive)."""
        return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

    def _similar(self, a, b, sim_thres: float = 0.65) -> bool:
        """Check if two inputs are sufficiently similar."""
        return self._degree_of_similarity(str(a), str(b)) > sim_thres

    def _semantic_similarity(self, a: str, b: str) -> float:
        """Calculate semantic similarity using BERTScore."""
        if not self.sbert_model:
            raise RuntimeError("SBERT model is not loaded.")
        # -1 = opposite, 0 = no similarity, 1 = identical
        return torch.cosine_similarity(
            self.sbert_model.encode(a, convert_to_tensor=True),
            self.sbert_model.encode(b, convert_to_tensor=True),
            dim=0,
        ).item()

    def _similarity_numbers(self, baseline, other):
        """
        Calculate the percentual similarity of two integers,
        using the first value as the baseline (100%).
        Converts strings to integers if needed.

        Example:
        similarity_percent(100, 90) -> 0.9
        """
        # Convert to int if strings
        baseline = int(baseline)
        other = int(other)

        if baseline == 0:
            return 0

        # Calculate percentage similarity
        similarity = other / baseline
        return similarity

    def _load_dataset(self, base_dir: Path, dataset_type: str) -> pd.DataFrame:
        return self.data_loader.load_dataset(base_dir=base_dir, dataset_type=dataset_type)

    def _obtain_complete_paraphrase_df_from_mongodb(self):
        return self.data_loader.obtain_complete_paraphrase_df_from_mongodb()

    # def _evaluate_model_on_data(
    #     self
    # ):
    #     """
    #     Evaluate a single paraphraser on different datasets.
    #     Returns metrics dictionary and length difference list.
    #     """
    #     df = self._obtain_complete_paraphrase_df_from_mongodb()
    #
    #     results_per_text = []
    #
    #
    #     if "century" not in df.columns and "date" in df.columns:
    #         # Blog dataset has 'date' column (eg. 12,May,2004), convert to 'century'
    #         df["century"] = df["date"].apply(
    #             lambda x: (
    #                 self._get_century(int(x.split(",")[2]))
    #                 if len(x.split(",")) > 2
    #                 else 0
    #             )
    #         )
    #
    #     lengths = pd.DataFrame()
    #     for row in tqdm(
    #         df.itertuples(), total=len(df), desc=f"Evaluating paraphrases"
    #     ):
    #         text = str(getattr(row, "text", ""))
    #         # filename = str(getattr(row, "filename", "unknown"))
    #         #
    #         #
    #         # if str(filename) in data_loaded.keys():
    #         #     extra, genre, century = (
    #         #         data_loaded[filename]["extra"],
    #         #         data_loaded[filename]["genre"],
    #         #         data_loaded[filename]["century"],
    #         #     )
    #         #
    #         #
    #         # if (
    #         #     "current" in str(century).lower()
    #         #     or "present" in str(century).lower()
    #         #     or "now" in str(century).lower()
    #         # ):
    #         #     century = 21
    #         # else:
    #         #     century = self._get_century(century)
    #         # gt_genre = getattr(row, "genre", "").lower() or ""
    #         # gt_century = getattr(row, "century", 0) or 0
    #         # gt_topic = getattr(row, "topic", "") or ""
    #         #
    #         # genre_match = np.max(
    #         #     [
    #         #         self._semantic_similarity(extr_g.strip().lower(), gt_genre)
    #         #         for extr_g in re.split(r"[ /,]+", str(genre).lower())
    #         #     ]
    #         # )
    #         # time_match = self._similarity_numbers(
    #         #     other=century, baseline=gt_century
    #         # )  # self._similar(century, gt_century)
    #         # extracted_topic = (
    #         #     extra.get("topic", extra) if isinstance(extra, dict) else extra
    #         # )
    #         # topic_match = np.max(
    #         #     [
    #         #         self._semantic_similarity(
    #         #             gt_sub_topic.strip(), extracted_topic.strip()
    #         #         )
    #         #         for gt_sub_topic in str(gt_topic).lower().split(",")
    #         #     ]
    #         # )
    #         lengths["original"].append(row["length_original_text"])
    #         lengths["paraphrase"].append(row["length_paraphrased_text"])
    #
    #         results_per_text.append(
    #             {
    #                 "filename": filename,
    #                 "genre_match": genre_match,
    #                 "time_match": time_match,
    #                 "topic_match": topic_match,
    #                 "original_length": row["length_original_text"],
    #                 "paraphrase_length": row["length_paraphrased_text"],
    #                 "ground_truth_genre": gt_genre,
    #                 "ground_truth_century": gt_century,
    #                 "ground_truth_topic": gt_topic,
    #                 "extracted_topic": extra,
    #                 "extracted_genre": genre,
    #                 "extracted_century": century,
    #             }
    #         )
    #
    #     # Convert to DataFrame
    #     results_df = pd.DataFrame(results_per_text)
    #     # Optionally compute total scores
    #     summary = {
    #         "genre_match": results_df["genre_match"].sum(),
    #         "time_match": results_df["time_match"].sum(),
    #         "topic_match": results_df["topic_match"].sum(),
    #         "total": len(results_df),
    #     }
    #
    #     return results_df, summary, lengths

    def plot_metric_radar_per_dataset(
        self,
        df_all: pd.DataFrame,
        metrics: list[str],
        save_path: Path | None = None,
        dataset_col: str = "dataset",
        display_plot: bool = True,
    ):
        self.plotter.plot_metric_radar_per_dataset(
            df_all=df_all,
            metrics=metrics,
            save_path=save_path,
            dataset_col=dataset_col,
            display_plot=display_plot,
        )

    def evaluate_extractors(
        self,
        save_to_disk: bool = True,
        plot_metrics: bool = True,
        display_plot: bool = True,
    ):
        """
        Non-naive paraphrasers extract information from the original text, such as bullet points, task, topic, title, tone, genre, time period and register.
        Some datasets provide some of this information,
        e.g. Blog (id, gender, age, topic, sign, date, text) and
        Gutenberg (title, filename, author, time_period, genre, author_time, summary, fine_genre, century).

        :param save_to_disk: If True, saves the results to a CSV file.
        :param detailed: If True, saves detailed results for each text in the datasets, if false, saves aggregated information for each dataset.
        """
        save_base_path = self.paraphrases_save_base_path
        detailed_detail_degree = "detailed"
        models = {
            k: v
            for k, v in self.paraphrasers.items()
            if not isinstance(v, OneStepParaphraser)
        }

        for dataset_category, base_dir in self.base_dirs.items():
            try:
                df = self._load_dataset(base_dir, dataset_category)
            except Exception as e:
                logger.error(
                    f"Failed to load dataset {dataset_category} from {base_dir}: {e}"
                )
                continue

            if "id" in df.columns:
                df.rename(columns={"id": "filename"}, inplace=True)
            df = df.drop_duplicates(subset="filename", keep="first")
            df = df.head(min(5, len(df)))  # subset: keep for evaluation

            logger.info(f"Dataset snapshot:\n{df.head()}")
            logging.info(f"Dataset snapshot:\n{df.head()}")

            aggregate_results = defaultdict(
                lambda: {
                    "genre_match": 0,
                    "time_match": 0,
                    "topic_match": 0,
                    "total": 0,
                }
            )
            length_differences = {}

            for paraphraser_name, paraphraser in models.items():
                logger.info(
                    f"Evaluating model '{paraphraser_name}' on {dataset_category} dataset..."
                )
                detailed_result_df, summary_df, lengths = self._evaluate_model_on_data(
                    paraphraser_name, paraphraser, df
                )
                logging.info(
                    f"[DEBUG] Detailed results for {paraphraser_name}:\n{detailed_result_df.head()}"
                )

                # Calculate length differences as percentages
                percent_diffs = [
                    ((p - o) / o) * 100 if o > 0 else 0
                    for o, p in zip(lengths["original"], lengths["paraphrase"])
                ]
                length_differences[paraphraser_name] = percent_diffs

                for key in ["genre_match", "time_match", "topic_match", "total"]:
                    aggregate_results[paraphraser_name][key] += summary_df.get(key, 0)

            # Reporting results
            logger.info(f"\n[RESULTS for {dataset_category} dataset]")
            for paraphraser_name, metrics in aggregate_results.items():
                total = metrics["total"]
                if total == 0:
                    logger.warning(
                        f"No evaluation data for model '{paraphraser_name}' on dataset '{dataset_category}'."
                    )
                    continue
                for key in ["genre_match", "time_match", "topic_match"]:
                    aggregate_results[paraphraser_name][key] = round(
                        metrics[key] / total, 2
                    )
                aggregate_results[paraphraser_name]["length_diff"] = round(
                    np.mean(length_differences[paraphraser_name]), 2
                )

                logger.info(f"Model: {paraphraser_name}")
                logger.info(
                    f"  Genre Accuracy: {aggregate_results[paraphraser_name]['genre_match']}"
                )
                logger.info(
                    f"  Time Accuracy (approx): {aggregate_results[paraphraser_name]['time_match']}"
                )
                logger.info(
                    f"  Topic Accuracy (approx): {aggregate_results[paraphraser_name]['topic_match']}"
                )
                logger.info(
                    f"  Length Difference (mean): {aggregate_results[paraphraser_name]['length_diff']}%"
                )

            # Save results if requested
            if save_to_disk:
                self._save_results(
                    detailed_result_df,
                    dataset_category,
                    save_base_path,
                    detail_degree=detailed_detail_degree,
                )
                self._save_results(
                    aggregate_results,
                    dataset_category,
                    save_base_path,
                    detail_degree="aggregated",
                )
                logging.info("Saved results to ", save_base_path)

        if plot_metrics:
            dfs = {}
            logging.info("Read results from disk for plotting from ", save_base_path)
            for dataset in self.base_dirs.keys():
                df = pd.read_csv(
                    save_base_path
                    / f"extractor_eval_results_{dataset}_detailDeg_{detailed_detail_degree}.csv"
                )
                df["length_diff"] = [
                    ((p - o) / o) if o > 0 else 0
                    for o, p in zip(df["original_length"], df["paraphrase_length"])
                ]  # between 0 and 1

                df["dataset"] = dataset
                dfs[dataset] = df

            # Long / tidy combined DataFrame
            df_all = pd.concat(dfs.values(), ignore_index=True)
            self.plot_metric_radar_per_dataset(
                df_all=df_all,
                metrics=["genre_match", "time_match", "topic_match", "length_diff"],
                display_plot=display_plot,
                save_path=save_base_path,
            )

    def _save_results(
        self,
        results: dict,
        dataset_type: str,
        save_base_path: Path,
        detail_degree: str = "detailed",
    ):
        if not save_base_path.exists():
            raise FileNotFoundError(f"Save path {save_base_path} does not exist.")

        save_path = (
            save_base_path
            / f"extractor_eval_results_{dataset_type}_detailDeg_{detail_degree}.csv"
        )
        if not isinstance(results, pd.DataFrame):
            results = pd.DataFrame.from_dict(results, orient="index")
        results.to_csv(save_path)
        logger.info(f"Results saved to {save_path}")

    def _load_or_generate_paraphrases(
        self,
        paraphraser_name: str,
        paraphraser,
        prompt: str,
        temperature: float,
        save_path: Path,
    ) -> List[str]:
        """
        Loads paraphrases from CSV if available; otherwise generates them.
        Returns: list of paraphrases
        """
        data_loaded = dict()
        filename = (
            save_path
            / f"generated_paraphrases_subset_{self.data_category.replace(' ', '_')}.json"
        )
        if filename.exists():
            with open(filename, "r") as f:
                data_loaded = json.load(f)

        try:
            # original text is already preprocessed
            paraphrases = data_loaded[paraphraser_name][self.original_text][prompt][
                temperature
            ]
            logging.info(
                f"Loaded paraphrases for {paraphraser_name} with prompt '{prompt}' and temperature {temperature} from JSON."
            )
            return paraphrases
        except KeyError:
            config = {
                "text": self.original_text,
                "n_responses": self.n_responses,
                "prompt": prompt,
                "temperature": temperature,
            }
            if isinstance(paraphraser, TwoStepParaphraser) and self.ground_truth:
                config["ground_truth"] = self.ground_truth

            paraphrases = [
                _preprocess_text(p) for p in paraphraser.paraphrase(**config)
            ]
            if not paraphrases:
                raise ValueError("Generated paraphrases are empty.")
            data_loaded.setdefault(paraphraser_name, {})
            data_loaded[paraphraser_name].setdefault(self.original_text, {})
            data_loaded[paraphraser_name][self.original_text].setdefault(prompt, {})
            data_loaded[paraphraser_name][self.original_text][prompt][
                temperature
            ] = paraphrases
            with open(filename, "w") as f:
                json.dump(data_loaded, f, indent=4)
            logger.info(f"Paraphrases saved to {save_path}")
            return paraphrases

    def _safe_compute_bertscore(self, paraphrases: list, references: list):
        try:
            return self.bertscore.compute(
                predictions=paraphrases,
                references=references,
                model_type="distilbert-base-uncased",
            )
        except Exception as e:
            logger.error(f"BERTScore computation failed: {e}")
            return {
                "precision": [0.0] * 50,#self.n_responses,
                "recall": [0.0] * 50,#self.n_responses,
                "f1": [0.0] * 50,#self.n_responses,
                "hashcode": "",
            }

    def _safe_compute_rouge(self, paraphrases: list, references: list = None):
        if references is None:
            raise ValueError("References are empty.")
        try:
            assert len(paraphrases) == len(references), f"Length of paraphrases {len(paraphrases)} and references {len(references)} not equal."
            return [self.rouge_score.compute(predictions=[p], references=[r]) for p, r in zip(paraphrases, references)]

        except Exception as e:
            logger.error(f"ROUGE computation failed: len paraphrases {len(paraphrases)} and len references {len(references)}")
            logger.error(f"ROUGE computation failed: {e}")
            return [
                {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
                for _ in paraphrases
            ]

    def evaluate(
        self, save_to_disk: bool = True, save_extremest_paraphr_per_score: bool = False
    ):
        """
        Evaluate the paraphrasers using BERTScore, BLEU and ROUGE metrics.
        :param save_to_disk: If True, saves the results to a CSV file.
        :param save_extremest_paraphr_per_score: If True, saves the worst and best paraphrase per score to a separate CSV file.
        :return: A pandas DataFrame containing the evaluation results.
        """
        df = self._obtain_complete_paraphrase_df_from_mongodb()
        logger.info(f"Evaluating {len(df)} paraphrases from df with columns {df.columns}")

        # use for over (1) llm, (2) prompt, (3) temperature
        group_cols = ["llm", "prompt", "temperature"]
        results = []
        print(df["dataset_name"].value_counts(dropna=False))

        for dataset_name in df["dataset_name"].unique():
            df_dataset = df[df["dataset_name"] == dataset_name]
            logger.info(f"Evaluating {dataset_name} of length {len(df_dataset)}")
            for (paraphraser_name, prompt, temperature), df_group in df_dataset.groupby(group_cols):
                logger.info(f"Evaluating {paraphraser_name}, temperature {temperature}, prompt {prompt} of length {len(df_group)}")
                paraphrases = df_group["paraphrase"].tolist()
                paraphrase_ids = df_group["_id"].tolist()
                references = df_group["text"].tolist()
                reference_ids = df_group["text_id"].tolist()
                assert len(paraphrases) == len(paraphrase_ids), f"Length of paraphrases {len(paraphrases)} != Length of references {len(paraphrase_ids)}"
                logger.info("Number of paraphrases and references found: {}".format(len(paraphrases)))


                bert_scores = self._safe_compute_bertscore(paraphrases=paraphrases, references=references)
                rouge_scores = self._safe_compute_rouge(paraphrases=paraphrases, references=references)

                try:
                    for i, (paraphrase_id, paraphrase, reference_id, reference) in enumerate(zip(paraphrase_ids, paraphrases, reference_ids, references)):
                        existing_scores_cursor = self.mongodb.find_document_by_multiple_fields(
                            collection=self.mongodb.paraphrase_score_collection,
                            search_args={"paraphrase_id": paraphrase_id, "reference_id": reference_id},
                        )
                        existing_result = list(existing_scores_cursor)
                        if not existing_result:
                            logger.info(f"No existing scores for {paraphrase_id} and {reference_id}")
                            row = self._build_result_row(
                                    paraphraser_name=paraphraser_name,
                                    prompt=prompt,
                                    paraphrase=paraphrase,
                                    bert_scores=bert_scores,
                                    rouge_scores=rouge_scores[i],
                                    idx=i,
                                    temperature=temperature,
                                    original_text=reference,
                                )
                            row.pop("paraphrased_text", None)
                            row.pop("original_text", None)
                            row["dataset_name"] = dataset_name
                            row["paraphrase_id"] = paraphrase_id
                            row["reference_id"] = reference_id
                            logger.info(f"Computed scores for {paraphrase_id} and {reference_id}: {row}")
                            self.mongodb.insert_document(collection=self.mongodb.paraphrase_score_collection, insert_data=row)
                            logger.info(f"Inserted {row}")
                        else:
                            logger.info(f"Existing scores for {paraphrase_id} and {reference_id}")
                            row = existing_result[0]
                        assert "dataset_name" in row.keys(), f"'dataset_name' not found in row keys: {row.keys()}"
                        results.append(row)

                except Exception as e:
                    logging.exception(
                        f"Scoring failed for '{paraphraser_name}' with prompt '{prompt}' for category '{self.data_category}': {e}"
                    )
                    logger.error("Error in evaluate function")
                    continue

        df = pd.DataFrame(results)
        # logger.info("%s are complete results.", results)
        logger.info("Number of evaluated paraphrases: %d", len(results))
        # drop any columns that are completely empty, i.e. all NaN
        df.dropna(axis=1, how="all", inplace=True)
        logger.info("Number of evaluated paraphrases after dropping NaNs: %d", len(results))
        if save_to_disk:
            save_path = (
                self.paraphrases_save_base_path
                / f"paraphrasing_results_comparison_dataset.csv"
            )
            df.to_csv(save_path, index=False, float_format="%.4f")
            logging.info(f"Results saved to {save_path}")

        assert "dataset_name" in df.columns, f"'dataset_name' not in df columns, only found: {df.columns}"
        # Save the worst and best paraphrase per score
        extremest_paraphrases = pd.DataFrame()
        logger.info("%s", df.columns)
        for metric in self.get_metric_names():
            min_row = df.loc[df[metric].idxmin()].copy()
            max_row = df.loc[df[metric].idxmax()].copy()

            # Add extra info
            min_row["metric"] = metric
            min_row["extreme"] = "min"

            max_row["metric"] = metric
            max_row["extreme"] = "max"

            # Convert to DataFrames and concat
            extremest_paraphrases = pd.concat(
                [
                    extremest_paraphrases,
                    pd.DataFrame([min_row]),
                    pd.DataFrame([max_row]),
                ],
                ignore_index=True,
            )
        if save_extremest_paraphr_per_score:
            extremest_save_path = (
                self.paraphrases_save_base_path
                / f"extremest_paraphrases_per_metric_dataset.csv"
            )
            extremest_paraphrases.to_csv(extremest_save_path, index=False)

        return df, extremest_paraphrases

    def _build_result_row(
        self,
        paraphraser_name: str,
        prompt: str,
        paraphrase: str,
        bert_scores: dict,
        rouge_scores: dict,
        idx: int,
        original_text:str,
        temperature: float=1.0,
    ) -> dict:
        """
        Build a result row for the DataFrame.
        :param paraphraser_name: Name of the paraphraser.
        :param prompt: The prompt used for paraphrasing excluding the text to paraphrase and tailoring whitespaces, but including bulletpoints etc.
        :param paraphrase: One of the generated paraphrase.
        :param bert_scores: BERTScore results.
        :param rouge_scores: ROUGE scores for the paraphrase.
        :param idx: Index of the paraphrase in the list of BERTScores.
        :param original_text: Original text.
        :param temperature: The temperature used for generating paraphrases.
        :return: A dictionary representing the result row.
        """
        original_split = original_text.split(" ")
        # in [-1, 1] range, where 1 is identical, 0 is no similarity, -1 is opposite
        if self.sbert_model:
            cos_sim = torch.cosine_similarity(
                self.sbert_model.encode(original_text, convert_to_tensor=True),
                self.sbert_model.encode(paraphrase, convert_to_tensor=True),
                dim=0,
            ).item()
        else:
            cos_sim = None
        res = {
            "model": paraphraser_name,
            "prompt": f"{prompt} <TEXT>",
            "parameters": {
                "temperature": temperature,
            },
            "original_text": original_text,
            "paraphrased_text": paraphrase,
            # avoid division by zero using smoothing
            # bleu averages scores obtained from splits of paraphrase and (one of the) reference(s); here: only one reference (i.e. original text)
            "bleu_score": bleu_score.sentence_bleu(  # in [0, 1]
                references=[original_split],
                hypothesis=paraphrase.split(),
                smoothing_function=bleu_score.SmoothingFunction().method1,
            ),  # syntactic similarity metric
            # METEOR requires tokens as input
            "meteor_score": meteor_score.single_meteor_score(
                original_split, paraphrase.split()
            ),  # in [0, 1]
            "rouge1": rouge_scores["rouge1"],  # syntactic similarity metric # in [0, 1]
            "rouge2": rouge_scores["rouge2"],  # in [0, 1]
            "rougeL": rouge_scores["rougeL"],  # syntactic similarity metric # in [0, 1]
            "rougeLsum": rouge_scores["rougeLsum"],  # in [0, 1]
            # bertscore metrics in range [0, 1] cf. https://docs.kolena.com/metrics/bertscore/ (03.07.2025)
            "bertscore_precision": bert_scores["precision"][
                idx
            ],  # semantic similarity metric
            "bertscore_recall": bert_scores["recall"][
                idx
            ],  # semantic similarity metric
            "bertscore_f1": bert_scores["f1"][idx],  # semantic similarity metric
            # normalized (in [0, 1] by using exp) word_mover_similarity
            "sbert_wms": np.exp(
                -self.wmd_model.wmdistance(
                    list(map(str.lower, original_split)), paraphrase.lower().split()
                )
            ),  # semantic similarity metric: exp(-distance) stable version of 1/distance
            # normalize: (cos - (-1)) / (1 - (-1)), so that it is in [0, 1] range
            "sbert_cos": (
                (cos_sim + 1) / 2 if cos_sim else None
            ),  # semantic similarity metric
            # bertscore hashcode for the paraphrase
            "bertscore_hash": bert_scores["hashcode"],
        }
        semantic_sim_average = [
            res["bertscore_precision"],
            res["bertscore_recall"],
            res["bertscore_f1"],
            res["sbert_wms"],
            res["sbert_cos"],
        ]
        # cosine similarity can be None if SentenceTransformer model is not loaded (fails on cluster)
        semantic_sim_average = np.mean(
            [v for v in semantic_sim_average if v is not None]
        )

        res["sem_sim_avg"] = semantic_sim_average
        syntactic_sim_average = np.mean(
            [res["bleu_score"], res["rouge1"], res["rougeL"]]
        )
        res["syn_sim_avg"] = syntactic_sim_average
        res["gohsen_delta"] = semantic_sim_average - syntactic_sim_average
        return res

    def get_metric_names(self) -> List[str]:
        """
        Get the names of the metrics used in the evaluation.
        :return: A list of metric names.
        """
        return [
            "bleu_score",
            "meteor_score",
            "rouge1",
            "rouge2",
            "rougeL",
            "rougeLsum",
            "bertscore_precision",
            "bertscore_recall",
            "bertscore_f1",
            "sbert_wms",
            "sbert_cos",
            "sem_sim_avg",
            "syn_sim_avg",
            "gohsen_delta",
        ]

    def plot_models_metrics(
        self,
        df: pd.DataFrame,
        data_category: Optional[str] = None,
        group_by: Optional[str] = "model",
        display_plot: bool = True,
    ):
        self.plotter.plot_models_metrics(
            df=df,
            metric_names=self.get_metric_names(),
            data_category=data_category,
            group_by=group_by,
            display_plot=display_plot,
        )

    def plot_metric_scatter(
        self,
        df: pd.DataFrame,
        data_category: Optional[str] = None,
        group_by: Optional[str] = "model",
        display_plot: bool = True,
    ):
        self.plotter.plot_metric_scatter(
            df=df,
            data_category=data_category,
            group_by=group_by,
            display_plot=display_plot,
        )

    def plot_metric_distributions(
        self,
        df: pd.DataFrame,
        data_category: Optional[str] = None,
        group_by: Optional[str] = "model",
        display_plot: bool = True,
    ):
        self.plotter.plot_metric_distributions(
            df=df,
            metric_names=self.get_metric_names(),
            data_category=data_category,
            group_by=group_by,
            display_plot=display_plot,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    evaluator = ParaphrasingEvaluator()
    logger.info("Starting evaluation of paraphrasers...")
    # evaluator.evaluate_extractors(save_to_disk=True)

    df, extremest_paraphrases = evaluator.evaluate()
    logger.info(f"Finished computing evaluation scores of paraphrasers... Got columns {df.columns}")
    metrics_names = evaluator.get_metric_names()
    evaluator.plot_metric_radar_per_dataset(df_all=df, dataset_col="dataset_name", display_plot=False, metrics=metrics_names, save_path=evaluator.paraphrases_save_base_path)
    for dataset_name in df["dataset_name"].unique():
        evaluator.plot_metric_scatter(df=df, data_category=dataset_name, display_plot=False)
        evaluator.plot_metric_distributions(df=df, data_category=dataset_name, display_plot=False)
        evaluator.plot_models_metrics(df=df, data_category=dataset_name, display_plot=False)
    logger.info("Evaluation complete.")
