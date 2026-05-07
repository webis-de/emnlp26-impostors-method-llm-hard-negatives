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
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.paraphrasing.one_step_paraphrasers import *
from genai_detection.experiments.paraphrases.paraphrase_data_loader import ParaphraseDataLoader
from genai_detection.experiments.paraphrases.paraphrase_metrics import (
    ParaphraseMetricCalculator,
)
from genai_detection.experiments.paraphrases.paraphrase_plots import ParaphrasePlotter
from genai_detection.paraphrasing.two_step_paraphrasers import *
from genai_detection.util import preprocess_text as _preprocess_text

logger = logging.getLogger(__name__)


class ParaphrasingEvaluator:
    def __init__(
        self,
        ground_truth: Optional[dict[str, Any]] = None,
        data_category: Optional[str] = None,
        metric_config: Optional[dict[str, bool]] = None,
    ):
        """
        Initializes the ParaphrasingEvaluator with the given paraphrasers and prompts.
        :param ground_truth: Optional ground truth data to compare against the generated paraphrases.
        :param data_category: Optional category of the data being evaluated, used for logging and saving results.
        :param original_file_name: Optional name of the original file, used for logging and saving results.
        :param metric_config: Optional config passed to ParaphraseMetricCalculator, e.g.
            {"include_meteor": False, "include_wmd": False, "include_sbert_cos": True}.
        """
        self.mongodb = ParaphraseMongoDB()
        self.metric_calculator = ParaphraseMetricCalculator(**(metric_config or {}))
        # Keep these attributes for backwards compatibility with existing callers.
        self.rouge_score = self.metric_calculator.rouge_score
        self.bertscore = self.metric_calculator.bertscore
        self.sbert_model = self.metric_calculator.sbert_model
        self.wmd_model = self.metric_calculator.wmd_model
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
            Path(__file__).resolve().parents[3]
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
        return self.metric_calculator.safe_compute_bertscore(
            paraphrases=paraphrases, references=references
        )

    def _safe_compute_rouge(self, paraphrases: list, references: list = None):
        return self.metric_calculator.safe_compute_rouge(
            paraphrases=paraphrases, references=references
        )

    def _enrich_results_with_source_context(
        self, results_df: pd.DataFrame, source_df: pd.DataFrame
    ) -> pd.DataFrame:
        if results_df.empty or source_df.empty:
            return results_df
        if "paraphrase_id" not in results_df.columns or "_id" not in source_df.columns:
            return results_df

        source_cols = [
            "_id",
            "llm",
            "prompt",
            "temperature",
            "dataset_name",
            "text",
            "paraphrase",
            "paraphrase_approach",
        ]
        available_source_cols = [c for c in source_cols if c in source_df.columns]
        source_info = source_df[available_source_cols].rename(
            columns={
                "_id": "paraphrase_id",
                "llm": "model",
                "text": "original_text",
                "paraphrase": "paraphrased_text",
            }
        )
        source_info = source_info.drop_duplicates(subset="paraphrase_id", keep="first")
        data = results_df.merge(
            source_info, on="paraphrase_id", how="left", suffixes=("", "_src")
        )

        fields_to_merge = [
            "model",
            "prompt",
            "temperature",
            "dataset_name",
            # "original_text",
            # "paraphrased_text",
            "paraphrase_approach",
        ]
        for field in fields_to_merge:
            src_field = f"{field}_src"
            if src_field in data.columns:
                if field in data.columns:
                    data[field] = data[field].fillna(data[src_field])
                    data.drop(columns=[src_field], inplace=True)
                else:
                    data.rename(columns={src_field: field}, inplace=True)

        if "model" in data.columns and "paraphrase_approach" in data.columns:
            data["paraphrase_technique"] = (
                data["model"].astype(str)
                + " ("
                + data["paraphrase_approach"].fillna("unknown").astype(str)
                + ")"
            )

        if {"original_text", "paraphrased_text"}.issubset(data.columns):
            original_lengths = (
                data["original_text"].fillna("").astype(str).str.split().str.len()
            )
            paraphrase_lengths = (
                data["paraphrased_text"].fillna("").astype(str).str.split().str.len()
            )
            data["original_word_count"] = original_lengths
            data["paraphrase_word_count"] = paraphrase_lengths
            data["paraphrase_length_pct_words"] = np.where(
                original_lengths > 0,
                (paraphrase_lengths / original_lengths) * 100,
                np.nan,
            )
        return data

    def evaluate(
        self, save_to_disk: bool = True, save_extremest_paraphr_per_score: bool = False
    ):
        """
        Evaluate paraphrases by reusing cached metric scores from MongoDB and
        computing missing scores as needed.

        Paraphrases are grouped by dataset, LLM, prompt, and temperature. For each
        paraphrase/reference pair, an existing score entry is loaded from MongoDB if
        available; otherwise, the metrics are computed and inserted into MongoDB.

        Parameters
        ----------
        save_to_disk : bool, default=True
            If True, save the full evaluation results to CSV.
        save_extremest_paraphr_per_score : bool, default=False
            If True, save the minimum- and maximum-scoring paraphrase for each metric
            to a separate CSV.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing:
            - the full evaluation results DataFrame
            - a DataFrame with the minimum and maximum paraphrase per metric
        """
        df = self._obtain_complete_paraphrase_df_from_mongodb()
        source_df = df.copy()
        logger.info(f"Evaluating {len(df)} paraphrases from df with columns {df.columns}")

        # use for over (1) llm, (2) prompt, (3) temperature
        group_cols = ["llm", "prompt", "temperature"]
        results: list[dict[str, Any]] = []

        for dataset_name in df["dataset_name"].unique():
            df_dataset = df[df["dataset_name"] == dataset_name]
            logger.info(f"Evaluating {dataset_name} of length {len(df_dataset)}")

            for (paraphraser_name, prompt, temperature), df_group in df_dataset.groupby(group_cols):
                logger.info(f"Evaluating {paraphraser_name}, temperature {temperature}, prompt {prompt} of length {len(df_group)}")

                paraphrases = df_group["paraphrase"].tolist()
                paraphrase_ids = df_group["_id"].tolist()
                references = df_group["text"].tolist()
                reference_ids = df_group["text_id"].tolist()

                assert len(paraphrases) == len(references)
                assert len(paraphrase_ids) == len(reference_ids)
                assert len(paraphrases) == len(paraphrase_ids)

                logger.info("Number of paraphrases and references found: {}".format(len(paraphrases)))

                group_results: list[dict[str, Any] | None] = [None] * len(paraphrases)

                missing_indices: list[int] = []
                missing_paraphrase_ids: list[str] = []
                missing_reference_ids: list[str] = []
                missing_paraphrases: list[str] = []
                missing_references: list[str] = []

                for i, (paraphrase_id, paraphrase, reference_id, reference) in enumerate(zip(paraphrase_ids, paraphrases, reference_ids, references)):
                    existing_scores_cursor = self.mongodb.find_document_by_multiple_fields(
                        collection=self.mongodb.paraphrase_score_collection,
                        search_args={"paraphrase_id": paraphrase_id, "reference_id": reference_id},
                    )
                    existing_result = list(existing_scores_cursor)
                    if existing_result:
                        group_results[i] = existing_result[0]
                    else:
                        missing_indices.append(i)
                        missing_paraphrase_ids.append(paraphrase_id)
                        missing_reference_ids.append(reference_id)
                        missing_paraphrases.append(paraphrase)
                        missing_references.append(reference)

                if missing_paraphrases:
                    # list of dicts
                    scores = self.metric_calculator.get_paraphrase_metrics(
                        paraphrases=missing_paraphrases,
                        original_texts=missing_references,
                    )
                    scores = [
                        {
                            **score,
                            "dataset_name": dataset_name,
                            "paraphrase_id": p_id,
                            "reference_id": r_id,
                        }
                        for score, p_id, r_id in zip(
                            scores, missing_paraphrase_ids, missing_reference_ids
                        )
                    ]

                    self.mongodb.insert_documents(
                        collection=self.mongodb.paraphrase_score_collection,
                        insert_data=scores,
                    )
                    for idx, score_row in zip(missing_indices, scores):
                        group_results[idx] = score_row

                assert all(row is not None for row in group_results)
                results.extend(group_results)

        df = pd.DataFrame(results)
        logger.info("Number of evaluated paraphrases: %d", len(results))
        # drop any columns that are completely empty, i.e. all NaN
        df.dropna(axis=1, how="all", inplace=True)
        df = self._enrich_results_with_source_context(results_df=df, source_df=source_df)
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

    def get_metric_names(self) -> List[str]:
        """
        Get the names of the metrics used in the evaluation.
        :return: A list of metric names.
        """
        return self.metric_calculator.get_metric_names()

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

    def plot_length_percentage_boxplot(
        self,
        df: pd.DataFrame,
        data_category: Optional[str] = None,
        group_by: Optional[str] = "paraphrase_technique",
        technique_label_mode: str = "model_prompt_approach",
        prompt_words: int = 6,
        display_plot: bool = True,
    ):
        self.plotter.plot_length_percentage_boxplot(
            df=df,
            data_category=data_category,
            group_by=group_by,
            technique_label_mode=technique_label_mode,
            prompt_words=prompt_words,
            display_plot=display_plot,
        )

    def plot_metric_boxplots_per_metric(
        self,
        df: pd.DataFrame,
        data_category: Optional[str] = None,
        group_by: Optional[str] = "paraphrase_technique",
        technique_label_mode: str = "model_prompt_approach",
        prompt_words: int = 6,
        display_plot: bool = True,
    ):
        self.plotter.plot_metric_boxplots_per_metric(
            df=df,
            metric_names=self.get_metric_names(),
            data_category=data_category,
            group_by=group_by,
            technique_label_mode=technique_label_mode,
            prompt_words=prompt_words,
            display_plot=display_plot,
        )
