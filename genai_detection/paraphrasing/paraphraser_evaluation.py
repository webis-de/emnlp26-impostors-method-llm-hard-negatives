import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, List, Optional
import nltk
from nltk.translate import bleu_score, meteor_score
import evaluate
import datetime
import difflib
from collections import defaultdict
from itertools import chain, cycle, product
from matplotlib import pyplot as plt
import numpy as np
import openai
import pandas as pd
import sklearn
from word_mover_distance import model  # https://pypi.org/project/word-mover-distance/
from sentence_transformers import SentenceTransformer
import gensim.downloader
import torch
import seaborn as sns
from tqdm import tqdm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from genai_detection.paraphrasing.paraphraser import (
    OllamaParaphraser,
    T5ChatGPTParaphraser,
    TopicParaphraser,
    TaskParaphraser,
    TitleParaphraser,
    BulletPointParaphraser,
    TranslationParaphraser,
    Paraphraser,
    NonNaiveParaphraser,
    NaiveParaphraser,
    TopicSchema,
    BulletSchema,
    TaskSchema,
    TitleSchema,
)

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from genai_detection.config import CONFIG
from genai_detection.util import preprocess_text as _preprocess_text

nltk.download("wordnet")  # necessary for METEOR score
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)


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
        paraphrasers: dict,
        prompts: List[str],
        original_text: str,
        n_responses: int = 3,
        max_length: int = CONFIG.MAX_LENGTH,
        temperature: float = CONFIG.TEMPERATURE,
        ground_truth: Optional[dict[str, Any]] = None,
        data_category: Optional[str] = None,
        original_file_name: Optional[str] = None,
    ):
        """
        Initializes the ParaphrasingEvaluator with the given paraphrasers and prompts.
        :param paraphrasers: A dictionary of paraphraser instances with their names as keys.
        :param prompts: A list of prompts to be used for paraphrasing.
        :param original_text: The original text to be paraphrased.
        :param n_responses: The number of paraphrases to generate for each paraphraser.
        :param max_length: The maximum length of the generated paraphrase.
        :param temperature: Controls the randomness of the output. Lower values make the output more deterministic.
        :param ground_truth: Optional ground truth data to compare against the generated paraphrases.
        :param data_category: Optional category of the data being evaluated, used for logging and saving results.
        :param original_file_name: Optional name of the original file, used for logging and saving results.
        """
        assert isinstance(paraphrasers, dict) and all(
            isinstance(p, Paraphraser) for p in paraphrasers.values()
        ), "paraphrasers must be a dictionary of Paraphraser instances."
        self.paraphrasers = paraphrasers
        assert isinstance(prompts, list) and all(
            isinstance(p, str) for p in prompts
        ), "prompts must be a list of strings."
        self.prompts = prompts
        assert (
            isinstance(original_text, str) and original_text.strip()
        ), "original_text must be a non-empty string."
        self.original_text = _preprocess_text(original_text)
        assert (
            isinstance(n_responses, int) and n_responses > 0
        ), "n_responses must be a positive integer."
        self.n_responses = n_responses
        assert (
            isinstance(max_length, int) and max_length > 0
        ), "max_length must be a positive integer."
        self.max_length = max_length
        self.temperature = temperature
        self.original_file_name = (
            str(original_file_name).replace("/", "_").replace(".", "_")
            if original_file_name
            else "unknown"
        )

        self.rouge_score = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        tries = 0
        self.sbert_model = None
        while not self.sbert_model and tries < 10:
            try:
                self.sbert_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"  # , device=device    # TODO: for server?
                )  # for cosine similarity
            except Exception as e:
                raise RuntimeError(
                    "Failed to load SentenceTransformer model: {e}"
                ) from e
                print(
                    "Failed to load SentenceTransformer model. Setting it to None. with error:",
                    e,
                )
                tries += 1
        # https://pypi.org/project/word-mover-distance/ Word Mover's Distance (WMD)
        print("Loading pre-trained word vectors for WMD...")
        tries = 0
        self.pretr_word_model = None
        while not self.pretr_word_model and tries < 10:
            try:
                self.pretr_word_model = WMDReadyKeyedVectors(
                    gensim.downloader.load("glove-twitter-25")
                )
            except Exception as e:
                print("Failed to load gensim glove-twitter-25. Retrying...")
                tries += 1
        self.wmd_model = model.WordEmbedding(model=self.pretr_word_model)
        self.base_dirs = {
            "blog": Path(__file__).resolve().parents[2] / "data/datasets/Blog_corpus/",
            "gutenberg": Path(__file__).resolve().parents[2]
            / "data/datasets/gutenberg/",
            "custom": Path(__file__).resolve().parents[2]
            / "data/datasets/custom_texts/",
            # TODO: Add student essays dataset
            # "student_essays": Path(__file__).resolve().parents[2]
            # / "data/datasets/student_essays/Intro2006/",
        }
        self.ground_truth = ground_truth or {}
        self.data_category = data_category or "unknown"
        self.paraphrases_save_base_path = (
            Path(__file__).resolve().parent.parent.parent
            / CONFIG.SAVE_PATH
            / "paraphrasing"
            / "experiments"
            / "paraphrase_evaluation"
            / self.data_category.replace(" ", "_").replace("/", "_")
        )
        self.paraphrases_save_base_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _degree_of_similarity(a: str, b: str) -> float:
        """Calculate similarity ratio between two strings (case-insensitive)."""
        return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

    def _similar(self, a, b, sim_thres: float = 0.65) -> bool:
        """Check if two inputs are sufficiently similar."""
        return self._degree_of_similarity(str(a), str(b)) > sim_thres

    @staticmethod
    def _get_century(time_period) -> int:
        """Convert a time period or year string/int to century."""
        if isinstance(time_period, str):
            if "present" in time_period.lower():
                return 21
            # Extract digits, fallback to 0 if none found or invalid
            try:
                time_period = int(re.sub(r"[^\d]", "", time_period))
            except ValueError:
                return 0
        if not isinstance(time_period, (int, float)) or time_period <= 0:
            return 0

        if time_period > 100:
            century = time_period // 100
            if time_period % 100 != 0:
                century += 1
            return int(century)
        return int(time_period)

    def _load_dataset(self, base_dir: Path, dataset_type: str) -> pd.DataFrame:
        """
        Load dataset texts and metadata (if available), preprocess and filter.
        dataset_type: 'blog', 'gutenberg', or 'custom'
        Returns a dataframe with all necessary columns.
        """
        if not base_dir.exists():
            raise FileNotFoundError(
                f"Base directory {base_dir} does not exist. Current path: {os.getcwd()}"
            )

        metadata = None
        data = []
        path2metadata = base_dir / "file_metadata.xlsx"
        if dataset_type != "blog":
            if not path2metadata.exists():
                raise FileNotFoundError(
                    f"Metadata file {path2metadata} does not exist. Current path: {os.getcwd()}"
                )
            metadata = pd.read_excel(path2metadata)

        logger.info(f"Loading data from {base_dir.name}...")

        for file in chain(base_dir.glob("*.txt"), base_dir.glob("*.csv")):
            if (
                dataset_type == "gutenberg"
                and "Complete_Works_of_William_Shakespeare" in file.name
            ):
                # Skip oversized text
                continue

            if file.suffix == ".csv" and dataset_type == "blog":
                df = pd.read_csv(file)
                df["text"] = df["text"].apply(_preprocess_text)
                df = df[df["text"].apply(lambda x: len(x.split()) >= 500)]

                df["year"] = (
                    pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
                    .dt.year.fillna(0)
                    .astype(int)
                )
                df["century"] = df["year"].apply(self._get_century)

                return df  # Blog dataset loaded directly as DataFrame

            elif file.suffix == ".txt":
                with open(file, "r", encoding="utf-8") as f:
                    author = " ".join(file.stem.split("_")[-2:])
                    content = _preprocess_text(f.read())
                    if len(content.split()) >= 500:
                        data.append(
                            {"author": author, "text": content, "filename": file.stem}
                        )
                    else:
                        logger.warning(
                            f"Skipping file {file.name} due to insufficient length."
                        )

        if metadata is not None:
            df = pd.DataFrame(data)
            df = df.join(
                metadata.set_index("filename"),
                on="filename",
                how="inner",
                rsuffix="_meta",
            )
        else:
            df = pd.DataFrame(data)

        df.dropna(how="all", inplace=True)
        logger.info(f"Loaded {len(df)} records from {base_dir.name}.")

        return df

    def _evaluate_model_on_data(
        self, paraphraser_name: str, paraphraser, df: pd.DataFrame
    ):
        """
        Evaluate a single paraphraser on the given dataset.
        Returns metrics dictionary and length difference list.
        """
        results_per_text = []
        lengths = {"original": [], "paraphrase": []}
        file_existing_paraphrases = (
            self.paraphrases_save_base_path
            / f"generated_paraphrases_subset_{self.data_category.replace(' ', '_')}.json"
        )
        if "century" not in df.columns and "date" in df.columns:
            # Blog dataset has 'date' column (eg. 12,May,2004), convert to 'century'
            df["century"] = df["date"].apply(
                lambda x: (
                    self._get_century(int(x.split(",")[2]))
                    if len(x.split(",")) > 2
                    else 0
                )
            )

        for row in tqdm(
            df.itertuples(), total=len(df), desc=f"Evaluating {paraphraser_name}"
        ):
            text = str(getattr(row, "text", ""))
            filename = getattr(row, "filename", "unknown")

            try:
                extra, _, genre, century, _, _ = paraphraser._extract_bullet_points(
                    text=text,
                    prompt=paraphraser.extractor_prompt,
                )
            except Exception as e:
                logger.warning(f"Extraction failed for file '{filename}': {e}")
                continue

            if (
                "current" in str(century).lower()
                or "present" in str(century).lower()
                or "now" in str(century).lower()
            ):
                century = 21
            else:
                century = self._get_century(century)
            gt_genre = getattr(row, "genre", "").lower() or ""
            gt_century = getattr(row, "century", 0) or 0
            gt_topic = getattr(row, "topic", "") or ""

            genre_match = any(
                self._similar(extr_g.strip().lower(), gt_genre)
                for extr_g in re.split(r"[ /,]+", str(genre).lower())
            )
            time_match = self._similar(century, gt_century)
            topic_match = np.max(
                [
                    self._degree_of_similarity(gt_sub_topic.strip(), extra)
                    for gt_sub_topic in str(gt_topic).lower().split(",")
                ]
            )

            # read existing paraphrase if available
            if file_existing_paraphrases.exists():
                with open(file_existing_paraphrases, "r") as f:
                    data_loaded = json.load(f)
            else:
                data_loaded = {}

            try:
                potential_paraphrases = data_loaded[paraphraser_name][text][None]
                paraphrases = []
                for temperature in potential_paraphrases.keys():
                    paraphrases.extend(potential_paraphrases[temperature])
            except Exception as e:
                print(
                    f"Paraphraser {paraphraser_name} not found in loaded data for {text[:15]}:\n{e}.\nGenerating new paraphrase."
                )
                # raise Exception(f"Should all be present, but not found: {e}") from e

                paraphrases = paraphraser.paraphrase(
                    text=text, temperature=self.temperature
                )

                if (
                    paraphrases
                    and isinstance(paraphrases, (list, tuple))
                    and len(paraphrases) > 0
                ):
                    # save generated paraphrases to file
                    data_loaded.setdefault(paraphraser_name, {})
                    data_loaded[paraphraser_name].setdefault(text, {})
                    data_loaded[paraphraser_name][text].setdefault(None, {})
                    data_loaded[paraphraser_name][text][None][
                        self.temperature
                    ] = paraphrases
                    with open(file_existing_paraphrases, "w") as f:
                        json.dump(data_loaded, f, indent=4)

            paraphrase_len = np.average([len(p.split()) for p in paraphrases])

            orig_len = len(text.split())
            lengths["original"].append(orig_len)
            lengths["paraphrase"].append(paraphrase_len)

            results_per_text.append(
                {
                    "filename": filename,
                    "genre_match": genre_match,
                    "time_match": time_match,
                    "topic_match": topic_match,
                    "original_length": orig_len,
                    "paraphrase_length": paraphrase_len,
                    "ground_truth_genre": gt_genre,
                    "ground_truth_century": gt_century,
                    "ground_truth_topic": gt_topic,
                    "extracted_topic": extra,
                    "extracted_genre": genre,
                    "extracted_century": century,
                }
            )

        # Convert to DataFrame
        results_df = pd.DataFrame(results_per_text)
        print(
            f"[DEBUG] Results DataFrame for {paraphraser_name} of len {len(results_df)}:\n{results_df.head()}"
        )

        # Optionally compute total scores
        summary = {
            "genre_match": results_df["genre_match"].sum(),
            "time_match": results_df["time_match"].sum(),
            "topic_match": results_df["topic_match"].sum(),
            "total": len(results_df),
        }

        return results_df, summary, lengths

    def plot_metric_radar_per_dataset(
        self,
        df_all: pd.DataFrame,
        metrics: list[str],
        save_path: Path | None = None,
        dataset_col: str = "dataset",
        display_plot: bool = True,
    ):
        metrics = [
            m
            for m in metrics
            if m in df_all.columns and pd.api.types.is_numeric_dtype(df_all[m])
        ]
        assert metrics, "No numeric metrics found to plot."
        grouped_mean = df_all.groupby(dataset_col)[metrics].mean()
        grouped_std = df_all.groupby(dataset_col)[metrics].std()

        # Compute angle of each axis
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        # Complete the loop
        angles += angles[:1]

        # Start plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        unique_labels = grouped_mean.index
        palette = sns.color_palette(
            "tab20" if len(unique_labels) > 10 else "tab10", n_colors=len(unique_labels)
        )
        label_to_color = {
            label: palette[i % len(palette)] for i, label in enumerate(unique_labels)
        }
        line_styles = [
            "solid",
            "dashed",
            "dotted",
            "dashdot",
            (0, (3, 1, 1, 1)),
            (0, (5, 1)),
        ]

        # groupby paraphraser model or prompt
        for i, groupby_value in enumerate(unique_labels):
            mean_values = grouped_mean.loc[groupby_value].tolist()
            std_values = grouped_std.loc[groupby_value].tolist()

            # Close the loop
            mean_values += mean_values[:1]
            std_values += std_values[:1]

            lower = np.maximum(0, np.array(mean_values) - np.array(std_values))
            upper = np.minimum(1, np.array(mean_values) + np.array(std_values))

            ax.plot(
                angles,
                mean_values,
                label=self._wrap_label(groupby_value),
                alpha=0.7,
                color=label_to_color[groupby_value],
                linewidth=2,
                linestyle=line_styles[i % len(line_styles)],
            )
            ax.fill_between(
                angles, lower, upper, color=label_to_color[groupby_value], alpha=0.2
            )

        # Add labels to axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.tick_params(axis="y")

        # Add legend and title
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(1.1, 0.7),
            fontsize=10,
            title=dataset_col.capitalize(),
        )

        title = "Radar plot of Metric Distributions by Dataset"
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.85])

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            for format in ["svg"]:
                out = save_path / f"radar_extraction_quality_per_dataset.{format}"
                fig.savefig(out, bbox_inches="tight", transparent=True, format=format)
                print(f"Saved radar plot to {out}")

        else:
            print("No save path provided, plot not saved.")
        if display_plot:
            plt.show()
        plt.close()

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
        # TODO: To debug plotting, remove in production
        # models = {
        #     k: v
        #     for k, v in self.paraphrasers.items()
        #     if not isinstance(v, NaiveParaphraser)
        # }

        # for dataset_category, base_dir in self.base_dirs.items():
        #     try:
        #         df = self._load_dataset(base_dir, dataset_category)
        #     except Exception as e:
        #         logger.error(
        #             f"Failed to load dataset {dataset_category} from {base_dir}: {e}"
        #         )
        #         continue

        #     df = df.head(min(2, len(df)))  # TODO: For debugging, remove in production
        #     if "id" in df.columns:
        #         df.rename(columns={"id": "filename"}, inplace=True)
        #     logger.info(f"Dataset snapshot:\n{df.head()}")
        #     print(f"Dataset snapshot:\n{df.head()}")

        #     aggregate_results = defaultdict(
        #         lambda: {
        #             "genre_match": 0,
        #             "time_match": 0,
        #             "topic_match": 0,
        #             "total": 0,
        #         }
        #     )
        #     length_differences = {}

        #     for paraphraser_name, paraphraser in models.items():
        #         logger.info(
        #             f"Evaluating model '{paraphraser_name}' on {dataset_category} dataset..."
        #         )
        #         detailed_result_df, summary_df, lengths = self._evaluate_model_on_data(
        #             paraphraser_name, paraphraser, df
        #         )
        #         print(
        #             f"[DEBUG] Detailed results for {paraphraser_name}:\n{detailed_result_df.head()}"
        #         )

        #         # Calculate length differences as percentages
        #         percent_diffs = [
        #             ((p - o) / o) * 100 if o > 0 else 0
        #             for o, p in zip(lengths["original"], lengths["paraphrase"])
        #         ]
        #         length_differences[paraphraser_name] = percent_diffs

        #         for key in ["genre_match", "time_match", "topic_match", "total"]:
        #             aggregate_results[paraphraser_name][key] += summary_df.get(key, 0)

        #     # Reporting results
        #     logger.info(f"\n[RESULTS for {dataset_category} dataset]")
        #     for paraphraser_name, metrics in aggregate_results.items():
        #         total = metrics["total"]
        #         if total == 0:
        #             logger.warning(
        #                 f"No evaluation data for model '{paraphraser_name}' on dataset '{dataset_category}'."
        #             )
        #             continue
        #         for key in ["genre_match", "time_match", "topic_match"]:
        #             aggregate_results[paraphraser_name][key] = round(
        #                 metrics[key] / total, 2
        #             )
        #         aggregate_results[paraphraser_name]["length_diff"] = round(
        #             np.mean(length_differences[paraphraser_name]), 2
        #         )

        #         logger.info(f"Model: {paraphraser_name}")
        #         logger.info(
        #             f"  Genre Accuracy: {aggregate_results[paraphraser_name]['genre_match']}"
        #         )
        #         logger.info(
        #             f"  Time Accuracy (approx): {aggregate_results[paraphraser_name]['time_match']}"
        #         )
        #         logger.info(
        #             f"  Topic Accuracy (approx): {aggregate_results[paraphraser_name]['topic_match']}"
        #         )
        #         logger.info(
        #             f"  Length Difference (mean): {aggregate_results[paraphraser_name]['length_diff']}%"
        #         )

        #     # Save results if requested
        #     if save_to_disk:
        #         self._save_results(
        #             detailed_result_df,
        #             dataset_category,
        #             save_base_path,
        #             detail_degree=detailed_detail_degree,
        #         )
        #         self._save_results(
        #             aggregate_results,
        #             dataset_category,
        #             save_base_path,
        #             detail_degree="aggregated",
        #         )
        #         print("Saved results to ", save_base_path)

        if plot_metrics:
            dfs = {}
            print("Read results from disk for plotting from ", save_base_path)
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
            print(
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
            if isinstance(paraphraser, NonNaiveParaphraser) and self.ground_truth:
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

    def _safe_compute_bertscore(self, paraphrases, references):
        try:
            return self.bertscore.compute(
                predictions=paraphrases,
                references=references,
                model_type="distilbert-base-uncased",
            )
        except Exception as e:
            logger.error(f"BERTScore computation failed: {e}")
            return {
                "precision": [0.0] * self.n_responses,
                "recall": [0.0] * self.n_responses,
                "f1": [0.0] * self.n_responses,
                "hashcode": "",
            }

    def _safe_compute_rouge(self, paraphrases):
        try:
            return [
                self.rouge_score.compute(
                    predictions=[p], references=[self.original_text]
                )
                for p in paraphrases
            ]
        except Exception as e:
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
        results = []
        references = [self.original_text] * self.n_responses
        original_split = self.original_text.split()

        # Non-naive paraphrasers take the same prompt, but are called with different temperatures to introduce variance
        temperatures = list(np.linspace(0, 1, max(2, len(self.prompts)), endpoint=True))

        # Cyclic iterator over temperatures
        temp_cycle = cycle(temperatures)

        test_configurations = []
        for paraphraser_name, paraphraser in self.paraphrasers.items():
            if isinstance(paraphraser, NaiveParaphraser):
                # naive paraphrasers: fixed temperature + prompt diversity
                for prompt_id, prompt in enumerate(self.prompts, start=1):
                    test_configurations.append(
                        (
                            (paraphraser_name, paraphraser),
                            prompt,
                            self.temperature,
                            prompt_id,
                        )
                    )
            else:
                # Non-naive: fixed prompt diversity + temperature diversity
                for _ in self.prompts:
                    test_configurations.append(
                        ((paraphraser_name, paraphraser), None, next(temp_cycle), 0)
                    )

        print(
            f"[DEBUG] Total configurations to evaluate: {len(test_configurations)}, {test_configurations}"
        )
        logger.info(
            f"[DEBUG] Total configurations to evaluate: {len(test_configurations)}, {test_configurations}"
        )
        results = []
        # load json object with existing paraphrases if available and append new ones
        paraphrase_file_path = self.paraphrases_save_base_path
        for (paraphraser_name, paraphraser), prompt, temperature, prompt_id in tqdm(
            test_configurations,
            desc="Evaluating Paraphrasers",
            total=len(test_configurations),
        ):
            # path2file = (
            #     self.paraphrases_save_base_path
            #     / f"{name}_paraphrases_temp{temperature}_prompt{prompt_id}_{self.original_file_name}.csv"
            # )
            try:
                paraphrases = self._load_or_generate_paraphrases(
                    paraphraser_name,
                    paraphraser,
                    prompt,
                    temperature,
                    paraphrase_file_path,
                )
            except Exception as e:
                logger.error(
                    f"Paraphraser '{paraphraser_name}' with prompt '{prompt}' failed: {e}"
                )
                paraphrases = [""] * self.n_responses

            bert_scores = self._safe_compute_bertscore(paraphrases, references)
            rouge_scores = self._safe_compute_rouge(paraphrases)

            try:
                for i, paraphrase in enumerate(paraphrases):
                    results.append(
                        self._build_result_row(
                            paraphraser_name,
                            prompt,
                            paraphrase,
                            original_split,
                            bert_scores,
                            rouge_scores[i],
                            i,
                        )
                    )

            except Exception as e:
                print(
                    f"[ERROR] Scoring failed for '{paraphraser_name}' with prompt '{prompt}' for category '{self.data_category}' and paraphrases '{paraphrases}': {e}"
                )
                continue

        df = pd.DataFrame(results)
        # drop any columns that are completely empty, i.e. all NaN
        df.dropna(axis=1, how="all", inplace=True)
        if save_to_disk:
            save_path = (
                self.paraphrases_save_base_path
                / f"paraphrasing_results_comparison_temp{self.temperature}_maxLength{self.max_length}_dataset_{self.data_category.replace(' ','_')}.csv"
            )
            df.to_csv(save_path, index=False, float_format="%.4f")
            print(f"Results saved to {save_path}")

        # Save the worst and best paraphrase per score
        extremest_paraphrases = pd.DataFrame()
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
            worst_save_path = (
                self.paraphrases_save_base_path
                / f"extremest_paraphrases_per_metric_temp{self.temperature}_maxLength{self.max_length}_dataset_{self.data_category.replace(' ','_')}.csv"
            )
            extremest_paraphrases.to_csv(worst_save_path, index=False)

        return df, extremest_paraphrases

    def _build_result_row(
        self,
        paraphraser_name: str,
        prompt: str,
        paraphrase: str,
        original_split: List[str],
        bert_scores: dict,
        rouge_scores: dict,
        idx: int,
    ) -> dict:
        """
        Build a result row for the DataFrame.
        :param paraphraser_name: Name of the paraphraser.
        :param prompt: The prompt used for paraphrasing excluding the text to paraphrase and tailoring whitespaces, but including bulletpoints etc.
        :param paraphrase: One of the generated paraphrase.
        :param original_split: The original text split into tokens.
        :param bert_scores: BERTScore results.
        :param rouge_score: ROUGE scores for the paraphrase.
        :param idx: Index of the paraphrase in the list of BERTScores.
        :return: A dictionary representing the result row.
        """
        # in [-1, 1] range, where 1 is identical, 0 is no similarity, -1 is opposite
        if self.sbert_model:
            cos_sim = torch.cosine_similarity(
                self.sbert_model.encode(self.original_text, convert_to_tensor=True),
                self.sbert_model.encode(paraphrase, convert_to_tensor=True),
                dim=0,
            ).item()
        else:
            cos_sim = None
        res = {
            "model": paraphraser_name,
            "prompt": f"{prompt} <TEXT>",
            "parameters": {
                "n_responses": self.n_responses,
                "max_tokens": self.max_length,
                "temperature": self.temperature,
            },
            "original_text": self.original_text,
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
        """
        Plot the performance of models per metric.

        :param df: DataFrame containing the evaluation results.
        :param metric_name: The name of the metric to plot.
        :param data_category: Optional category of the data, used for the plot title.
        :param group_by: The column to group the data by (default is 'model'). Alternatives could be 'prompt'.
        :param display_plot: Whether to display the plot (default is True).
        :return: A list of matplotlib figures.
        """
        save_path = self.paraphrases_save_base_path / "radar_charts"
        save_path.mkdir(parents=True, exist_ok=True)
        # Enforce fixed metric order
        all_labels = self.get_metric_names()
        labels = [metric for metric in all_labels if metric in df.columns]
        assert (
            group_by in df.columns
        ), f"Group by column '{group_by}' not found in DataFrame."
        if group_by == "model" and "Paraphraser" not in df.columns:
            data = df.rename(columns={group_by: "Paraphraser"}, inplace=False)
            group_by = "Paraphraser"
        else:
            data = df.copy()
        grouped_mean = data.groupby(group_by)[labels].mean()
        grouped_std = data.groupby(group_by)[labels].std()

        # Compute angle of each axis
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        # Complete the loop
        angles += angles[:1]

        # Start plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        unique_labels = grouped_mean.index
        palette = sns.color_palette(
            "tab20" if len(unique_labels) > 10 else "tab10", n_colors=len(unique_labels)
        )
        label_to_color = {
            label: palette[i % len(palette)] for i, label in enumerate(unique_labels)
        }

        # groupby paraphraser model or prompt
        for groupby_value in unique_labels:
            mean_values = grouped_mean.loc[groupby_value].tolist()
            std_values = grouped_std.loc[groupby_value].tolist()

            # Close the loop
            mean_values += mean_values[:1]
            std_values += std_values[:1]

            lower = np.maximum(0, np.array(mean_values) - np.array(std_values))
            upper = np.minimum(1, np.array(mean_values) + np.array(std_values))

            ax.plot(
                angles,
                mean_values,
                label=self._wrap_label(groupby_value),
                alpha=0.7,
                color=label_to_color[groupby_value],
            )
            ax.fill_between(
                angles, lower, upper, color=label_to_color[groupby_value], alpha=0.2
            )

        # Add labels to axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

        # Optional: Set value range
        ax.set_ylim(0, 1)

        # Add legend and title
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(1.1, 0.7),
            fontsize=9,
            title=group_by.capitalize(),
        )
        title = (
            f"Radar Chart of Paraphrasing Metrics\non {' '.join(word.capitalize() for word in data_category.split())} Dataset, grouped by {group_by.capitalize()}"
            if data_category
            else f"Radar Chart of Paraphrasing Metrics\ngrouped by {group_by.capitalize()}"
        )
        plt.title(title, fontsize=12)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            for format in ["svg"]:
                file_name = (
                    save_path
                    / f"{data_category.replace(' ','_')}_paraphrasing_metrics_grouped_by_{group_by}_radar_chart.{format}"
                )
                plt.savefig(
                    file_name, bbox_inches="tight", transparent=True, format=format
                )
                print(f"Plot saved to {file_name}")
        if display_plot:
            plt.show()
        plt.close()

    def plot_metric_scatter(
        self,
        df: pd.DataFrame,
        data_category: Optional[str] = None,
        group_by: Optional[str] = "model",
        display_plot: bool = True,
    ):
        """
        Scatter plot of semantic similarity vs syntactic similarity per model.

        :param df: DataFrame with at least 'sem_sim_avg', 'syn_sim_avg', and 'model' columns.
        :param data_category: Optional category of the data, used for the plot title.
        :param group_by: The column to group the data by (default is 'model'). Alternatives could be 'prompt'.
        :param display_plot: Whether to display the plot (default is True).
        :return: matplotlib Figure object.
        """
        save_path = self.paraphrases_save_base_path / "metric_scatter"
        save_path.mkdir(parents=True, exist_ok=True)
        if group_by == "model" and "Paraphraser" not in df.columns:
            data = df.rename(columns={group_by: "Paraphraser"}, inplace=False)
            group_by = "Paraphraser"
        else:
            data = df.copy()
        required_cols = ["sem_sim_avg", "syn_sim_avg", group_by]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

        # Calculate min and max with padding
        x_min, x_max = data["sem_sim_avg"].min(), data["sem_sim_avg"].max()
        y_min, y_max = data["syn_sim_avg"].min(), data["syn_sim_avg"].max()

        x_pad = (x_max - x_min) * 0.05 if (x_max - x_min) > 0 else 0.05
        y_pad = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.05

        x_lim = (max(0, x_min - x_pad), min(1, x_max + x_pad))
        y_lim = (max(0, y_min - y_pad), min(1, y_max + y_pad))

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        unique_labels = data[group_by].unique()
        palette = sns.color_palette("tab20", n_colors=len(unique_labels))
        label_to_color = {
            label: palette[i % len(palette)] for i, label in enumerate(unique_labels)
        }
        sns.scatterplot(
            data=data,
            x="sem_sim_avg",
            y="syn_sim_avg",
            hue=group_by,
            palette=label_to_color,
            alpha=0.7,
            s=100,
            edgecolor="k",
            ax=ax,
        )

        ax.set_xlabel("Semantic Similarity (sem_sim_avg)")
        ax.set_ylabel("Syntactic Similarity (syn_sim_avg)")
        title = (
            f"Semantic vs Syntactic Similarity\non {' '.join(word.capitalize() for word in data_category.split())} Dataset, grouped by {group_by.capitalize()}"
            if data_category
            else f"Semantic vs Syntactic Similarity\ngrouped by {group_by.capitalize()}"
        )
        ax.set_title(title)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(True)

        # Place legend outside the plot on the right
        legend_patches = [
            mpatches.Patch(color=color, label=self._wrap_label(label=label))
            for label, color in label_to_color.items()
        ]
        ax.legend(
            handles=legend_patches,
            title=group_by.capitalize(),
            loc="upper left",
            bbox_to_anchor=(1.03, 1),
            borderaxespad=0.0,
            frameon=True,
            fontsize=9,
        )

        # Inset with full range
        inset_size = 0.25
        inset_ax = fig.add_axes([0.9, 0.1, inset_size, inset_size])

        sns.scatterplot(
            data=data,
            x="sem_sim_avg",
            y="syn_sim_avg",
            hue=group_by,
            palette="tab20",
            alpha=0.7,
            s=40,
            edgecolor="k",
            legend=False,  # No legend on inset
            ax=inset_ax,
        )

        inset_ax.set_xlim(0, 1)
        inset_ax.set_ylim(0, 1)
        inset_ax.set_title("Full range")
        inset_ax.grid(True)
        inset_ax.set_xticks([0, 0.5, 1])
        inset_ax.set_yticks([0, 0.5, 1])
        inset_ax.tick_params(axis="both", which="major", labelsize=8)

        # Adjust layout to leave 25% room for legend/inset on the right
        # plt.tight_layout(rect=[0, 0, 0.75, 1])

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            for format in ["svg"]:
                full_path = (
                    save_path
                    / f"{data_category.replace(' ','_')}_sem_syn_scatter_grouped_by_{group_by}.{format}"
                )
                plt.savefig(
                    full_path, bbox_inches="tight", transparent=True, format=format
                )
                print(f"Plot saved to {full_path}")

        if display_plot:
            plt.show()
        plt.close()

    def _wrap_label(self, label: str, words_per_line: int = 6) -> str:
        assert (
            isinstance(words_per_line, int) and words_per_line > 0
        ), "words_per_line must be a positive integer."
        words = str(label).split()
        return "\n".join(
            [
                " ".join(words[i : i + words_per_line])
                for i in range(0, len(words), words_per_line)
            ]
        )

    def _plot_one_plot_per_metric_distribution(
        self,
        data: pd.DataFrame,
        metric_names: list,
        group_by: str,
        data_category: str,
        display_plot: bool = False,
    ):
        unique_labels = data[group_by].unique()
        palette = sns.color_palette("tab20", n_colors=len(unique_labels))

        for metric in metric_names:
            sns.kdeplot(
                data=data,
                x=metric,
                hue=group_by,
                fill=True,
                common_norm=False,
                alpha=0.4,
                palette=palette,
                legend=True,
            )

            metric_for_tile = " ".join([t.capitalize() for t in metric.split("_")])
            if data_category:
                plt.title(
                    f"Distribution of {metric_for_tile}\non {data_category} Dataset, grouped by {group_by.capitalize()}"
                )
            else:
                plt.title(
                    f"Distribution of {metric_for_tile}\ngrouped by {group_by.capitalize()}"
                )
            plt.xlabel(metric)
            plt.ylabel("Density")
            handles = [
                mpatches.Patch(color=palette[i], label=self._wrap_label(str(label)))
                for i, label in enumerate(unique_labels)
            ]
            plt.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                title=group_by.capitalize(),
                fontsize=10,
                title_fontsize=12,
            )

            plt.tight_layout()

            # Save with unique name
            safe_metric = str(metric).replace(" ", "_").replace("/", "_")
            safe_category = (
                str(data_category).replace(" ", "_") if data_category else "dataset"
            )
            file_name = f"{safe_category}_{safe_metric}_grouped_by_{group_by}.svg"
            path2dir = (
                self.paraphrases_save_base_path
                / "metric_distributions"
                / "distribution_per_metric"
            )
            path2dir.mkdir(parents=True, exist_ok=True)
            full_path = path2dir / file_name
            plt.savefig(full_path, bbox_inches="tight", transparent=True, format="svg")
            print(f"Plot saved to {full_path}")

            if display_plot:
                plt.show()
            plt.close()

    def plot_metric_distributions(
        self,
        df: pd.DataFrame,
        data_category: Optional[str] = None,
        group_by: Optional[str] = "model",
        display_plot: bool = True,
    ):
        """
        Plot distribution of each metric per model in subplots.

        :param df: DataFrame containing metric scores and a 'model' column.
        :param data_category: Optional string for plot title context.
        :param group_by: The column to group the data by (default is 'model'). Alternatives could be 'prompt'.
        :param display_plot: Whether to display the plot (default is True).
        :return: None
        """
        save_path = self.paraphrases_save_base_path / "metric_distributions"
        save_path.mkdir(parents=True, exist_ok=True)
        metric_names = [
            metric for metric in self.get_metric_names() if metric in df.columns
        ]
        if group_by == "model" and "Paraphraser" not in df.columns:
            data = df.rename(columns={group_by: "Paraphraser"}, inplace=False)
            group_by = "Paraphraser"
        else:
            data = df.copy()
        assert len(metric_names) > 0, "No valid metrics found in DataFrame."
        assert (
            group_by in data.columns
        ), f"Group by column '{group_by}' not found in DataFrame."

        n_metrics = len(metric_names)
        n_cols = 2
        n_rows = (n_metrics + 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        unique_labels = data[group_by].unique()
        max_words_in_label = max(len(str(label).split()) for label in unique_labels)
        use_shared_legend = max_words_in_label > 3 or len(unique_labels) > 5
        palette = sns.color_palette("tab20", n_colors=len(unique_labels))
        label_to_color = {
            label: palette[i % len(palette)] for i, label in enumerate(unique_labels)
        }

        for i, metric in enumerate(metric_names):
            ax = axes[i]

            # Find models with only one data point for this metric
            counts = data.groupby(group_by)[metric].count()
            assert (
                counts != 0
            ).any(), (
                f"No data points found for metric '{metric}' in group '{group_by}'."
            )

            # Models with multiple entries (for KDE)
            models_multi = counts[counts > 1].index
            # Models with single entry (for scatter)
            models_single = counts[counts == 1].index

            # Plot KDE for models with multiple points
            if len(models_multi) > 0:
                sns.kdeplot(
                    data=data[data[group_by].isin(models_multi)],
                    x=metric,
                    hue=group_by,
                    fill=True,
                    common_norm=False,
                    alpha=0.4,
                    ax=ax,
                    palette=label_to_color,
                    legend=False,
                )

            # Scatter for models with a single point
            for model in models_single:
                single_val = data[(data[group_by] == model)][metric].values[0]
                label = model if not use_shared_legend else None
                color = label_to_color[model]
                ax.scatter(
                    single_val,
                    1,
                    label=label,
                    color=color,
                    s=50,
                    edgecolor="k",
                    zorder=5,
                )
            metric_for_tile = " ".join([t.capitalize() for t in metric.split("_")])
            ax.set_title(f"Distribution of {metric_for_tile}")
            ax.set_xlim(0, 1)  # assuming similarity metrics in [0, 1]
            ax.set_xlabel(metric_for_tile)
            ax.set_ylabel("Density")

        # if use_shared_legend:
        legend_patches = [
            mpatches.Patch(color=color, label=self._wrap_label(label))
            for label, color in label_to_color.items()
        ]
        fig.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),  # outside the plot on right
            title=group_by.capitalize(),
            frameon=True,
            borderaxespad=0,
            fontsize=10,
            title_fontsize=12,
        )

        # Remove unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        title = (
            f"Metric Distributions\non {' '.join(word.capitalize() for word in data_category.split())} Dataset, grouped by {group_by.capitalize()}"
            if data_category
            else f"Metric Distributions\ngrouped by {group_by.capitalize()}"
        )
        fig.suptitle(title, fontsize=18, y=0.95)
        plt.tight_layout(rect=[0, 0, 0.85, 0.90])

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            for format in ["svg"]:
                full_path = (
                    save_path
                    / f"{data_category.replace(' ','_')}_metric_distributions_grouped_by_{group_by}.{format}"
                )
                plt.savefig(
                    full_path, bbox_inches="tight", transparent=True, format=format
                )
                print(f"Plot saved to {full_path}")

        if display_plot:
            plt.show()
        plt.close()
        self._plot_one_plot_per_metric_distribution(
            data=data,
            metric_names=metric_names,
            group_by=group_by,
            data_category=data_category,
            display_plot=False,
        )


if __name__ == "__main__":
    # models
    ollama_model_id = "zephyr:7b"  # "mistral:7b"  # "default:latest"
    paraphrasers = {
        "T5_ChatGPT": T5ChatGPTParaphraser(),
        # 'T5_Google_PAWS': T5GooglePAWSParaphraser(),
        # 'Blablador': BlabladorParaphraser(model_id="1 - Llama3 405 the best general model and big context size"),
        "Ollama": OllamaParaphraser(model_id=ollama_model_id),
        "TopicParaphraser": TopicParaphraser(
            text_extractor=OllamaParaphraser(model_id=ollama_model_id),
            text_generator=OllamaParaphraser(model_id=ollama_model_id),
        ),
        # "TaskParaphraser": TaskParaphraser(
        #     text_extractor=OllamaParaphraser(model_id=ollama_model_id),
        #     text_generator=OllamaParaphraser(model_id=ollama_model_id),
        # ),
        # "TitleParaphraser": TitleParaphraser(
        #     text_extractor=OllamaParaphraser(model_id=ollama_model_id),
        #     text_generator=OllamaParaphraser(model_id=ollama_model_id),
        # ),
        # "BulletPointParaphraser": BulletPointParaphraser(
        #     text_extractor=OllamaParaphraser(model_id=ollama_model_id),
        #     text_generator=OllamaParaphraser(model_id=ollama_model_id),
        # ),
        # "TranslationParaphraser": TranslationParaphraser(
        #     text_extractor=OllamaParaphraser(model_id=ollama_model_id),
        #     text_generator=OllamaParaphraser(model_id=ollama_model_id),
        # ),
    }
    n_responses = 3  # number of paraphrases to generate
    max_length = CONFIG.MAX_LENGTH  # Maximum length of the generated paraphrase
    temperature = (
        CONFIG.TEMPERATURE
    )  # Controls the randomness of the output. Lower values make the output more deterministic.
    evaluator = ParaphrasingEvaluator(
        paraphrasers=paraphrasers,
        prompts=[
            "Paraphrase the following text and output only the paraphrased version:"
        ],  # use a single prompt for simplicity
        original_text="This is a sample text to be paraphrased.",
        n_responses=n_responses,
        max_length=max_length,
        temperature=temperature,
    )
    print("Starting evaluation of paraphrasers...")
    evaluator.evaluate_extractors(save_to_disk=True)
    print("Evaluation complete.")
    # paraphrase_evaluator = ParaphrasingEvaluator(paraphrasers=paraphrasers, prompts=prompts, original_text=original_text, n_responses=n_responses, max_length=max_length, temperature=temperature)
    # paraphrase_evaluator.evaluate()
