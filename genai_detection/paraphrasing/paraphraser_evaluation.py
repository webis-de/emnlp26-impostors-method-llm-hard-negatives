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
import logging
from collections import defaultdict
from itertools import chain, cycle, zip_longest
from pathlib import Path

import chardet
import evaluate
import gensim.downloader
import matplotlib.patches as mpatches
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from bson import ObjectId
from matplotlib import pyplot as plt
from nltk.translate import bleu_score, meteor_score
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from word_mover_distance import model  # https://pypi.org/project/word-mover-distance/

from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.paraphrasing.one_step_paraphrasers import *
from genai_detection.paraphrasing.two_step_paraphrasers import *
from genai_detection.util import preprocess_text as _preprocess_text

nltk.download("wordnet")  # necessary for METEOR score
logger = logging.getLogger(__name__)

# FIXME: Restructure code to get extracted information from mongoDB instead of re-extracting it here.
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
            century += 1
            return int(century)
        return int(time_period)

    def _create_student_essays_metadata(self, base_dir: Path):
        task_description = {
            "Ass1": "Stream of consciousness, personal journal, diary, essay",
            "Ass2": "childhood, personal journal, diary, essay",
            "Ass3": "personality, essay, reflection, self-analysis",
            "Ass4": "Thematic Apperception Test, essay",
            "Ass5": "different theories, essay",  # not used in Koppel et al. (2014)
        }
        rows = []
        files_by_ass = []

        # Gather valid files for each assignment
        for assignment in list(task_description.keys()):
            files = [
                file for file in (base_dir / assignment).glob("*.txt") if file.is_file()
            ]
            files_by_ass.append(files)

        # Interleave: take one from each assignment in turn
        interleaved_files = []
        for group in zip_longest(*files_by_ass):
            for f in group:
                if f is not None:
                    interleaved_files.append(f)

        # Now process in interleaved order
        rows = []
        for file in interleaved_files:
            with open(file, "rb") as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected["encoding"]
            text = raw_data.decode(encoding)
            if len(text.split()) < 700:
                continue
            filename = file.stem
            assignment = file.parent.name
            if assignment == "Ass5" or assignment not in task_description:
                continue
            author = (
                filename
                if assignment != "Ass1" or "2006_" not in filename
                else filename.split("_")[-1]
            )
            metadata = {
                "filename": filename,
                "author": author,
                "topic": task_description.get(assignment, "Unknown"),
                "text": _preprocess_text(text),
                "year": "2006",
                "genre": "Essay",
                "century": 21,
            }
            rows.append(metadata)
        metadata_df = pd.DataFrame(rows)
        metadata_df.to_excel(base_dir / "file_metadata.xlsx", index=False)
        return metadata_df

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
        if dataset_type == "student_essays":
            if not path2metadata.exists():
                self._create_student_essays_metadata(base_dir)
            return pd.read_excel(path2metadata)

        elif dataset_type in ["gutenberg", "custom"]:
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
                continue

            if file.suffix == ".csv" and dataset_type == "blog":
                df = pd.read_csv(file)
                df["text"] = df["text"].apply(_preprocess_text)
                df = df[df["text"].apply(lambda x: len(x.split()) >= 700)]

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
                    if len(content.split()) >= 700:
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

    def _obtain_complete_paraphrase_df_from_mongodb(self):
        # contains paraphrase, prompt, temperature, text_id, extracted_info, length_original_text, length_paraphrased_text
        paraphrases_cursor = self.mongodb.non_naive_paraphrase_collection.find({})
        paraphrases = pd.DataFrame(paraphrases_cursor)
        # flatten extracted_info into columns
        extracted_df = pd.json_normalize(paraphrases["extracted_info"])

        # drop nested column and concat
        paraphrases = pd.concat(
            [paraphrases.drop(columns=["extracted_info"]), extracted_df],
            axis=1
        )

        # obtain original text via text_id (which is _id in original_texts collection)
        paraphrases["text_id"] = paraphrases["text_id"].map(ObjectId)
        original_texts_cursor = self.mongodb.original_collection.find(
            {"_id": {"$in": paraphrases["text_id"].tolist()}},  # $in matches any ID in the list
            {"_id": True, "text": True, "dataset": True},
        )
        original_texts = pd.DataFrame(original_texts_cursor)
        logger.info(f"Obtained {len(original_texts)} original texts from df with columns {original_texts.columns}")
        print(original_texts["dataset"].value_counts(dropna=False))

        # merge data on paraphrases' text_id and original_texts_cursor's _id field
        df = paraphrases.merge(
            original_texts,
            left_on="text_id",
            right_on="_id",
            how="left"
        )
        df.rename(columns={"_id_x": "_id"}, inplace=True)    # renamed due to merging; _id is paraphrase ID
        df.drop(columns=["_id_y"], inplace=True)            # drop _id_y because we have text_id
        return df

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
        metrics = [
            m
            for m in metrics
            if m in df_all.columns and pd.api.types.is_numeric_dtype(df_all[m])
        ]
        assert metrics, "No numeric metrics found to plot."
        grouped_mean = df_all.groupby(dataset_col)[metrics].mean().round(2)
        grouped_std = df_all.groupby(dataset_col)[metrics].std().round(2)
        grouped_median = df_all.groupby(dataset_col)[metrics].median().round(2)

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            mean_std_df = pd.concat(
                {"mean": grouped_mean, "std": grouped_std, "median": grouped_median},
                axis=1,
            )
            csv_out = save_path / "extraction_metrics_mean_std_median_per_dataset.csv"
            mean_std_df.to_csv(csv_out)
            logging.info(f"Saved mean/std values as CSV file to {csv_out}")

        # Compute angle of each axis
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        # Complete the loop
        angles += angles[:1]

        # Start plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

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

            lower = np.maximum(-1, np.array(mean_values) - np.array(std_values))
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
            bbox_to_anchor=(1.1, 0.8),
            fontsize=10,
            title=dataset_col.capitalize(),
        )

        title = "Radar plot of Metric Distributions by Dataset"
        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            for format in ["svg"]:
                out = save_path / f"radar_extraction_quality_per_dataset.{format}"
                fig.savefig(out, bbox_inches="tight", transparent=True, format=format)
                logging.info(f"Saved radar plot to {out}")

        else:
            logging.info("No save path provided, plot not saved.")
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
        print(df["dataset"].value_counts(dropna=False))

        for dataset_name in df["dataset"].unique():
            df_dataset = df[df["dataset"] == dataset_name]
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
                            row["dataset"] = dataset_name
                            row["paraphrase_id"] = paraphrase_id
                            row["reference_id"] = reference_id
                            logger.info(f"Computed scores for {paraphrase_id} and {reference_id}: {row}")
                            self.mongodb.insert_document(collection=self.mongodb.paraphrase_score_collection, insert_data=row)
                            logger.info(f"Inserted {row}")
                        else:
                            logger.info(f"Existing scores for {paraphrase_id} and {reference_id}")
                            row = existing_result[0]
                        results.append(row)

                except Exception as e:
                    logging.exception(
                        f"Scoring failed for '{paraphraser_name}' with prompt '{prompt}' for category '{self.data_category}': {e}"
                    )
                    logger.error("Error in evaluate function")
                    continue

        df = pd.DataFrame(results)
        logger.info("%s are complete results.", results)
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

        # do not set value range for x-axis -> results barely visible
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
            save_path.mkdir(parents=True, exist_ok=True)
            for format in ["svg"]:
                file_name = (
                    save_path
                    / f"{data_category.replace(' ', '_')}_paraphrasing_metrics_grouped_by_{group_by}_radar_chart.{format}"
                )
                plt.savefig(
                    file_name, bbox_inches="tight", transparent=True, format=format
                )
                logging.info(f"Plot saved to {file_name}")
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
            save_path.mkdir(parents=True, exist_ok=True)
            for format in ["svg"]:
                full_path = (
                    save_path
                    / f"{data_category.replace(' ', '_')}_sem_syn_scatter_grouped_by_{group_by}.{format}"
                )
                plt.savefig(
                    full_path, bbox_inches="tight", transparent=True, format=format
                )
                logging.info(f"Plot saved to {full_path}")

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
            for scale in ["linear", "symlog"]:
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
                plt.yscale(scale)
                if scale == "symlog":
                    plt.grid(which="both", linestyle="--", linewidth=0.5)

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
                min_val = data[metric].min()
                max_val = data[metric].max()
                plt.xlim(left=max(0, min_val), right=min(1, max_val))
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
                file_name = f"{safe_category}_{safe_metric}_grouped_by_{group_by}_{scale}_scale.svg"
                path2dir = (
                    self.paraphrases_save_base_path
                    / "metric_distributions"
                    / "distribution_per_metric"
                    / f"{scale}_scale"
                )
                path2dir.mkdir(parents=True, exist_ok=True)
                full_path = path2dir / file_name
                plt.savefig(
                    full_path, bbox_inches="tight", transparent=True, format="svg"
                )
                logging.info(f"Plot saved to {full_path}")

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

        unique_labels = data[group_by].unique()
        max_words_in_label = max(len(str(label).split()) for label in unique_labels)
        use_shared_legend = max_words_in_label > 3 or len(unique_labels) > 5
        palette = sns.color_palette("tab20", n_colors=len(unique_labels))
        label_to_color = {
            label: palette[i % len(palette)] for i, label in enumerate(unique_labels)
        }

        for scale in ["linear", "symlog"]:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
            axes = axes.flatten()
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
                if scale == "symlog":
                    ax.grid(which="both", linestyle="--", color="gray", alpha=0.5)
                ax.set_yscale(scale)
                metric_for_tile = " ".join([t.capitalize() for t in metric.split("_")])
                ax.set_title(f"Distribution of {metric_for_tile}")
                # do not set value range -> results barely visible
                min_val = data[metric].min()
                max_val = data[metric].max()
                ax.set_xlim(left=max(0, min_val), right=min(1, max_val))
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
                bbox_to_anchor=(
                    0.95,
                    0.95,
                ),  # outside the plot on right (x pos, y pos)
                title=group_by.capitalize(),
                frameon=True,
                borderaxespad=0,
                fontsize=10,
                title_fontsize=12,
            )

            # Remove unused axes
            for j in range(len(metric_names), len(axes)):
                fig.delaxes(axes[j])
            title = (
                f"Metric Distributions\non {' '.join(word.capitalize() for word in data_category.split())} Dataset, grouped by {group_by.capitalize()}"
                if data_category
                else f"Metric Distributions\ngrouped by {group_by.capitalize()}"
            )
            fig.suptitle(title, fontsize=18)
            plt.tight_layout(rect=[0, 0, 0.95, 0.95])

            if save_path:
                scale_save_path = Path(save_path) / f"{scale}_scale"
                scale_save_path.mkdir(parents=True, exist_ok=True)
                for format in ["svg"]:
                    filenaname = f"{data_category.replace(' ', '_')}_metric_distributions_grouped_by_{group_by}_{scale}_scale.{format}"
                    full_path = scale_save_path / filenaname
                    plt.savefig(
                        full_path, bbox_inches="tight", transparent=True, format=format
                    )
                    logging.info(f"Plot saved to {full_path}")

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
    evaluator.plot_metric_radar_per_dataset(df_all=df, dataset_col="dataset_name", display_plot=False, metrics=metrics_names)
    for dataset_name in df["dataset_name"].unique():
        evaluator.plot_metric_scatter(df=df, data_category=dataset_name, display_plot=False)
        evaluator.plot_metric_distributions(df=df, data_category=dataset_name, display_plot=False)
        evaluator.plot_models_metrics(df=df, data_category=dataset_name, display_plot=False)
    logger.info("Evaluation complete.")
