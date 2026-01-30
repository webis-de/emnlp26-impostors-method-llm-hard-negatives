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
import logging
import os
import sys
from abc import ABC
from itertools import chain
from pathlib import Path
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk, concatenate_datasets

from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

sys.path.append(os.path.abspath(".."))
from genai_detection.config import CONFIG
from genai_detection.detectors.impostor import ImpostorDetector

logger = logging.getLogger(__name__)

class BaseDatasetVisualization(ABC):
    def __init__(
        self, name: str, savefig_base: Union[str, os.PathLike] = CONFIG.SAVE_PATH
    ):
        """
        Initializes dataset visualization for specific dataset.

        :param name (str): Name of the dataset to load.
        """
        self.name = name
        # self.dataset = self.load_dataset()
        self.mongoDB =  ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
        test_dataset = pd.DataFrame(self.mongoDB.find_document_by_non_id_field(collection=self.mongoDB.test_pairs_collection, document_field_name="dataset_name", document_value=self.name))
        train_dataset = pd.DataFrame(self.mongoDB.find_document_by_non_id_field(collection=self.mongoDB.train_pairs_collection, document_field_name="dataset_name", document_value=self.name))
        self.dataset = pd.concat(
            [train_dataset, test_dataset],
            axis=0,
            ignore_index=True,
        )
        print("Train docs:", len(train_dataset))
        print("Test docs:", len(test_dataset))
        print(f"Loaded dataset {self.name} {self.dataset.columns}")
        self.savefig_base = Path(__file__).resolve().parents[2] / savefig_base
        assert (
            self.savefig_base.exists()
        ), f"Savefig base path {self.savefig_base} does not exist."
        self.savefig_base = self.savefig_base / "datasets" / self.name
        os.makedirs(self.savefig_base, exist_ok=True)

    def load_dataset(self) -> pd.DataFrame:
        """
        Loads and combines datasets from the specified path.

        :param dataset_name (str): Name of the dataset to load.

        Returns:
            pd.DataFrame: Combined dataset as a pandas DataFrame.
        """
        name2path = {
            CONFIG.PAN23: CONFIG.PATH2PAN23,
            CONFIG.PAN20: CONFIG.PATH2PAN20,
            CONFIG.BLOG: CONFIG.PATH2BLOG,
            CONFIG.GUTENBERG: CONFIG.PATH2GUTENBERG,
            CONFIG.PAN25: CONFIG.PATH2PAN25,
            CONFIG.KOPPEL: CONFIG.PATH2KOPPEL_WEBIS,
            CONFIG.STUDENT_ESSAYS: CONFIG.PATH2STUDENT_ESSAYS,
            CONFIG.ARTIFICIAL_STUDENT_ESSAYS: CONFIG.PATH2ARTIFICIAL_STUDENT_ESSAYS,
        }
        assert self.name in list(
            name2path.keys()
        ), f"Invalid dataset {self.name} provided."

        dataset = load_from_disk(
            Path(__file__).resolve().parents[2]/ name2path[self.name]
        )
        combined = concatenate_datasets([split for split in dataset.values()])
        return combined.to_pandas()

    def dataset_stats(self, dataset=None, save: bool = True) -> pd.DataFrame:
        """
        Prints the dataset statistics. Text length is calculated in characters and number of words.
        :param save: Whether to save the statistics to a CSV file.
        :return: A DataFrame containing the dataset statistics.
        """
        if dataset is None:
            dataset = self.dataset

        # collect unique ids first
        all_ids = set(dataset["left_id"]) | set(dataset["right_id"])
        all_authors = set(dataset["left_author"]) | set(dataset["right_author"])
        logger.info(f"Obtained {len(all_ids)} ids and {len(all_authors)} authors.")

        # fetch texts once per id
        id_to_text = {
            id: self.mongoDB.get_text_or_id_from_orginal_collection(
                text=None, text_id=id
            )[0]
            for id in all_ids
        }
        logger.info("Obtained mapping from ID to text")

        # rebuild list in original order
        all_texts = [id_to_text[id] for id in dataset["left_id"]]
        all_texts += [id_to_text[id] for id in dataset["right_id"]]
        logger.info("Obtained texts from IDs.")

        text_lengths = [len(text) for text in all_texts]
        num_words = [len(text.split()) for text in all_texts]
        text_lengths = np.array(text_lengths)
        num_words = np.array(num_words)
        stats = {
            "dataset_name": self.name,
            "num_pairs": len(dataset),
            "num_authors": len(all_authors),
            "num_same_pairs": dataset["same"].sum(),
            "num_different_pairs": len(dataset) - dataset["same"].sum(),
            "avg_text_len_chars": round(text_lengths.mean(), 2),
            "min_text_len_chars": text_lengths.min(),
            "max_text_len_chars": text_lengths.max(),
            "std_text_len_chars": round(text_lengths.std(), 2),
            "avg_text_len_words": round(num_words.mean(), 2),
            "min_text_len_words": num_words.min(),
            "max_text_len_words": num_words.max(),
            "std_text_len_words": round(num_words.std(), 2),
            "median_text_len_words": int(np.median(num_words)),
        }

        stats_df = pd.DataFrame([stats])

        if save:
            save_path = self.savefig_base / "statistics"
            save_path.mkdir(parents=True, exist_ok=True)
            stats_df.to_csv(
                save_path / f"{self.name}_stats.csv", index=False, float_format="%.2f"
            )
            logger.info(f"Statistics saved to {save_path / self.name}_stats.csv")

        return stats_df

    def plot_text_length_histogram(
        self,
        dataset=None,
        bins=10,
        preprocess_fn: Optional[Callable[[str], str]] = None,
        verbose: bool = False,
        unit: str = None,
    ):
        """
        Plots a histogram of text lengths (in characters) for a list of strings.

        Args:
            text_list (list): List of strings or ngrams.
            bins (int): Number of bins in the histogram.
            preprocessed (bool): If True, indicates that the text has been stemmed.
            unit (str): Unit of measurement for text length, either 'characters' or 'ngrams'.
        """
        if dataset is None:
            dataset = self.dataset
        text_list = chain.from_iterable(dataset["pair"])
        if preprocess_fn:
            text_list = [preprocess_fn(text) for text in text_list]
        lengths = [len(text) for text in text_list]
        if not unit:
            unit = "characters" if type(lengths[0]) is str else "ngrams"

        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=bins, color="skyblue", edgecolor="black")
        title = f"Histogram of Text Lengths\nDataset: {self.name}"
        plt.title(title)
        plt.xlabel(f"Text Length ({unit})")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        save_path = self.savefig_base / "plots"
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"{self.name}_text_length_histogram.png")
        plt.show()
        if not verbose:
            logging.info(f"Average length (characters) per text: {np.mean(lengths):.2f}")

    def plot_avg_text_length_per_author(self, dataset=None, unit="characters"):
        """
        Plots a bar chart of average text length per author.

        :param dataset: Huggingsface Dataset with 'authors' and 'pair' columns.
        """
        if dataset is None:
            dataset = self.dataset
        df = self._extract_authors_and_texts(dataset)

        if unit == "ngrams":
            impostor_det = ImpostorDetector(
                df, top_n=250, portion_delete=0.5, shared_vocab_only=True
            )
            df["text_length"] = df["text"].apply(
                lambda text: len(
                    impostor_det.tokenize_char_ngrams(
                        text, n=4, normalize_ws=True, space_free=True
                    )
                )
            )
        else:
            # Default to characters
            df["text_length"] = df["text"].str.len()
        avg_len = df.groupby("author")["text_length"].mean().sort_values()

        # Plot
        plt.figure(figsize=(10, max(10, len(df["author"].unique()) * 0.15)))
        avg_len.plot(kind="barh", color="mediumseagreen", edgecolor="black")
        plt.xlabel(f"Average Text Length ({unit})")
        plt.title(f"Average Text Length per Author\nDataset: {self.name}")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        save_path = self.savefig_base / "plots"
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"{self.name}_avg_text_length_per_author.png")
        plt.show()

    def _extract_authors_and_texts(self, dataset):
        """
        Extracts authors and texts from the dataset and returns them as a DataFrame.
        :param dataset: Huggingsface Dataset with 'authors' and 'pair' columns.
        :return: DataFrame with 'author' and 'text'
        """
        authors = dataset["authors"].tolist()
        authors = [
            s
            for item in authors
            for s in item.flatten().tolist()
            if isinstance(item, np.ndarray)
        ]
        pairs = dataset["pair"].tolist()
        texts = [
            s
            for item in pairs
            for s in item.flatten().tolist()
            if isinstance(item, np.ndarray)
        ]
        df = pd.DataFrame({"author": authors, "text": texts})
        return df

    def boxplots_text_len_ngrams(self, dataset=None):
        """
        Plot two boxplots side by side in the same plot:
        1) Average text length per author
        2) All text lengths
        Outliers on average lengths are annotated.

        :param dataset: Huggingsface Dataset with 'authors' and 'pair' columns.
        """
        if dataset is None:
            dataset = self.dataset
        df = self._extract_authors_and_texts(dataset)

        impostor_det = ImpostorDetector(
            df, top_n=250, portion_delete=0.5, shared_vocab_only=True
        )
        df["n_ngram"] = df["text"].apply(
            lambda text: len(
                impostor_det.tokenize_char_ngrams(
                    text, n=4, normalize_ws=True, space_free=True
                )
            )
        )
        df["text_length"] = df["text"].str.len()

        avg_text_len = df.groupby("author")["text_length"].mean()
        avg_n_gram = df.groupby("author")["n_ngram"].mean()

        fig, ax = plt.subplots(figsize=(10, 6))

        # boxplots vertically stacked (y=2 for text length, y=1 for ngram)
        data = [avg_text_len, avg_n_gram]
        positions = [2, 1]

        box = ax.boxplot(
            data, positions=positions, vert=False, widths=0.6, patch_artist=True
        )

        colors = ["lightblue", "lightgreen"]
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)

        # Y-axis labels
        ax.set_yticks(positions)
        ax.set_yticklabels(["Avg Text Length per Author", "Avg ngram Count per Author"])
        ax.set_xlabel("Value")
        ax.set_title(
            f"Boxplots of Average Text Length and ngram Count per Author\nDataset: {self.name}"
        )

        # Function to find and annotate outliers
        def annotate_outliers(variable, pos, color):
            Q1 = variable.quantile(0.25)
            Q3 = variable.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = variable[(variable < lower) | (variable > upper)]
            for author, val in outliers.items():
                ax.plot(val, pos, "o", color=color)
                ax.text(
                    val,
                    pos + 0.1,
                    author,
                    rotation=45,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Annotate outliers in each boxplot
        annotate_outliers(avg_text_len, 2, "blue")
        annotate_outliers(avg_n_gram, 1, "green")

        # Legend for outliers
        ax.plot([], [], "o", color="blue", label="Outliers in Avg Text Length")
        ax.plot([], [], "o", color="green", label="Outliers in Avg ngram")
        ax.legend(loc="upper right")

        ax.set_ylim(0.5, 2.5)
        plt.tight_layout()
        save_path = self.savefig_base / "plots"
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f"{self.name}_boxplots_text_len_ngrams.png")
        plt.show()


class Pan23Visualization(BaseDatasetVisualization):
    def __init__(self, name: str = CONFIG.PAN23):
        super().__init__(name=name)


class Pan20Visualization(Pan23Visualization):
    def __init__(self, name: str = CONFIG.PAN20):
        super().__init__(name=name)


class BlogVisualization(BaseDatasetVisualization):
    def __init__(self, name: str = CONFIG.BLOG):
        super().__init__(name=name)


class GutenbergVisualization(BaseDatasetVisualization):
    def __init__(self, name: str = CONFIG.GUTENBERG):
        super().__init__(name=name)


class Pan25Visualization(BaseDatasetVisualization):
    def __init__(self, name: str = CONFIG.PAN25):
        super().__init__(name=name)


class KoppelWebisVisualization(BaseDatasetVisualization):
    def __init__(self, name: str = CONFIG.KOPPEL):
        super().__init__(name=name)


class StudentEssaysVisualization(BaseDatasetVisualization):
    def __init__(self, name: str = CONFIG.STUDENT_ESSAYS):
        super().__init__(name=name)


class ArtificialStudentEssaysVisualization(BaseDatasetVisualization):
    def __init__(self, name: str = CONFIG.ARTIFICIAL_STUDENT_ESSAYS):
        super().__init__(name=name)

if __name__ == "__main__":
    student_essays_visualization = StudentEssaysVisualization()
    student_essays_visualization.dataset_stats()

    blog_visualization = BlogVisualization()
    blog_visualization.dataset_stats()
