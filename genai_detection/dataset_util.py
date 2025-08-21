import argparse
import gc
from itertools import combinations, product
import os
import json
from pathlib import Path
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
import random
import re
import sys
from time import sleep
import unicodedata
import typing as t
import chardet

from datasets import (
    Dataset,
    DatasetDict,
    ClassLabel,
    Features,
    Value,
    load_from_disk,
    Sequence,
)
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from genai_detection.config import CONFIG
from genai_detection.paraphrasing.paraphraser import (
    NaiveParaphraser,
    NonNaiveParaphraser,
    get_paraphraser_dict,
)
from genai_detection.util import preprocess_text as _preprocess_text

random.seed(42)
# Koppel et al. (2004): 500 words
# Bevendorff et al. (2019): 700 words (https://www.degruyterbrill.com/document/doi/10.1515/itit-2019-0046/html?casa_token=pbCaF7FgUXoAAAAA:8Vw71FUWE5spAbSsEuGGTdIjjm_o1_eb_inHwU3BR6eSrdVMOYy3--iqvDJwCV7EQ1HWtQBh610)
# Bevendorff et al. (2025): 3000 characters (https://aclanthology.org/2025.findings-acl.194.pdf)
MIN_NUM_WORDS = 700  # minimum number of words in a text to be considered valid

# === BASE CLASS ===


class BaseDatasetLoader(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load(self) -> DatasetDict:
        pass

    def preprocess(
        self, text: t.Union[str, t.Iterable[str]]
    ) -> t.Union[str, t.List[str]]:
        return _preprocess_text(text=text)

    @staticmethod
    def _load_jsonl(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _load_ids(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())

    def _generate_pairs(self, data):
        """
        Generate pairs of texts from the dataset irrespective of confounders (i.e. naive combinations).
        Dataset is expected to be a list of dictionaries with 'text' and 'author' keys.
        Each pair consists of two texts, their authors, and a boolean indicating if they are from the same author.
        """
        pairs = []
        for a, b in combinations(data, 2):
            pairs.append(
                {
                    "pair": [a["text"], b["text"]],
                    "authors": [a["author"], b["author"]],
                    "same": a["author"] == b["author"],
                }
            )

        return pairs


# === Blog Corpus LOADER ===


class BlogCorpusDatasetLoader(BaseDatasetLoader):
    def __init__(self, path: str, name: str = CONFIG.BLOG):
        """Loader for the Blog Corpus dataset.
        Contains blog posts with dates from 01 January 1999 to 23 August 2006.
        When pairing texts, it is important to control confounders (i.e. pair similar external situations).
        Confounders can be:
        - topic
        - time period (e.g. 1999 vs. 2006)
        - age (?!)
        - gender (?!)

        Originally dataset is available at: https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus?resource=download (07.06.2025)
        """
        super().__init__(name=name)
        self.path = Path(path)

    def load(self) -> Dataset:
        df = pd.read_csv(self.path)
        print("Initial number of entries:", len(df))
        df["text"] = df["text"].apply(lambda x: self.preprocess(x))
        df = df[
            df["text"].apply(lambda x: len(x.split()) >= MIN_NUM_WORDS)
        ]  # filter out text with less than MIN_NUM_WORDS words (not characters, bc there are 501 characters one-word entries)

        df["year"] = pd.to_datetime(
            df["date"], format="mixed", dayfirst=True, errors="coerce"
        ).dt.year
        print("number of entries after filtering:", len(df))

        topics = df["topic"].unique().tolist()
        random.shuffle(topics)

        split_idx = int(0.8 * len(topics))
        train_topics = set(topics[:split_idx])
        test_topics = set(topics[split_idx:])

        def generate_pairs(topic_subset, n_pairs=2, groupby_cols: list = ["topic"]):
            """
            Generate pairs of texts from the dataset based on the specified topic subset.
            :param topic_subset: Set of topics to filter the dataset.
            :param n_pairs: Number of pairs to generate per group.
            :param groupby_cols: Columns to group by, should include 'topic'.
            :return: List of pairs with their authors and a boolean indicating if they are from the
            same author.
            """
            topic_df = df[df["topic"].isin(topic_subset)]
            assert (
                "topic" in groupby_cols
            ), "The 'topic' should be one of the columns to group by."

            grouped = topic_df.groupby(groupby_cols)
            pairs = []

            for group_values, group in grouped:
                data = group.to_dict(orient="records")

                # Group texts by author
                author_groups = {}
                for item in data:
                    author_groups.setdefault(item["id"], []).append(item)

                # Same-author pairs
                same_author_pairs = []
                for author, texts in author_groups.items():
                    if len(texts) < 2:
                        continue
                    selected = random.sample(texts, min(n_pairs * 2, len(texts)))
                    random.shuffle(selected)
                    for i in range(0, len(selected) - 1, 2):
                        a, b = selected[i], selected[i + 1]
                        same_author_pairs.append(
                            {
                                "pair": [a["text"], b["text"]],
                                "authors": [author, author],
                                "same": True,
                            }
                        )
                pairs.extend(same_author_pairs)

                # Different-author pairs, balanced to same author pairs count
                authors = list(author_groups.keys())
                if len(authors) > 1 and same_author_pairs:
                    n_diff_pairs_target = len(
                        same_author_pairs
                    )  # goal: match number of different-author pairs to same-author pairs
                    author_pairs = []
                    for i in range(len(authors)):
                        for j in range(i + 1, len(authors)):
                            author_pairs.append((authors[i], authors[j]))
                    random.shuffle(author_pairs)

                    count = 0
                    for a1, a2 in author_pairs:
                        if (
                            count >= n_diff_pairs_target
                        ):  # generate enough different-author pairs
                            break
                        if (
                            not author_groups[a1] or not author_groups[a2]
                        ):  # ensured above that authors are different
                            continue
                        t1 = random.choice(author_groups[a1])
                        t2 = random.choice(author_groups[a2])
                        pairs.append(
                            {
                                "pair": [t1["text"], t2["text"]],
                                "authors": [a1, a2],
                                "same": False,
                            }
                        )
                        count += 1

            return pairs

        features = Features(
            {
                "pair": [Value("string")],
                "authors": [Value("string")],
                "same": Value("bool"),
            }
        )

        # define the columns to group by
        groupby_cols = ["topic", "year", "gender", "age"]
        train_pairs = generate_pairs(train_topics, groupby_cols=groupby_cols)
        test_pairs = generate_pairs(test_topics, groupby_cols=groupby_cols)

        return DatasetDict(
            {
                "train": Dataset.from_list(train_pairs, features=features),
                "test": Dataset.from_list(test_pairs, features=features),
            }
        )


# === Koppel Webis LOADER ===


class KoppelWebisDatasetLoader(BaseDatasetLoader):
    def __init__(self, path: str, name: str = CONFIG.KOPPEL):
        super().__init__(name=name)
        self.path = Path(path)
        # assert (
        #     self.path.exists()
        # ), f"Path {self.path} does not exist. Current path: {os.getcwd()}"

    def load(self) -> DatasetDict:
        data = []
        for author in self.path.iterdir():
            if author.is_dir():
                for file in author.iterdir():
                    if file.is_file() and file.suffix == ".txt":
                        with open(file, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                            if len(content.split()) < MIN_NUM_WORDS:
                                continue
                            content = self.preprocess(content)
                            data.append({"author": author.name, "text": content})
        pairs = self._generate_pairs(data)

        # Define the structure for Hugging Face datasets
        features = Features(
            {
                "pair": [Value("string")],
                "authors": [Value("string")],
                "same": Value("bool"),
            }
        )
        return DatasetDict({"train": Dataset.from_list(pairs, features=features)})


# === PAN23 LOADER ===


class Pan23DatasetLoader(BaseDatasetLoader):
    def __init__(self, train_dir: str, test_dir: str, name: str = CONFIG.PAN23):
        super().__init__(name=name)
        self.train_dir = train_dir
        self.test_dir = test_dir

    def _load_dataset_from_directory(self, directory_path: str) -> Dataset:
        pairs = self._load_jsonl(os.path.join(directory_path, "pairs.jsonl"))
        path2truth = os.path.join(directory_path, "truth.jsonl")
        if not os.path.exists(path2truth):
            path2truth = os.path.join(
                directory_path.strip("/") + "-truth/", "truth.jsonl"
            )
        truth = self._load_jsonl(path2truth)
        truth_map = {item["id"]: item for item in truth}

        merged_data = []
        for item in tqdm(pairs, desc=f"Processing {directory_path}"):
            merged_data.append({**item, **truth_map.get(item["id"], {})})
        return Dataset.from_list(merged_data)

    def load(self) -> DatasetDict:
        train = self._load_dataset_from_directory(self.train_dir)
        test = self._load_dataset_from_directory(self.test_dir)
        return DatasetDict(train=train, test=test)


#  === PAN20 LOADER ===


class Pan20DatasetLoader(Pan23DatasetLoader):
    def __init__(self, train_dir: str, test_dir: str):
        """
        Loader for the PAN 2020 Authorship Verification dataset about fanfiction.
        The dataset is available at: https://zenodo.org/records/5106099 (08.06.2025)

        References:
        =========
        Sebastian Bischoff, Niklas Deckers, Marcel Schliebs, Ben Thies, Matthias Hagen, Efstathios Stamatatos, Benno Stein, and Martin Potthast. The Importance of Suppressing Domain Style in Authorship Analysis. CoRR, abs/2005.14714, May 2020.
        """
        super().__init__(name=CONFIG.PAN20, train_dir=train_dir, test_dir=test_dir)

    def _load_dataset_from_directory(self, directory_path: str) -> Dataset:
        pair_name = "pan20-authorship-verification-test.jsonl"
        truth_name = "pan20-authorship-verification-test-truth.jsonl"
        if not os.path.exists(os.path.join(directory_path, pair_name)):
            pair_name = "pan20-authorship-verification-training-small.jsonl"
            truth_name = "pan20-authorship-verification-training-small-truth.jsonl"
        pairs = self._load_jsonl(os.path.join(directory_path, pair_name))
        truth = self._load_jsonl(os.path.join(directory_path, truth_name))
        truth_map = {item["id"]: item for item in truth}

        merged_data = []
        for item in tqdm(pairs, desc=f"Processing {directory_path}"):
            merged_data.append({**item, **truth_map.get(item["id"], {})})

        merged_data = pd.DataFrame(merged_data)
        merged_data["pair"] = merged_data["pair"].apply(
            lambda pair: [self.preprocess(text) for text in pair]
        )
        # Keep only pairs with at least MIN_NUM_WORDS words in each text, bc there are one-word entries
        merged_data = merged_data[
            merged_data["pair"].apply(
                lambda pair: all(len(text.split()) >= MIN_NUM_WORDS for text in pair)
            )
        ]

        return Dataset.from_list(merged_data.to_dict(orient="records"))


# === PAN24 LOADER ===


class Pan24DatasetLoader(BaseDatasetLoader):
    def __init__(self, train_dir: str, test_dir: str, name: str = "pan24"):
        """
        Loader for the PAN 2024 Authorship Verification dataset.
        Dataset download at https://zenodo.org/records/10718757 (08.06.2025), restricted access.
        The dataset contains human and machine-generated texts on news articles.
        """
        super().__init__(name=name)
        self.train_dir = train_dir
        self.test_dir = test_dir

    def load(self) -> DatasetDict:
        # TODO: Implement the loading logic for PAN24 dataset
        pass


# === PAN25 LOADER ===


class Pan25DatasetLoader(BaseDatasetLoader):
    def __init__(
        self,
        human_dir: str,
        machine_dir: str,
        train_ids_path: str,
        test_ids_path: str,
        model_name_parent: int = 1,
    ):
        """
        Loader for the PAN 2025 Authorship Verification dataset.
        More information found at https://pan.webis.de/clef25/pan25-web/generated-content-analysis.html (08.06.2025)
        The dataset from subtask 1 contains human and machine-generated texts, with IDs for training and testing.
        """
        super().__init__(CONFIG.PAN25)
        self.human_dir = human_dir
        self.machine_dir = machine_dir
        self.train_ids_path = train_ids_path
        self.test_ids_path = test_ids_path
        self.model_name_parent = model_name_parent

    def _read_txts_from_dir(
        self, directory: str, label: str, is_human: bool, recursive: bool = True
    ):
        data = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if not filename.endswith(".txt"):
                    continue
                path = Path(os.path.join(root, filename))
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                file_id = f"{path.parents[0].name}/{path.stem}"
                model = (
                    path.parents[self.model_name_parent].name
                    if not is_human
                    else "human"
                )
                data.append(
                    {"id": file_id, "text": text, "label": label, "model": model}
                )
            if not recursive:
                break
        return data

    def load(self) -> DatasetDict:
        train_ids = self._load_ids(self.train_ids_path)
        test_ids = self._load_ids(self.test_ids_path)

        assert train_ids, "Train IDs list is empty."
        assert test_ids, "Test IDs list is empty."
        assert not train_ids.intersection(test_ids), "Train/Test ID sets overlap."

        data = self._read_txts_from_dir(
            self.human_dir, "human", is_human=True
        ) + self._read_txts_from_dir(self.machine_dir, "machine", is_human=False)

        print("Label counts:", Counter(d["label"] for d in data))
        print("Model counts:", Counter(d["model"] for d in data))

        train_data = [d for d in data if d["id"] in train_ids]
        test_data = [d for d in data if d["id"] in test_ids]

        print(
            f"Split into {len(train_data)} training and {len(test_data)} test records."
        )
        features = Features(
            {
                "id": Value("string"),
                "text": Value("string"),
                "label": ClassLabel(names=["human", "machine"]),
                "model": Value("string"),
            }
        )

        return DatasetDict(
            {
                "train": Dataset.from_list(train_data, features=features),
                "test": Dataset.from_list(test_data, features=features),
            }
        )


# === Gutenberg LOADER ===


class GutenbergDatasetLoader(BaseDatasetLoader):
    def __init__(self, path: str, name: str = CONFIG.GUTENBERG):
        super().__init__(name=name)
        self.path = Path(path)
        assert (
            self.path.exists()
        ), f"Path {self.path} does not exist. Current path: {os.getcwd()}"
        self.path2metadata = self.path / "file_metadata.xlsx"
        assert (
            self.path2metadata.exists()
        ), f"Metadata file {self.path2metadata} does not exist. Current path: {os.getcwd()}"

    def generate_pairs(self, df, n_pairs=2, groupby_cols: list = ["genre"]):
        """
        Generate pairs of texts from the dataset based on the specified groupby columns.
        :param df: DataFrame containing the dataset with at least 'text' and 'author' columns.
        :param n_pairs: Number of pairs to generate per group.
        :param groupby_cols: Columns to group by, should include 'genre'.
        :return: List of pairs with their authors and a boolean indicating if they are from the
        same author.
        """
        for col in groupby_cols:
            assert col in df.columns, f"Column '{col}' not found in DataFrame."

        for col in groupby_cols:
            num_nans = df[col].isna().sum()
            if num_nans > 0:
                print(f"Column '{col}' has {num_nans} NaN values.")
                print(f"Rows with NaN in '{col}':\n{df[df[col].isna()]}\n")

        grouped = df.groupby(groupby_cols)
        # print(f"\nTotal groups: {len(grouped)}")
        # print(f"Groups: {list(grouped.groups.keys())}\n\n")
        pairs = []

        for group_values, group in grouped:
            print(f"Processing group: {group_values}, size: {len(group)}")
            data = group.to_dict(orient="records")

            # Group texts by author
            author_groups = {}
            for item in data:
                author_groups.setdefault(item["author"], []).append(item)

            # Same-author pairs
            same_author_pairs = []
            for author, texts in author_groups.items():
                if len(texts) < 2:
                    continue
                selected = random.sample(texts, min(n_pairs * 2, len(texts)))
                random.shuffle(selected)
                for i in range(0, len(selected) - 1, 2):
                    a, b = selected[i], selected[i + 1]
                    same_author_pairs.append(
                        {
                            "pair": [a["text"], b["text"]],
                            "authors": [author, author],
                            "same": True,
                        }
                    )
            pairs.extend(same_author_pairs)

            # Different-author pairs, balanced to same author pairs count
            authors = list(author_groups.keys())
            if len(authors) > 1 and same_author_pairs:
                n_diff_pairs_target = len(
                    same_author_pairs
                )  # goal: match number of different-author pairs to same-author pairs
                author_pairs = []
                for i in range(len(authors)):
                    for j in range(i + 1, len(authors)):
                        author_pairs.append((authors[i], authors[j]))
                random.shuffle(author_pairs)

                count = 0
                # Calculate max number of pairs to sample per author pair (reduce class (i.e. different-author) imbalance introdoced prior when only one pair per author pair was sampled)
                max_pairs_per_pair = max(n_diff_pairs_target // len(author_pairs), 1)
                for a1, a2 in author_pairs:
                    texts_a1 = author_groups[a1]
                    texts_a2 = author_groups[a2]

                    if not texts_a1 or not texts_a2:
                        continue

                    # All possible combinations between texts from different authors
                    all_combinations = list(product(texts_a1, texts_a2))
                    random.shuffle(all_combinations)  # Shuffle to introduce randomness

                    num_to_sample = min(len(all_combinations), max_pairs_per_pair)
                    for t1, t2 in all_combinations[:num_to_sample]:
                        pairs.append(
                            {
                                "pair": [t1["text"], t2["text"]],
                                "authors": [a1, a2],
                                "same": False,
                            }
                        )
                        count += 1

                    if count >= n_diff_pairs_target:
                        break

        return pairs

    def load(self, train_split_portion: float = 0.8) -> Dataset:
        """
        Loader for the Gutenberg dataset.
        The dataset is expected to be a directory with text files, where each file is named in the format "title_firstName_sirname.txt".
        Each file contains the text of a book, and the author is derived from the filename.

        Authors cannot appear in both training and test sets.
        """
        assert (
            train_split_portion > 0 and train_split_portion < 1
        ), "train_split_portion must be between 0 and 1."
        data = []
        # obtain texts
        for file in self.path.glob("*.txt"):
            if "Complete_Works_of_William_Shakespeare" in file.name:
                # Skip the complete works of Shakespeare as it is way longer than other texts
                continue
            with open(file, "r", encoding="utf-8") as f:
                author = " ".join(
                    file.stem.split("_")[-2:]
                )  # filename format is "title_firstName_sirname.txt"
                content = f.read()
                if len(content.split()) < MIN_NUM_WORDS:
                    continue
                content = self.preprocess(content)
                data.append({"author": author, "text": content, "filename": file.stem})

        df = pd.DataFrame(data)
        # obtain metadata: contains time_period, author, topic (incomplete), summary, genre, century etc.
        metadata = pd.read_excel(self.path2metadata)
        # join on data's filename column and metadata's index (always 'others' index)
        df = df.join(
            metadata.set_index("filename"), on="filename", how="left", rsuffix="_meta"
        )

        groupyby_cols = ["genre", "century"]
        assert all(
            col in df.columns for col in groupyby_cols
        ), "Missing required metadata columns."

        # Group authors by (genre, century)
        author_meta = (
            df.groupby("author").first().reset_index()
        )  # keep only first occurrence of each author
        group_map = defaultdict(list)  # {(genre, century): [author1, author2, ...]}
        for _, row in author_meta.iterrows():
            key = (row["genre"], row["century"])
            group_map[key].append(row["author"])

        # Shuffle and split groups
        all_groups = list(group_map.items())
        random.shuffle(all_groups)

        train_authors = set()
        test_authors = set()
        train_count = 0
        total_authors = sum(len(authors) for _, authors in all_groups)
        author_limit = int(train_split_portion * total_authors)

        for key, authors in all_groups:
            if (
                train_count + len(authors) <= author_limit
            ):  # adds multiple authors from the same group at once
                train_authors.update(authors)
                train_count += len(authors)
            else:
                test_authors.update(authors)

        # Final sanity check
        assert train_authors.isdisjoint(
            test_authors
        ), "Author overlap between train and test."
        train_df = (
            df[df["author"].isin(train_authors)]
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        test_df = (
            df[df["author"].isin(test_authors)]
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        print(f"Train authors: {train_authors}, Test authors: {test_authors}\n\n")

        # TODO: add similarity on summary sbert?
        train_pairs = self.generate_pairs(df=train_df, groupby_cols=groupyby_cols)
        test_pairs = self.generate_pairs(df=test_df, groupby_cols=groupyby_cols)
        print(
            f"Generated {len(train_pairs)} training pairs and {len(test_pairs)} test pairs."
        )
        # print("training pairs:", [train_pairs[i]['authors'] for i in range(len(train_pairs))]if train_pairs else "No training pairs generated.")
        # print("test pairs:", [test_pairs[i]['authors'] for i in range(len(test_pairs))] if test_pairs else "No test pairs generated.")

        features = Features(
            {
                "pair": [Value("string")],
                "authors": [Value("string")],
                "same": Value("bool"),
            }
        )

        return DatasetDict(
            {
                "train": Dataset.from_list(train_pairs, features=features),
                "test": Dataset.from_list(test_pairs, features=features),
            }
        )


# === Student Essay LOADER ===


class StudentEssayDatasetLoader(BaseDatasetLoader):
    def __init__(self, path: str, name: str = CONFIG.STUDENT_ESSAYS):
        super().__init__(name=name)
        self.path = Path(path)
        assert (
            self.path.exists()
        ), f"Path {self.path} does not exist. Current path: {os.getcwd()}"

    def load(
        self, train_split_portion: float = 0.7, min_num_words: int = MIN_NUM_WORDS
    ) -> DatasetDict:
        """
        Loader for the Student Essay dataset.
        The dataset can be obtained from James W. Pennebaker.
        """
        # build student essays dataset
        df = self._load_student_essays(min_num_words=min_num_words)
        print("obtained student essays dataset with", len(df), "entries.")

        # metadata dataframe
        author_metadata = self._load_student_metadata()
        # run once to save metadata
        # author_metadata.to_excel(
        #     Path(__file__).resolve().parent.parent
        #     / CONFIG.DATA_BASE_PATH
        #     / "student_essays/Intro2006"
        #     / "file_metadata.xlsx",
        #     index=False,
        # )
        # keep NaNs, no answer is also an answer group
        print("obtained author metadata with", len(author_metadata), "entries.")

        df = df.join(
            author_metadata.set_index("author_id"),
            on="author_id",
            how="left",
            rsuffix="_meta",
        )
        print("joined student essays with metadata.")

        # construct pairs
        groupby_cols = [
            "task",
            "sex",
            "ethnicity",
            "political_orientation",
            # "teacher",
            # "year",
        ]
        assert all(
            col in df.columns for col in groupby_cols
        ), "Missing required metadata columns."

        # shuffle and split groups such that tasks are not overlapping between train and test sets
        all_tasks = df["task"].unique().tolist()
        random.seed(352)
        random.shuffle(all_tasks)

        task_limit = max(
            int(train_split_portion * len(all_tasks)), 2
        )  # at least 2 to ensure same-author pairs (bc each author appears <=1 time per task)
        train_tasks = set(all_tasks[:task_limit])
        test_tasks = set(all_tasks[task_limit:])

        # Final sanity check
        assert train_tasks.isdisjoint(
            test_tasks
        ), "Task overlap between train and test."
        train_df = (
            df[df["task"].isin(train_tasks)]
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        test_df = (
            df[df["task"].isin(test_tasks)]
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        print(f"Train tasks: {train_tasks}, Test tasks: {test_tasks}\n\n")

        train_pairs = self.generate_pairs(df=train_df, groupby_cols=groupby_cols)
        test_pairs = self.generate_pairs(df=test_df, groupby_cols=groupby_cols)
        print(
            f"Generated {len(train_pairs)} training pairs and {len(test_pairs)} test pairs."
        )

        features = Features(
            {
                "pair": [Value("string")],
                "authors": [Value("string")],
                "same": Value("bool"),
            }
        )

        return DatasetDict(
            {
                "train": Dataset.from_list(train_pairs, features=features),
                "test": Dataset.from_list(test_pairs, features=features),
            }
        )

    def _load_student_essays(self, min_num_words: int = MIN_NUM_WORDS) -> pd.DataFrame:
        """
        Load student essays.
        Koppel et al. (2014) use only the first 4 assignments.
        """
        student_essays_df = pd.DataFrame(
            columns=["author_id", "text", "task", "task_description"]
        )
        task_description = {
            "Ass1": "Stream of consciousness",
            "Ass2": "Talk about your childhood",
            "Ass3": "Describe your personality",
            "Ass4": "Thematic Apperception Test",
            # "Ass5": "Give four examples of four different theories",  # not used in Koppel et al. (2014)
        }
        # iterate over all TXT files in assignment directories and extract the essays
        for dir in [f"Ass{i}" for i in range(1, len(task_description) + 1)]:
            path2ass_dir = self.path / dir
            assert (
                path2ass_dir.exists()
            ), f'Path {path2ass_dir} to Assignment "{task_description[dir]}" of student essays dataset does not exist.'

            # if dir == "Ass5":  # has subtasks (directories)
            #     text_files = []
            #     for subtask in path2ass_dir.iterdir():
            #         text_files.extend(subtask.glob("*.txt"))
            # else:
            text_files = path2ass_dir.glob("*.txt")

            for txt_file in text_files:
                with open(txt_file, "rb") as f:
                    raw_data = f.read()
                    detected = chardet.detect(raw_data)
                    encoding = detected["encoding"]

                essay_text = self.preprocess(raw_data.decode(encoding))
                if len(essay_text.split()) < min_num_words:  # texts are < 1500 words
                    continue
                task = txt_file.parent.name
                if task == "Ass1" and "2006_" in txt_file.stem:
                    # files are named "2006_author_id.txt": Hence crop the year
                    author_id = txt_file.stem.split("_", 1)[1]  # remove "2006_"
                else:
                    author_id = txt_file.stem
                student_essays_df = pd.concat(
                    [
                        student_essays_df,
                        pd.DataFrame(
                            [
                                {
                                    "author_id": author_id,
                                    "text": essay_text,
                                    "task": task,
                                    "task_description": task_description[dir],
                                }
                            ]
                        ),
                    ]
                )
        return student_essays_df

    def generate_pairs(
        self, df, n_pairs=2, groupby_cols: list = ["task", "sex", "ethnicity"]
    ):
        """
        Generate pairs of texts from the dataset based on the specified groupby columns.
        Koppel et al. (2014) select pairs of texts from the different tasks, regardless of same or different author label.

        :param df: DataFrame containing the dataset with at least 'text' and 'author' columns.
        :param n_pairs: Number of pairs to generate per group.
        :param groupby_cols: Columns to group by, should include 'genre'.
        :return: List of pairs with their authors and a boolean indicating if they are from the
        same author.
        """
        for col in groupby_cols:
            assert col in df.columns, f"Column '{col}' not found in DataFrame."

        for col in groupby_cols:
            num_nans = df[col].isna().sum()
            if num_nans > 0:
                print(f"Column '{col}' has {num_nans} NaN values.")
                # print(f"Rows with NaN in '{col}':\n{df[df[col].isna()]}\n")

        grouped = df.groupby(
            groupby_cols, dropna=False, observed=True
        )  # keep NaNs, no answer is also an answer group
        pairs = []

        # each author appears <=1 time per task: same-author pairs have to be generated across tasks
        # Same-author pairs
        same_author_pairs = []
        # group texts by author disregarding the task
        author_groups = {}
        records = df.to_dict(orient="records")
        for item in records:
            author_groups.setdefault(item["author_id"], []).append(item)
        for author, texts in author_groups.items():
            if len(texts) < 2:
                # print(f"Skipping author {author} with only {len(texts)} text(s).")
                continue

            selected = random.sample(texts, min(n_pairs * 2, len(texts)))
            random.shuffle(selected)
            for i in range(0, len(selected) - 1, 2):
                a, b = selected[i], selected[i + 1]
                same_author_pairs.append(
                    {
                        "pair": [a["text"], b["text"]],
                        "authors": [author, author],
                        "same": True,
                    }
                )
        pairs.extend(same_author_pairs)
        print(f"Generated {len(same_author_pairs)} same-author pairs.")

        # Store author_id -> list of texts across all groups
        global_author_texts = defaultdict(list)

        # Mapping author_id to group_id to ensure authors come from different tasks
        author_group_map = {}

        # First loop: collect data across groups
        for group_values, group in grouped:
            group_id = str(group_values)
            data = group.to_dict(orient="records")

            for item in data:
                author_id = item["author_id"]
                global_author_texts[author_id].append(item)
                author_group_map[author_id] = group_id

        # Prepare cross-task (cf. Koppel et al. (2014)) different-author pairs
        authors = list(global_author_texts.keys())
        author_pairs = []
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                a1, a2 = authors[i], authors[j]
                if (
                    author_group_map[a1] != author_group_map[a2]
                ):  # ensure from different tasks
                    author_pairs.append((a1, a2))

        random.shuffle(author_pairs)

        # Determine how many different-author pairs to generate
        n_diff_pairs_target = len(same_author_pairs)
        max_pairs_per_pair = max(n_diff_pairs_target // len(author_pairs), 1)

        count = 0
        for a1, a2 in author_pairs:
            texts_a1 = global_author_texts[a1]
            texts_a2 = global_author_texts[a2]

            if not texts_a1 or not texts_a2:
                continue

            all_combinations = list(product(texts_a1, texts_a2))
            random.shuffle(all_combinations)

            num_to_sample = min(len(all_combinations), max_pairs_per_pair)
            for t1, t2 in all_combinations[:num_to_sample]:
                pairs.append(
                    {
                        "pair": [t1["text"], t2["text"]],
                        "authors": [a1, a2],
                        "same": False,
                    }
                )
                count += 1

            if count >= n_diff_pairs_target:
                break

        print(
            f"Generated {len(pairs) - len(same_author_pairs)} different-author pairs.\n"
        )
        return pairs

    def _load_student_metadata(self):
        author_metadata_columns = [
            "ID",
            "TEACHER",
            "BIRTHORD",
            "SEX",
            "ETHNIC",
            "POLITOR",
            "YEAR",
        ]
        rename_map = {
            "ID": "author_id",
            "TEACHER": "teacher",
            "BIRTHORD": "birthorder",
            "SEX": "sex",
            "ETHNIC": "ethnicity",
            "POLITOR": "political_orientation",
            "YEAR": "year",
        }
        raw_metadata, _ = pyreadstat.read_sav(
            self.path / "2006Big5xxx.sav",
            apply_value_formats=True,
            metadataonly=False,
        )
        available_columns = [
            col for col in raw_metadata.columns if col in author_metadata_columns
        ]
        author_metadata = raw_metadata[available_columns].copy()
        author_metadata.rename(columns=rename_map, inplace=True)

        return author_metadata


# === Cross-genre Dataset Loader ===
class CrossGenreDatasetLoader(BaseDatasetLoader):
    def __init__(self, path: str = "", name: str = CONFIG.PATH2CROSS_GENRE):
        """
        Loader for the Cross-genre dataset, i.e. from Blog, Gutenberg, and Student Essays.
        """
        super().__init__(name=name)

    def load(
        self, n_samples: int = 10, train_split_portion: float = 0.7
    ) -> DatasetDict:
        """
        Loader for the Cross-genre dataset.
        The dataset is expected to be a directory with text files, where each file is named in the format "author_genre.txt".
        Each file contains the text of a book, and the author and genre are derived from the filename.
        """
        # not-artifical generated pairs
        seed = 42
        np.random.seed(seed)

        category2directory = {
            "Blog": CONFIG.PATH2BLOG,
            "Gutenberg": CONFIG.PATH2GUTENBERG,
            "Student Essays": CONFIG.PATH2STUDENT_ESSAYS,
            "Pan 20": CONFIG.PATH2PAN20,
        }

        dataset = pd.DataFrame(
            columns=[
                "category",
                "disputed_text",
                "candidate_text",
                "same",
                "pair",
                "artificial_generation",
            ]
        )

        def load_dataset(path: Path) -> pd.DataFrame:
            return load_from_disk(path)["train"].to_pandas()

        for data_category in category2directory.keys():
            path2datasets = (
                Path(os.getcwd()).resolve() / category2directory[data_category]
            )
            complete_df = load_dataset(path2datasets)
            all_positive_samples = complete_df[complete_df["same"]]
            positive_sample = all_positive_samples.sample(
                n=min(n_samples, len(all_positive_samples)), random_state=seed
            )
            print(
                f"Obtained {len(positive_sample)} positive samples for {data_category}."
            )
            all_negative_samples = complete_df[~complete_df["same"]]
            negative_sample = all_negative_samples.sample(
                n=min(n_samples, len(all_negative_samples)), random_state=seed
            )
            # Delete to save memory
            del complete_df
            gc.collect()

            rows_to_add = []
            combined = pd.concat([positive_sample, negative_sample], ignore_index=True)
            if not combined.empty:
                pair_list = combined["pair"].values.tolist()
                disputed = [p[0] for p in pair_list]
                candidate = [p[1] for p in pair_list]

                new_rows = pd.DataFrame(
                    {
                        "category": [data_category] * len(pair_list),
                        "disputed_text": disputed,
                        "authors": combined["authors"].tolist(),
                        "candidate_text": candidate,
                        "same": combined["same"].tolist(),
                        "pair": pair_list,
                        "artificial_generation": [False] * len(pair_list),
                    }
                )

                dataset = pd.concat([dataset, new_rows], ignore_index=True)

        dataset = pd.concat([dataset, pd.DataFrame(rows_to_add)], ignore_index=True)
        all_paraphrasers_dict = get_paraphraser_dict()
        paraphrasers = {
            k: all_paraphrasers_dict[k]
            for k in [
                # "T5_ChatGPT",
                # "T5_Google_PAWS",
                # "Ollama",
                "qwen3-32b",
                "mistral-large-instruct",
                "openai-gpt-oss-120b",
                "meta-llama-3.1-8b-instruct",
                "BulletPoint",
                "Translation",
            ]
            if k in all_paraphrasers_dict
        }
        # only as many artificial samples as normal ones (multiplied by number of paraphrasers)
        unique_rows = dataset.drop_duplicates(subset=["disputed_text"])
        unique_rows = unique_rows.sample(
            n=max(1, min(len(unique_rows), n_samples)), random_state=seed
        )
        for i in tqdm(
            unique_rows.index, desc="Processing unique disputed texts for paraphrasing"
        ):
            text = unique_rows.loc[i, "disputed_text"]
            author = unique_rows.loc[i, "authors"][0]
            paraphrase_config = {
                "text": text,
                "n_responses": 1,
                "temperature": CONFIG.TEMPERATURE,
            }
            assert text is not None, f"Text at index {i} is None."
            rows_to_add = []
            for paraphraser_name, paraphraser in paraphrasers.items():
                try:
                    if isinstance(paraphraser, NaiveParaphraser):
                        paraphrase_config["prompt"] = (
                            "Paraphrase the text above. Do not use direct quotes or new lines. Respond ONLY with the paraphrase."
                        )
                    elif isinstance(paraphraser, NonNaiveParaphraser):
                        paraphrase_config["prompt"] = None
                    else:
                        raise ValueError(
                            f"Unknown paraphraser type: {type(paraphraser)}"
                        )
                    paraphrase = _preprocess_text(
                        paraphraser.paraphrase(**paraphrase_config)[0]
                    )
                    if not paraphrase:
                        continue
                except Exception as e:
                    print(f"[ERROR] Failed to paraphrase with {paraphraser_name}: {e}")
                    continue
                rows_to_add.append(
                    {
                        "category": unique_rows.loc[i, "category"],
                        "disputed_text": text,
                        "candidate_text": paraphrase,
                        "same": False,
                        "pair": [text, paraphrase],
                        "artificial_generation": True,
                        "authors": [author, paraphraser_name],
                    }
                )

        dataset = pd.concat([dataset, pd.DataFrame(rows_to_add)], ignore_index=True)
        features = Features(
            {
                "pair": Sequence(Value("string")),
                "authors": Sequence(Value("string")),
                "same": Value("bool"),
                "category": Value("string"),
                "disputed_text": Value("string"),
                "candidate_text": Value("string"),
                "artificial_generation": Value("bool"),
            }
        )

        # Convert DataFrames to list of dictionaries
        dataset = dataset.to_dict(orient="records")
        random.seed(seed)
        random.shuffle(dataset)
        train_data = dataset[: int(len(dataset) * train_split_portion)]
        test_data = dataset[int(len(dataset) * train_split_portion) :]
        print(f"Total dataset size: {len(train_data)} records.")

        # Create DatasetDict
        return DatasetDict(
            {
                "train": Dataset.from_list(train_data, features=features),
                "test": Dataset.from_list(test_data, features=features),
            }
        )


# === Artificial Student Essay LOADER ===


class ArtificialStudentEssayDatasetLoader(CrossGenreDatasetLoader):
    def __init__(
        self, path: str = "", name: str = CONFIG.PATH2ARTIFICIAL_STUDENT_ESSAYS
    ):
        """
        Loader for the Cross-genre dataset, i.e. from Blog, Gutenberg, and Student Essays.
        """
        super().__init__(name=name)

    def generate_existing_pairs(
        self, df: pd.DataFrame, n: int, random_state: int = 42
    ) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)

        pairs = []

        # --- SAME AUTHOR, DIFFERENT TASK ---
        same_author_groups = df.groupby("author_id")
        same_author_pairs = []
        for author, group in same_author_groups:
            tasks = group["task"].unique()
            if len(tasks) < 2:
                continue
            task_pairs = [
                (t1, t2) for i, t1 in enumerate(tasks) for t2 in tasks[i + 1 :]
            ]
            for t1, t2 in task_pairs:
                g1 = group[group["task"] == t1]
                g2 = group[group["task"] == t2]
                for _, row1 in g1.iterrows():
                    for _, row2 in g2.iterrows():
                        same_author_pairs.append((row1, row2))
        random.seed(42)  # for reproducibility
        same_author_samples = random.sample(
            same_author_pairs, min(n, len(same_author_pairs))
        )

        for row1, row2 in same_author_samples:
            print(row1)
            pairs.append(
                {
                    "candidate_assignment": row1["task"],
                    "candidate_assignment_description": row1["task_description"],
                    "disputed_assignment": row2["task"],
                    "disputed_assignment_description": row2["task_description"],
                    "disputed_text": row1["text"],
                    "candidate_text": row2["text"],
                    "same": True,
                    "pair": [row1["text"], row2["text"]],
                    "artificial_generation": False,
                    "disputed_author": row1["author_id"],
                    "candidate_author": row2["author_id"],
                    "authors": [row1["author_id"], row2["author_id"]],
                }
            )

        # --- DIFFERENT AUTHORS, DIFFERENT TASKS ---
        all_rows = df.to_dict("records")
        diff_author_pairs = []
        for i, row1 in enumerate(all_rows):
            for row2 in all_rows[i + 1 :]:
                if (
                    row1["author_id"] != row2["author_id"]
                    and row1["task"] != row2["task"]
                ):
                    diff_author_pairs.append((row1, row2))

        diff_author_samples = random.sample(
            same_author_pairs, min(n, len(diff_author_pairs))
        )

        for row1, row2 in diff_author_samples:
            pairs.append(
                {
                    "candidate_assignment": row1["task"],
                    "candidate_assignment_description": row1["task_description"],
                    "disputed_assignment": row2["task"],
                    "disputed_assignment_description": row2["task_description"],
                    "disputed_text": row1["text"],
                    "candidate_text": row2["text"],
                    "same": False,
                    "pair": [row1["text"], row2["text"]],
                    "artificial_generation": False,
                    "disputed_author": row1["author_id"],
                    "candidate_author": row2["author_id"],
                    "authors": [row1["author_id"], row2["author_id"]],
                }
            )

        return pd.DataFrame(
            pairs,
            columns=[
                "candidate_assignment",
                "candidate_assignment_description",
                "disputed_assignment",
                "disputed_assignment_description",
                "disputed_text",
                "candidate_text",
                "same",
                "pair",
                "artificial_generation",
                "disputed_author",
                "candidate_author",
                "authors",
            ],
        )

    def generate_llm_paraphrase_pairs(
        self,
        df: pd.DataFrame,
        paraphrasers: dict,
        task_description: dict,
        n: int = 100,
        random_state=42,
    ):
        rng = np.random.default_rng(random_state)
        pairs = []

        # ===================================
        # PROMPT TEMPLATE for LLM generated texts
        # ===================================
        def llm_student_prompt(assignment_desc: str) -> str:
            return (
                f"You are an 18-year-old first-year psychology major in 2006 at the "
                f"University of Texas in Austin (U.S.A.). "
                f"Your voice reflects the mindset of a college freshman in 2006: culturally aware "
                f"of the era, slightly anxious about school, curious about big ideas, and peppered "
                f"with references to early-2000s life, music, technology, and campus culture."
                f"Write a response to this task: '{assignment_desc}' with at least 700 words that is authentic to this persona."
            )

        # ==========================
        # STEP 1: LLM–AUTHOR PAIRS
        # ==========================
        sampled_rows = []
        tasks = list(task_description.keys())
        for t in tasks:  # ensure coverage of tasks
            candidates = df[df["task"] == t]
            if not candidates.empty:
                sampled_rows.append(
                    candidates.sample(1, random_state=random_state).iloc[0]
                )

        if len(sampled_rows) < n:
            extra = df.sample(n - len(sampled_rows), random_state=random_state).to_dict(
                "records"
            )
            sampled_rows.extend(extra)

        for row in sampled_rows[:n]:
            text = row["text"]
            task1 = row["task"]
            # choose a different task
            task2 = rng.choice([t for t in tasks if t != task1])

            paraphrase_config = {
                "text": text,
                "n_responses": 1,
                "temperature": 0.5,
            }
            for paraphraser_name, paraphraser in paraphrasers.items():
                paraphrase_config["prompt"] = llm_student_prompt(
                    task_description[task2]
                )
                try:
                    paraphrase = _preprocess_text(
                        paraphraser.paraphrase(**paraphrase_config)[0]
                    )
                    sleep(5)  # only 14 requests per minute
                except Exception as e:
                    print(f"[ERROR] {paraphraser_name} failed: {e}")
                    continue

                pairs.append(
                    {
                        "candidate_assignment": task2,
                        "candidate_assignment_description": task_description[task2],
                        "disputed_assignment": task1,
                        "disputed_assignment_description": task_description[task1],
                        "disputed_text": text,
                        "candidate_text": paraphrase,
                        "same": False,
                        "pair": [text, paraphrase],
                        "artificial_generation": True,
                        "disputed_author": row["author_id"],
                        "candidate_author": paraphraser_name,
                        "authors": [row["author_id"], paraphraser_name],
                    }
                )

        # ================================
        # STEP 2: LLM–LLM SAME PARAPHRASER
        # ================================
        rows_for_llm = df.sample(n, random_state=random_state).to_dict("records")
        for row in rows_for_llm:
            task1 = row["task"]
            # choose a different task
            task2 = rng.choice([t for t in tasks if t != task1])
            for paraphraser_name, paraphraser in paraphrasers.items():
                prompt1 = llm_student_prompt(task_description[task1])
                prompt2 = llm_student_prompt(task_description[task2])
                try:
                    text1 = _preprocess_text(
                        paraphraser.paraphrase(
                            text=row["text"],
                            n_responses=1,
                            temperature=1,
                            prompt=prompt1,
                        )[0]
                    )
                    text2 = _preprocess_text(
                        paraphraser.paraphrase(
                            text=row["text"],
                            n_responses=1,
                            temperature=1,
                            prompt=prompt2,
                        )[0]
                    )
                    sleep(10)  # only 14 requests per minute
                except Exception as e:
                    print(f"[ERROR] {paraphraser_name} failed: {e}")
                    continue

                pairs.append(
                    {
                        "candidate_assignment": task1,
                        "candidate_assignment_description": task_description[task1],
                        "disputed_assignment": task2,
                        "disputed_assignment_description": task_description[task2],
                        "disputed_text": text2,
                        "candidate_text": text1,
                        "same": True,
                        "pair": [text2, text1],
                        "artificial_generation": True,
                        "disputed_author": paraphraser_name,
                        "candidate_author": paraphraser_name,
                        "authors": [paraphraser_name, paraphraser_name],
                    }
                )

        # ================================
        # STEP 3: LLM–LLM DIFFERENT PARAPHRASERS
        # ================================
        rows_for_llm = df.sample(n, random_state=random_state + 1).to_dict("records")
        paraphraser_names = list(paraphrasers.keys())
        for row in rows_for_llm:
            task1 = row["task"]
            # choose a different task
            task2 = rng.choice([t for t in tasks if t != task1])
            p1, p2 = rng.choice(paraphraser_names, size=2, replace=False)
            prompt1 = llm_student_prompt(task_description[task1])
            prompt2 = llm_student_prompt(task_description[task2])
            try:
                text1 = _preprocess_text(
                    paraphrasers[p1].paraphrase(
                        text=row["text"], n_responses=1, temperature=1, prompt=prompt1
                    )[0]
                )
                text2 = _preprocess_text(
                    paraphrasers[p2].paraphrase(
                        text=row["text"], n_responses=1, temperature=1, prompt=prompt2
                    )[0]
                )
                sleep(10)  # only 14 requests per minute
            except Exception as e:
                print(f"[ERROR] LLM diff failed ({p1},{p2}): {e}")
                continue

            pairs.append(
                {
                    "candidate_assignment": task2,
                    "candidate_assignment_description": task_description[task2],
                    "disputed_assignment": task1,
                    "disputed_assignment_description": task_description[task1],
                    "disputed_text": text1,
                    "candidate_text": text2,
                    "same": False,
                    "pair": [text1, text2],
                    "artificial_generation": True,
                    "disputed_author": p1,
                    "candidate_author": p2,
                    "authors": [p1, p2],
                }
            )

        return pd.DataFrame(
            pairs,
            columns=[
                "candidate_assignment",
                "candidate_assignment_description",
                "disputed_assignment",
                "disputed_assignment_description",
                "disputed_text",
                "candidate_text",
                "same",
                "pair",
                "artificial_generation",
                "disputed_author",
                "candidate_author",
                "authors",
            ],
        )

    def balanced_train_test_split(
        self,
        dataset: pd.DataFrame,
        paraphrasers: dict,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        """
        Splits the dataset into train/test such that all combinations of:
            - artificial_generation (True/False)
            - same (True/False)
            - author type (LLM vs HUMAN)
        are represented in both sets.

        Args:
            dataset: pd.DataFrame with columns 'artificial_generation', 'same', 'disputed_author', 'candidate_author'
            paraphrasers: dict of LLM names (used to determine which authors are LLM)
            test_size: fraction of dataset to use as test set
            seed: random seed
        Returns:
            train_df, test_df
        """
        df = dataset.copy()

        # Define author type
        df["disputed_author_type"] = df["disputed_author"].apply(
            lambda x: "LLM" if x in paraphrasers.keys() else "HUMAN"
        )
        df["candidate_author_type"] = df["candidate_author"].apply(
            lambda x: "LLM" if x in paraphrasers.keys() else "HUMAN"
        )

        # Create a composite strata column for stratification
        df["strata"] = (
            df["artificial_generation"].astype(str)
            + "_"
            + df["same"].astype(str)
            + "_"
            + df["disputed_author_type"]
            + "_"
            + df["candidate_author_type"]
        )

        # Perform stratified split
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df["strata"]
        )

        # Drop helper columns
        train_df = train_df.drop(
            columns=["disputed_author_type", "candidate_author_type", "strata"]
        )
        test_df = test_df.drop(
            columns=["disputed_author_type", "candidate_author_type", "strata"]
        )

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def load(
        self, n_samples: int = 10, train_split_portion: float = 0.3
    ) -> DatasetDict:
        """
        Loader for the Artificial Student Essay dataset.
        The dataset is expected to be a directory with text files, where each file is named in the format "author_genre.txt".
        Each file contains the text of a book, and the author and genre are derived from the filename.
        """
        # not-artifical generated pairs
        seed = 42
        np.random.seed(seed)
        task_description = {
            "Ass1": "Stream of consciousness",
            "Ass2": "Talk about your childhood",
            "Ass3": "Describe your personality",
            "Ass4": "Thematic Apperception Test",
            # "Ass5": "Give four examples of four different theories",  # not used in Koppel et al. (2014)
        }

        student_essay_base_dir = (
            Path(__file__).resolve().parent.parent
            / CONFIG.DATA_BASE_PATH
            / "student_essays/Intro2006"
        )
        assert (
            student_essay_base_dir.exists()
        ), f"Path {student_essay_base_dir} to student essays dataset does not exist."

        loader = StudentEssayDatasetLoader(path=student_essay_base_dir)
        complete_df = loader._load_student_essays()

        all_paraphrasers_dict = get_paraphraser_dict()
        paraphrasers = {
            k: all_paraphrasers_dict[k]
            for k in [
                # "T5_ChatGPT",
                # "T5_Google_PAWS",
                # "Ollama",
                "qwen3-32b",
                "mistral-large-instruct",
                "openai-gpt-oss-120b",
                "meta-llama-3.1-8b-instruct",
            ]
            if k in all_paraphrasers_dict
        }

        print("Generating existing pairs from student essays dataset...")
        existing_pairs = self.generate_existing_pairs(
            complete_df, n_samples, random_state=seed
        )
        print(f"Generated {len(existing_pairs)} existing pairs from student essays.")
        print("Generating artifical pairs from student essays dataset...")

        artificial_pairs = self.generate_llm_paraphrase_pairs(
            complete_df,
            paraphrasers,
            task_description,
            n=n_samples,
            random_state=seed,
        )
        print(f"Generated {len(artificial_pairs)} artificial pairs.")

        dataset = pd.concat([existing_pairs, artificial_pairs], ignore_index=True)
        features = Features(
            {
                "candidate_assignment": Value("string"),
                "candidate_assignment_description": Value("string"),
                "disputed_assignment": Value("string"),
                "disputed_assignment_description": Value("string"),
                "disputed_author": Value("string"),
                "candidate_author": Value("string"),
                "pair": Sequence(Value("string")),
                "authors": Sequence(Value("string")),
                "same": Value("bool"),
                "disputed_text": Value("string"),
                "candidate_text": Value("string"),
                "artificial_generation": Value("bool"),
            }
        )
        train_df, test_df = self.balanced_train_test_split(
            dataset, paraphrasers, test_size=1 - train_split_portion, seed=seed
        )
        # Create DatasetDict
        return DatasetDict(
            {
                "train": Dataset.from_list(
                    train_df.to_dict("records"), features=features
                ),
                "test": Dataset.from_list(
                    test_df.to_dict("records"), features=features
                ),
            }
        )


# === SYSTEM SPECIFIC USAGE ===
def run_artificial_student_essay_dataset():
    base_dir = (
        Path(__file__).resolve().parent.parent
        / CONFIG.DATA_BASE_PATH
        / "student_essays/Intro2006"
    )
    assert (
        base_dir.exists()
    ), f"Path {base_dir} to student essays dataset does not exist."
    output_dir = (
        Path(__file__).resolve().parent.parent
        / CONFIG.PATH2ARTIFICIAL_STUDENT_ESSAYS
        / "sleep_time_limit"  # TODO: Omit
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ArtificialStudentEssayDatasetLoader(path=base_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)


def run_student_essay():
    base_dir = (
        Path(__file__).resolve().parent.parent
        / CONFIG.DATA_BASE_PATH
        / "student_essays/Intro2006"
    )
    assert (
        base_dir.exists()
    ), f"Path {base_dir} to student essays dataset does not exist."
    output_dir = os.path.join(base_dir, "student-essays-dataset-converted")

    loader = StudentEssayDatasetLoader(path=base_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)


def run_pan23(base_dir: str, save_path: str):
    base_dir = Path(__file__).resolve().parent / base_dir
    train_dir = os.path.join(base_dir, "pan23-authorship-verification-training-dataset")
    test_dir = os.path.join(base_dir, "pan23-authorship-verification-test-dataset")
    output_dir = os.path.join(save_path, "pan23-dataset-converted")

    loader = Pan23DatasetLoader(train_dir=train_dir, test_dir=test_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)
    # print("PAN23 example:", dataset["train"][0])


def run_pan20():
    base_dir = (
        Path(__file__).resolve().parent.parent
        / CONFIG.DATA_BASE_PATH
        / "pan20-authorship-verification/"
    )
    train_dir = os.path.join(base_dir, "pan20-authorship-verification-training-dataset")
    test_dir = os.path.join(base_dir, "pan20-authorship-verification-test-dataset")
    output_dir = os.path.join(base_dir, "pan20-dataset-converted")

    loader = Pan20DatasetLoader(train_dir=train_dir, test_dir=test_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)
    # print("PAN20 example:", dataset["train"][0])


def run_pan25():
    base_dir = (
        Path(__file__).resolve().parent
        / CONFIG.DATA_BASE_PATH
        / "dataset-extended-2025-part/"
    )
    human_dir = os.path.join(base_dir, "human")
    machine_dir = os.path.join(base_dir, "machines")
    train_ids_path = os.path.join(base_dir, "ids-train.txt")
    test_ids_path = os.path.join(base_dir, "ids-test.txt")
    output_dir = os.path.join(base_dir, "pan25-dataset-converted")

    loader = Pan25DatasetLoader(human_dir, machine_dir, train_ids_path, test_ids_path)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)
    # print("PAN25 example:", dataset["train"][0])


def run_koppel_webis():
    base_dir = (
        Path(__file__).resolve().parent.parent
        / CONFIG.DATA_BASE_PATH
        / "corpus-webis-authorship/koppel/"
    )
    output_dir = os.path.join(base_dir, "koppel-webis-dataset-converted")

    loader = KoppelWebisDatasetLoader(path=base_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)
    # print("Koppel Webis example:", dataset["train"][0])


def run_blog_corpus():
    # sys.path.append(os.path.abspath(".."))
    base_dir = (
        Path(__file__).resolve().parent.parent / CONFIG.DATA_BASE_PATH / "Blog_corpus/"
    )
    assert (
        base_dir.exists()
    ), f"Base directory {base_dir} does not exist. Current path: {os.getcwd()}"
    output_dir = base_dir / "blog-dataset-converted"

    loader = BlogCorpusDatasetLoader(path=base_dir / "blogtext.csv")
    dataset = loader.load()
    dataset.save_to_disk(output_dir)


def run_gutenberg_corpus():
    # sys.path.append(os.path.abspath(".."))
    base_dir = (
        Path(__file__).resolve().parent.parent / CONFIG.DATA_BASE_PATH / "gutenberg/"
    )
    assert (
        base_dir.exists()
    ), f"Base directory {base_dir} does not exist. Current path: {os.getcwd()}"
    output_dir = base_dir / "gutenberg-dataset-converted"

    loader = GutenbergDatasetLoader(path=base_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)


def run_cross_genre():
    loader = CrossGenreDatasetLoader()
    dataset = loader.load(n_samples=10)
    dataset.save_to_disk(
        Path(__file__).resolve().parent.parent / CONFIG.PATH2CROSS_GENRE
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dataset creation.")
    parser.add_argument(
        "--path",
        type=str,
        default="data/datasets/pan23-authorship-verification/",
        help="Path to the input dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/datasets/pan23-authorship-verification/",
        help="Path where Huggingface dataset should be saved (default: %(default)s)",
    )
    print("jksfkj %(parser)s")
    args = parser.parse_args()

    # run_pan23(base_dir=args.path, save_path=args.out)
    # # run_pan25()
    # run_pan20()
    # run_koppel_webis()
    # run_blog_corpus()
    # run_gutenberg_corpus()
    # run_student_essay()
    # run_cross_genre()
    run_artificial_student_essay_dataset()
