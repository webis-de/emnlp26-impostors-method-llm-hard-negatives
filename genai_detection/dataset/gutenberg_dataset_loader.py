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

import random
from collections import defaultdict
from itertools import product
from pathlib import Path

import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
)

from genai_detection.dataset.base_dataset_loader import BaseDatasetLoader
from genai_detection.paraphrasing.two_step_paraphrasers import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", )

random.seed(42)
# Koppel et al. (2004): 500 words
# Bevendorff et al. (2019): 700 words (https://www.degruyterbrill.com/document/doi/10.1515/itit-2019-0046/html?casa_token=pbCaF7FgUXoAAAAA:8Vw71FUWE5spAbSsEuGGTdIjjm_o1_eb_inHwU3BR6eSrdVMOYy3--iqvDJwCV7EQ1HWtQBh610)
# Bevendorff et al. (2025): 3000 characters (https://aclanthology.org/2025.findings-acl.194.pdf)
MIN_NUM_WORDS = 700  # minimum number of words in a text to be considered valid


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
                logging.info(f"Column '{col}' has {num_nans} NaN values.")
                logging.info(f"Rows with NaN in '{col}':\n{df[df[col].isna()]}\n")

        grouped = df.groupby(groupby_cols)
        # logging.info(f"\nTotal groups: {len(grouped)}")
        # logging.info(f"Groups: {list(grouped.groups.keys())}\n\n")
        pairs = []

        for group_values, group in grouped:
            logging.info(f"Processing group: {group_values}, size: {len(group)}")
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
        logging.info(f"Train authors: {train_authors}, Test authors: {test_authors}\n\n")

        train_pairs = self.generate_pairs(df=train_df, groupby_cols=groupyby_cols)
        test_pairs = self.generate_pairs(df=test_df, groupby_cols=groupyby_cols)
        logging.info(
            f"Generated {len(train_pairs)} training pairs and {len(test_pairs)} test pairs."
        )
        # logging.info("training pairs: %s", [train_pairs[i]['authors'] for i in range(len(train_pairs))]if train_pairs
        # else "No training pairs generated.")
        # logging.info("test pairs: %s", [test_pairs[i]['authors'] for i in range(len(test_pairs))] if test_pairs else
        # "No test pairs generated.")

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
