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

random.seed(42)
# Koppel et al. (2004): 500 words
# Bevendorff et al. (2019): 700 words (https://www.degruyterbrill.com/document/doi/10.1515/itit-2019-0046/html?casa_token=pbCaF7FgUXoAAAAA:8Vw71FUWE5spAbSsEuGGTdIjjm_o1_eb_inHwU3BR6eSrdVMOYy3--iqvDJwCV7EQ1HWtQBh610)
# Bevendorff et al. (2025): 3000 characters (https://aclanthology.org/2025.findings-acl.194.pdf)
MIN_NUM_WORDS = 700  # minimum number of words in a text to be considered valid

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
        logging.info("Initial number of entries: %d", len(df))
        df["text"] = df["text"].apply(lambda x: self.preprocess(x))
        df = df[
            df["text"].apply(lambda x: len(x.split()) >= MIN_NUM_WORDS)
        ]  # filter out text with less than MIN_NUM_WORDS words (not characters, bc there are 501 characters one-word entries)

        df["year"] = pd.to_datetime(
            df["date"], format="mixed", dayfirst=True, errors="coerce"
        ).dt.year
        logging.info("number of entries after filtering: %d", len(df))

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
