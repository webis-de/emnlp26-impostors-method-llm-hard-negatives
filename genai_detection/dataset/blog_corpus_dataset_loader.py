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
from itertools import combinations
from pathlib import Path

import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    NamedSplit,
)

from genai_detection.dataset.base_dataset_loader import BaseDatasetLoader, MIN_NUM_WORDS, AUTHOR_COL_NAME, \
    ASSIGNMENT_COL_NAME
from genai_detection.paraphrasing.two_step_paraphrasers import *

logger = logging.getLogger(__name__)

# === Blog Corpus LOADER ===

random.seed(42)
TOPIC_COL_NAME = "topic"



class BlogCorpusDatasetLoader(BaseDatasetLoader):
    def __init__(self, path: str, name: str = CONFIG.BLOG):
        """Loader for the Blog Corpus dataset.
        Contains blog posts with dates from 01 January 1999 to 23 August 2006.
        When pairing texts, it is important to control confounders (i.e. pair similar external situations).
        Confounders can be:
        - topic
        - time period (e.g. 1999 vs. 2006)
        - age
        - gender

        Originally dataset is available at: https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus?resource=download (07.06.2025)
        """
        super().__init__(name=name)
        self.topic_col_name = TOPIC_COL_NAME
        self.path = Path(path)

    def load_texts(self) -> pd.DataFrame:
        # return already indexed documents if existent
        self._return_existing_original_mongodb_collection()

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

        topics = df[TOPIC_COL_NAME].unique().tolist()
        random.shuffle(topics)

        df = df.rename(
            columns={
                "id": AUTHOR_COL_NAME,
                TOPIC_COL_NAME: ASSIGNMENT_COL_NAME,
            }
        )
        self.topic_col_name = ASSIGNMENT_COL_NAME

        # ensure stable ids for Mongo / HF
        df["_id"] = df.index.astype(str)
        # save obtained data in mongoDB collection
        self._save_df2original_mongoDB_collection(df=df, id_col_name=AUTHOR_COL_NAME)

        logger.info("Entries after filtering: %d", len(df))
        return df

    def load(self) -> DatasetDict[str | NamedSplit, Dataset]:

        df = self.load_texts()

        topics = df[self.topic_col_name].unique().tolist()
        random.shuffle(topics)

        split_idx = int(0.8 * len(topics))
        train_topics = set(topics[:split_idx])
        test_topics = set(topics[split_idx:])
        assert train_topics.isdisjoint(
            test_topics
        ), "Topic overlap between train and test."
        logger.info(f"Train topics: {train_topics}, Test topics: {test_topics}\n\n")

        groupby_cols = [self.topic_col_name, "year", "gender", "age"]

        train_pairs = self._generate_temporal_pairs(
            df[df[self.topic_col_name].isin(train_topics)],
            groupby_cols,
        )
        self.save2mongoDB(train_pairs, is_train_split=True)
        test_pairs = self._generate_temporal_pairs(
            df[df[self.topic_col_name].isin(test_topics)],
            groupby_cols,
        )
        self.save2mongoDB(test_pairs, is_train_split=False)

        return DatasetDict(
            {
                "train": Dataset.from_list(train_pairs, features=self.features),
                "test": Dataset.from_list(test_pairs, features=self.features),
            }
        )


    def _generate_temporal_pairs(
            self,
            df: pd.DataFrame,
            groupby_cols: list[str],
            n_pairs_per_author: int = 1,
            early_frac: float = 0.3,
            late_frac: float = 0.3,
            min_year_gap: int = 0,
    ):
        """
        Temporal pairing for both same- and different-author pairs.

        SAME author:
          left  -> early texts
          right -> late texts

        DIFFERENT authors:
          left  -> early text of author A
          right -> late text of author B

        Ensures: left.year <= right.year
        """
        pairs = []

        grouped = df.groupby(groupby_cols)

        for _, group in grouped:
            records = group.to_dict(orient="records")

            # group by author
            author2texts: dict[str, list[dict]] = {}
            for r in records:
                author2texts.setdefault(r["author"], []).append(r)

            # pre-sort texts chronologically per author
            for author in author2texts:
                author2texts[author] = sorted(
                    author2texts[author], key=lambda x: x["year"]
                )

            same_author_pairs = []

            # ===== SAME AUTHOR PAIRS =====
            for author, texts in author2texts.items():
                if len(texts) < 2:
                    continue

                k_early = max(1, int(len(texts) * early_frac))
                k_late = max(1, int(len(texts) * late_frac))

                early_pool = texts[:k_early]
                late_pool = texts[-k_late:]

                for _ in range(n_pairs_per_author):
                    left = random.choice(early_pool)
                    right = random.choice(late_pool)

                    if (
                            left["_id"] == right["_id"]
                            or right["year"] - left["year"] < min_year_gap
                    ):
                        continue

                    same_author_pairs.append(
                        self._make_pair_dict(left, right, same=True)
                    )

            pairs.extend(same_author_pairs)
            logger.info(f"Number same author pairs (for this group): {len(same_author_pairs)}")

            # ===== DIFFERENT AUTHOR PAIRS (TEMPORAL) =====
            authors = list(author2texts.keys())
            if len(authors) < 2 or not same_author_pairs:
                continue

            target = len(same_author_pairs)
            author_pairs = list(combinations(authors, 2))
            random.shuffle(author_pairs)

            count = 0
            for a1, a2 in author_pairs:
                if count >= target:
                    break

                texts_a = author2texts[a1]
                texts_b = author2texts[a2]

                # define early / late pools per author
                ea = texts_a[: max(1, int(len(texts_a) * early_frac))]
                la = texts_a[-max(1, int(len(texts_a) * late_frac)):]

                eb = texts_b[: max(1, int(len(texts_b) * early_frac))]
                lb = texts_b[-max(1, int(len(texts_b) * late_frac)):]

                # two possible temporal directions → pick valid one
                candidates = [
                    (random.choice(ea), random.choice(lb)),
                    (random.choice(eb), random.choice(la)),
                ]

                random.shuffle(candidates)

                for left, right in candidates:
                    if (
                            left["author"] != right["author"]
                            and right["year"] - left["year"] >= min_year_gap
                    ):
                        pairs.append(
                            self._make_pair_dict(left, right, same=False)
                        )
                        count += 1
                        break

        return pairs
