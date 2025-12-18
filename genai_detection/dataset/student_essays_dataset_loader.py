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
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )


import hashlib
import random
from collections import defaultdict
from itertools import product
from pathlib import Path

import chardet
import numpy as np
import pandas as pd
import pyreadstat
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
)
from pymongo import errors

from genai_detection.dataset.base_dataset_loader import BaseDatasetLoader
from genai_detection.paraphrasing.two_step_paraphrasers import *


random.seed(42)
# Minimum length constraints motivated by prior work:
# - Koppel et al. (2004): 500 words
# - Bevendorff et al. (2019): 700 words
# - Bevendorff et al. (2025): 3000 characters
MIN_NUM_WORDS = 700

# Canonical column names used throughout the loader
ASSIGNMENT_COL_NAME = "assignment"
AUTHOR_COL_NAME = "author"

class StudentEssayDatasetLoader(BaseDatasetLoader):
    """
       Dataset loader for the Student Essay (Intro2006) corpus.

       The loader:
         1. Reads raw essay texts from disk.
         2. Filters texts by minimum length.
         3. Joins essays with demographic metadata.
         4. Stores normalized documents in MongoDB.
         5. Generates labeled text pairs for authorship verification.

       Pair generation follows Koppel et al. (2014), ensuring:
         - same-author and different-author pairs are drawn from different assignments,
         - train and test splits do not share assignments.

        Additionally, our different-author pairs share demographic subgroups.
       """
    def __init__(self, path: Optional[str], name: str = CONFIG.STUDENT_ESSAYS):
        super().__init__(name=name)
        self.path = Path(path)
        assert (
                path is None or self.path.exists()
        ), f"Path {self.path} is explicit input parameter but does not exist. Current path: {os.getcwd()}"
        self.features = Features(
            {
                "_id": Value("string"),
                "left_id": Value("string"),
                "right_id": Value("string"),
                f"left_{AUTHOR_COL_NAME}": Value("string"),
                f"right_{AUTHOR_COL_NAME}": Value("string"),
                f"left_{ASSIGNMENT_COL_NAME}": Value("string"),
                f"right_{ASSIGNMENT_COL_NAME}": Value("string"),
                "dataset_name": Value("string"),
                "same": Value("bool"),
            }
        )

    def _save_df2original_mongoDB_collection(self, df: pd.DataFrame):
        """
        Persist the processed essay DataFrame to the MongoDB collection
        for original (non-paired) texts.

        Each document is stored with:
          - author identifier,
          - text and metadata,
          - dataset name for later retrieval.

        Duplicate inserts are ignored.
        """
        records = df.to_dict(orient="records")
        docs_to_insert = [
            {
                "author": r.pop("author_id"),
                **r,
                "dataset": self.name,
            }
            for r in records
        ]

        if docs_to_insert:
            try:
                result = self.mongoDB.original_collection.insert_many(
                    docs_to_insert, ordered=False
                )
                logger.info(
                    f"Inserted {len(result.inserted_ids)} documents into '{self.name}' collection.'"
                )
            except errors.BulkWriteError:
                logger.warning(
                    "Duplicate key error encountered during insertMany. Some documents may already exist."
                )
        else:
            logger.info("No new documents to insert.")

    def load_texts(self, min_num_words: int = MIN_NUM_WORDS):
        """
        Load essay texts with metadata.

        If the dataset is already indexed in MongoDB, it is loaded directly.
        Otherwise, raw essays are read from disk, enriched with metadata,
        stored in MongoDB, and reloaded to ensure consistent document IDs.

        :param min_num_words: Minimum word count required for an essay.
        :return: DataFrame containing essays and metadata.
        """
        # return already indexed documents if existent
        cursor = self.mongoDB.find_document_by_non_id_field(collection=self.mongoDB.original_collection, document_field_name="dataset", document_value=self.name)
        records = list(cursor)
        if records and len(records) > 0:
            logger.debug(f"Found {records} documents in {CONFIG.MONGO_ORIGINAL_TEXT_COLLECTION}")
            return pd.DataFrame(records)

        # build student essays dataset
        df = self._load_student_essays(min_num_words=min_num_words)
        logger.info(f"obtained student essays dataset with {len(df)} entries.")

        # metadata dataframe
        author_metadata = self._load_student_metadata()
        # run once to save metadata
        path2metadata = (Path(__file__).resolve().parents[2]
            / CONFIG.DATA_BASE_PATH
            / "student_essays/Intro2006"
            / "file_metadata.xlsx")
        if not path2metadata.exists():
            author_metadata.to_excel(path2metadata,
                index=False,
            )
        logger.info(f"obtained author metadata with {len(author_metadata)} entries.")

        # join plain data with metadata
        df = df.join(
            author_metadata.set_index("author_id"),
            on="author_id",
            how="left",
            rsuffix="_meta",
        )
        df["text_hash"] = df["text"].apply(
            lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
        )
        logger.info("joined student essays with metadata.")

        # save obtained data in mongoDB collection
        self._save_df2original_mongoDB_collection(df=df)

        # obtain data from mongodb collection for correct text IDs
        return pd.DataFrame(list(self.mongoDB.find_document_by_non_id_field(collection=self.mongoDB.original_collection, document_field_name="dataset", document_value=self.name)))

    def load(
        self, train_split_portion: float = 0.7, min_num_words: int = MIN_NUM_WORDS
    ) -> DatasetDict:
        """
        Build the full Student Essay dataset with train/test splits.

        Essays are split by assignment to ensure that no assignment
        appears in both training and test sets.

        For each split, labeled text pairs are generated and stored
        in MongoDB before being returned as HuggingFace datasets.

        :param train_split_portion: Fraction of assignments used for training.
        :param min_num_words: Minimum word count per essay.
        :return: DatasetDict with 'train' and 'test' splits.
        """
        # build student essays dataset including metadata
        # load from mongoDB collection, if not existent; create one
        df = self.load_texts(min_num_words=min_num_words)

        # construct pairs
        assignment_col_name = "assignment"
        groupby_cols = [
            assignment_col_name,
            "sex",
            "ethnicity",
            "political_orientation",
            # "teacher",
            # "year",
        ]
        assert all(
            col in df.columns for col in groupby_cols
        ), f"Missing required metadata columns. Only got {df.columns}, but requires {groupby_cols}"
        # shuffle and split groups such that tasks are not overlapping between train and test sets
        all_tasks = df[assignment_col_name].unique().tolist()
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
            df[df[assignment_col_name].isin(train_tasks)]
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        test_df = (
            df[df[assignment_col_name].isin(test_tasks)]
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        logger.info(f"Train tasks: {train_tasks}, Test tasks: {test_tasks}\n\n")

        train_pairs = self.generate_pairs(df=train_df, groupby_cols=groupby_cols)
        self.save2mongoDB(train_pairs, is_train_split=True)
        test_pairs = self.generate_pairs(df=test_df, groupby_cols=groupby_cols)
        self.save2mongoDB(test_pairs, is_train_split=False)
        logger.info(
            f"Generated {len(train_pairs)} training pairs and {len(test_pairs)} test pairs."
        )

        return DatasetDict(
            {
                "train": Dataset.from_list(train_pairs, features=self.features),
                "test": Dataset.from_list(test_pairs, features=self.features),
            }
        )

    def _load_student_essays(self, min_num_words: int = MIN_NUM_WORDS) -> pd.DataFrame:
        """
        Load and preprocess raw student essays from disk.

        Essays are read from assignment-specific directories, decoded using
        automatic encoding detection, and filtered by length.

        Only the first four assignments are used, following Koppel et al. (2014).

        :param min_num_words: Minimum word count required for an essay.
        :return: DataFrame containing raw essays and assignment metadata.
        """
        assignment_col_name = "assignment"
        student_essays_df = pd.DataFrame(
            columns=["author_id", "text", assignment_col_name, "task_description"]
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
                                    assignment_col_name: task,
                                    "task_description": task_description[dir],
                                }
                            ]
                        ),
                    ]
                )
        return student_essays_df

    def _make_pair_dict(self, left_text:dict, right_text:dict, same:bool=True):
        """
        Construct a standardized pair dictionary for HuggingFace datasets.

        :param left_text: First essay record.
        :param right_text: Second essay record.
        :param same: Whether both essays originate from the same author.
        :return: Dictionary representing a labeled text pair.
        """
        FEATURE_MAP = {
            "_id": "_id",
            "_author": AUTHOR_COL_NAME,
            "_assignment": ASSIGNMENT_COL_NAME,
        }
        return {
            "dataset_name": self.name,
            **{f"left{key}": left_text[val] for key, val in FEATURE_MAP.items()},
            **{f"right{key}": right_text[val] for key, val in FEATURE_MAP.items()},
            "same": same,
        }

    def generate_pairs(
        self, df, n_pairs=2, groupby_cols: list[str] = [ASSIGNMENT_COL_NAME, "sex", "ethnicity"]
    ):
        """
        Generate labeled text pairs for authorship verification.

        Pair construction strategy:
          - Same-author pairs are sampled across different assignments.
          - Different-author pairs are sampled across assignments within
            the same demographic subgroup.
          - The number of different-author pairs is balanced to match
            the number of same-author pairs.

        :param df: DataFrame containing essays and metadata.
        :param n_pairs: Number of same-author pairs per author (upper bound).
        :param groupby_cols: Metadata columns defining demographic subgroups.
        :return: List of labeled text pair dictionaries.
        """
        for col in groupby_cols:
            assert col in df.columns, f"Column '{col}' not found in DataFrame."

        for col in groupby_cols:
            num_nans = df[col].isna().sum()
            if num_nans > 0:
                logger.info(f"Column '{col}' has {num_nans} NaN values.")
        pairs = []

        # each author appears <=1 time per task: same-author pairs have to be generated across tasks
        # ----------------
        # Same-author pairs (same subgroup, different tasks)
        # ----------------
        same_author_pairs = []
        # group texts by author disregarding the task
        author_groups = {}
        records = df.to_dict(orient="records")
        for item in records:
            author_groups.setdefault(item[AUTHOR_COL_NAME], []).append(item)
        for author, texts in author_groups.items():
            if len(texts) < 2:
                continue

            selected = random.sample(texts, min(n_pairs * 2, len(texts)))
            random.shuffle(selected)
            for i in range(0, len(selected) - 1, 2):
                a, b = selected[i], selected[i + 1]
                if (
                    a[ASSIGNMENT_COL_NAME] != b[ASSIGNMENT_COL_NAME]
                ):  # enforce different tasks (should always be the case)
                    same_author_pairs.append(self._make_pair_dict(left_text=a,right_text=b,same=True))
        pairs.extend(same_author_pairs)
        logger.info(f"Generated {len(same_author_pairs)} same-author pairs.")

        # Prepare cross-task (cf. Koppel et al. (2014)) different-author pairs
        # ----------------
        # Different-author pairs (same subgroup, different tasks)
        # ----------------
        # Separate task from other grouping cols
        subgroup_cols = [c for c in groupby_cols if c != ASSIGNMENT_COL_NAME]
        if not subgroup_cols:
            # create a random subgroup
            logging.warning("No subgroup columns, generating random pairs.")
            df["_random_subgroup"] = np.random.randint(0, 20, size=len(df))
            subgroup_cols = ["_random_subgroup"]

        subgrouped = df.groupby(subgroup_cols, dropna=False, observed=True)
        diff_author_pairs = []

        for subgroup_values, subgroup_df in subgrouped:
            # Collect authors by task inside this subgroup
            task_buckets = defaultdict(list)
            for row in subgroup_df.to_dict(orient="records"):
                task_buckets[row[ASSIGNMENT_COL_NAME]].append(row)

            tasks = list(task_buckets.keys())
            if len(tasks) < 2:
                continue  # need at least 2 tasks to cross-pair

            # All cross-task combinations
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    t1, t2 = tasks[i], tasks[j]
                    texts1, texts2 = task_buckets[t1], task_buckets[t2]

                    for a, b in product(texts1, texts2):
                        if a[AUTHOR_COL_NAME] == b[AUTHOR_COL_NAME]:
                            continue  # skip same-author, already handled
                        diff_author_pairs.append(self._make_pair_dict(left_text=a,right_text=b,same=False))

        # Randomly sample to balance with same-author pairs
        n_diff_pairs_target = len(same_author_pairs)
        random.shuffle(diff_author_pairs)
        diff_author_pairs = diff_author_pairs[:n_diff_pairs_target]

        pairs.extend(diff_author_pairs)
        logger.info(f"Generated {len(diff_author_pairs)} different-author pairs.\n")

        return pairs

    def _load_student_metadata(self):
        """
        Load and normalize demographic metadata for student authors.

        Metadata is read from the original SPSS file and restricted
        to attributes required for subgroup-based pairing.

        :return: DataFrame containing cleaned author metadata.
        """
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
