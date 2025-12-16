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

import chardet
import pandas as pd
import pyreadstat
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
)

from genai_detection.dataset.base_dataset_loader import BaseDatasetLoader
from genai_detection.paraphrasing.two_step_paraphrasers import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )

random.seed(42)
# Koppel et al. (2004): 500 words
# Bevendorff et al. (2019): 700 words (https://www.degruyterbrill.com/document/doi/10.1515/itit-2019-0046/html?casa_token=pbCaF7FgUXoAAAAA:8Vw71FUWE5spAbSsEuGGTdIjjm_o1_eb_inHwU3BR6eSrdVMOYy3--iqvDJwCV7EQ1HWtQBh610)
# Bevendorff et al. (2025): 3000 characters (https://aclanthology.org/2025.findings-acl.194.pdf)
MIN_NUM_WORDS = 700  # minimum number of words in a text to be considered valid


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
        logging.info(f"obtained student essays dataset with {len(df)} entries.")

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
        logging.info(f"obtained author metadata with {len(author_metadata)} entries.")

        df = df.join(
            author_metadata.set_index("author_id"),
            on="author_id",
            how="left",
            rsuffix="_meta",
        )
        logging.info("joined student essays with metadata.")

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
        logging.info(f"Train tasks: {train_tasks}, Test tasks: {test_tasks}\n\n")

        train_pairs = self.generate_pairs(df=train_df, groupby_cols=groupby_cols)
        test_pairs = self.generate_pairs(df=test_df, groupby_cols=groupby_cols)
        logging.info(
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
                logging.info(f"Column '{col}' has {num_nans} NaN values.")
                # logging.info(f"Rows with NaN in '{col}':\n{df[df[col].isna()]}\n")
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
            author_groups.setdefault(item["author_id"], []).append(item)
        for author, texts in author_groups.items():
            if len(texts) < 2:
                # logging.info(f"Skipping author {author} with only {len(texts)} text(s).")
                continue

            selected = random.sample(texts, min(n_pairs * 2, len(texts)))
            random.shuffle(selected)
            for i in range(0, len(selected) - 1, 2):
                a, b = selected[i], selected[i + 1]
                if (
                    a["task"] != b["task"]
                ):  # enforce different tasks (should always be the case)
                    same_author_pairs.append(
                        {
                            "pair": [a["text"], b["text"]],
                            "authors": [author, author],
                            "same": True,
                        }
                    )
        pairs.extend(same_author_pairs)
        logging.info(f"Generated {len(same_author_pairs)} same-author pairs.")

        # Prepare cross-task (cf. Koppel et al. (2014)) different-author pairs
        # ----------------
        # Different-author pairs (same subgroup, different tasks)
        # ----------------
        # Separate task from other grouping cols
        subgroup_cols = [c for c in groupby_cols if c != "task"]

        subgrouped = df.groupby(subgroup_cols, dropna=False, observed=True)
        diff_author_pairs = []

        for subgroup_values, subgroup_df in subgrouped:
            # Collect authors by task inside this subgroup
            task_buckets = defaultdict(list)
            for row in subgroup_df.to_dict(orient="records"):
                task_buckets[row["task"]].append(row)

            tasks = list(task_buckets.keys())
            if len(tasks) < 2:
                continue  # need at least 2 tasks to cross-pair

            # All cross-task combinations
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    t1, t2 = tasks[i], tasks[j]
                    texts1, texts2 = task_buckets[t1], task_buckets[t2]

                    for r1, r2 in product(texts1, texts2):
                        if r1["author_id"] == r2["author_id"]:
                            continue  # skip same-author, already handled
                        diff_author_pairs.append(
                            {
                                "pair": [r1["text"], r2["text"]],
                                "authors": [r1["author_id"], r2["author_id"]],
                                "same": False,
                            }
                        )

        # Randomly sample to balance with same-author pairs
        n_diff_pairs_target = len(same_author_pairs)
        random.shuffle(diff_author_pairs)
        diff_author_pairs = diff_author_pairs[:n_diff_pairs_target]

        pairs.extend(diff_author_pairs)
        logging.info(f"Generated {len(diff_author_pairs)} different-author pairs.\n")

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
