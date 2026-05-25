# Copyright 2026 Klara M. Gutekunst, Webis
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
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    ClassLabel,
    Features,
    Value,
)
from tqdm import tqdm

from genai_detection.dataset.base_dataset_loader import BaseDatasetLoader
from genai_detection.paraphrasing.two_step_paraphrasers import *

logger = logging.getLogger(__name__)

random.seed(42)
# Koppel et al. (2004): 500 words
# Bevendorff et al. (2019): 700 words (https://www.degruyterbrill.com/document/doi/10.1515/itit-2019-0046/html?casa_token=pbCaF7FgUXoAAAAA:8Vw71FUWE5spAbSsEuGGTdIjjm_o1_eb_inHwU3BR6eSrdVMOYy3--iqvDJwCV7EQ1HWtQBh610)
# Bevendorff et al. (2025): 3000 characters (https://aclanthology.org/2025.findings-acl.194.pdf)
MIN_NUM_WORDS = 700  # minimum number of words in a text to be considered valid


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

        logging.info("Label counts: %s", Counter(d["label"] for d in data))
        logging.info("Model counts: %s", Counter(d["model"] for d in data))

        train_data = [d for d in data if d["id"] in train_ids]
        test_data = [d for d in data if d["id"] in test_ids]

        logging.info(
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