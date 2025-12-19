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
import typing as t
from abc import ABC, abstractmethod
from itertools import combinations

from datasets import (
    DatasetDict,
)

from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.paraphrasing.one_step_paraphrasers import *
from genai_detection.paraphrasing.two_step_paraphrasers import *
from genai_detection.util import preprocess_text as _preprocess_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", )

random.seed(42)
# Koppel et al. (2004): 500 words
# Bevendorff et al. (2019): 700 words (https://www.degruyterbrill.com/document/doi/10.1515/itit-2019-0046/html?casa_token=pbCaF7FgUXoAAAAA:8Vw71FUWE5spAbSsEuGGTdIjjm_o1_eb_inHwU3BR6eSrdVMOYy3--iqvDJwCV7EQ1HWtQBh610)
# Bevendorff et al. (2025): 3000 characters (https://aclanthology.org/2025.findings-acl.194.pdf)
MIN_NUM_WORDS = 700  # minimum number of words in a text to be considered valid


# === BASE CLASS ===


class BaseDatasetLoader(ABC):
    def __init__(self, name: str):
        self.name = name
        # Connect to MongoDB (default host/port for container)
        self.mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))

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

    def save2mongoDB(self, data: List[dict], is_train_split: bool=True):
        collection_name = f"{'train' if is_train_split else 'test'}_pairs"
        collection = self.mongoDB.db[collection_name]
        self.mongoDB.insert_documents(collection=collection, insert_data=data)





