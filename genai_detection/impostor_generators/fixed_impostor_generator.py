# Copyright 2025 Klara M. Gutekunst, Webis
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

import os
from typing import Dict

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.ImpostorGenerator import NonLLMImpostorGenerator


class FixedImpostorGenerator(NonLLMImpostorGenerator):
    def __init__(self, n_impostors: int, path2imp: str, split: str = "test"):
        """
        :param n_impostors: Number of impostors to generate.
        :param path2imp: Path to the directory where impostors are sampled from.
        :param split: The split to use when generating paraphrases.

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’. Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.
        """
        super().__init__(n_impostors=n_impostors, split=split, path2imp=path2imp)

    def generate_impostors(
        self, text: str
    ) -> Dict[str, str]:
        """
        Generates impostors from a pre-defined dataset.
        :param text: Input text to generate impostors for (not used in this implementation).
        :return: Dictionary of impostors with keys as ids and values as texts.
        """
        ds = self._get_dataset_split_from_path(self.path2imp)

        sampled = ds.shuffle().select(
            range(max(1, min(len(ds), self.n_impostors // 2)))
        )
        impostors = {}
        for i, row in sampled.to_pandas().iterrows():
            entry = row.to_dict()
            assert isinstance(
                entry, dict
            ), "Each entry in the dataset must be a dictionary."
            if "pair" not in entry:
                continue
            key = entry.get("id", f"impostor_{i}_fixed")
            impostors[f"{key}_left"] = entry["pair"][0]
            impostors[f"{key}_right"] = entry["pair"][1]

        if not impostors:
            raise ValueError("No impostors found with 'pair' field.")

        return impostors


class BlogImpostorGenerator(FixedImpostorGenerator):
    def __init__(self, n_impostors: int, split: str = "test"):
        super().__init__(n_impostors=n_impostors, split=split, path2imp=os.path.join(os.path.abspath(".."),
        CONFIG.PATH2BLOG))
