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
from typing import Dict

import numpy as np

from genai_detection.impostor_generators.ImpostorGenerator import NonLLMImpostorGenerator


class TextLenImpostorGenerator(NonLLMImpostorGenerator):
    def __init__(self, n_impostors: int, path2imp: str, split: str = "test"):
        super().__init__(n_impostors=n_impostors, split=split, path2imp=path2imp)

    def generate_impostors(
        self,
        text: str,
        valid_relative_text_len_dif: float = 0.3,
    ) -> Dict[str, str]:
        ds = self._get_dataset_split_from_path(self.path2imp)
        max_subset_size = min(
            self.n_impostors, len(ds)
        )  # generate around twice as many impostors as requested, to ensure diversity
        sampled = ds.shuffle(seed=42).select(range(max_subset_size))
        candidate_texts = []
        for entry in sampled:
            pair = entry.get("pair", [])
            candidate_texts.extend(pair)
        text_len = len(text)
        threshold = text_len * valid_relative_text_len_dif
        # filter candidates based on length
        filtered_candidates = [
            s for s in candidate_texts if abs(len(s) - text_len) < threshold
        ]
        if not filtered_candidates:
            return self.generate_impostors(
                text=text,
                path2imp=self.path2imp,
                real_time_generation=self.path2imp,
                valid_relative_text_len_dif=min(1, valid_relative_text_len_dif * 2),
            )  # try again with a larger threshold

        lengths = np.array([len(s) for s in filtered_candidates])
        diffs = np.abs(lengths - text_len)
        probs = 1 / (1 + diffs)
        probs /= probs.sum()

        num_to_sample = min(self.n_impostors, len(filtered_candidates))
        selected = np.random.choice(
            filtered_candidates, size=num_to_sample, replace=False, p=probs
        )

        return {f"impostor_{i}_text_len": imp for i, imp in enumerate(selected)}
