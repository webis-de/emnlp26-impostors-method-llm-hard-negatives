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
import logging
from typing import Dict

import torch
from sentence_transformers import util, SentenceTransformer

from genai_detection.impostor_generators.ImpostorGenerator import NonLLMImpostorGenerator

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )

class ContentImpostorGenerator(NonLLMImpostorGenerator):
    def __init__(
        self,
        n_impostors: int,
        path2imp: str,
        split: str = "test",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        :param n_impostors: Number of impostors to generate.
        :param path2imp: Path to the directory where impostors are sampled from.
        :param model_name: Embedding model from sentence-transformers (default: "all-MiniLM-L6-v2").
        :param split: Dataset split to use (default: 'test').
        """
        super().__init__(n_impostors=n_impostors, split=split, path2imp=path2imp)
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        logging.info(f"Using device: {device}")
        self.model = SentenceTransformer(model_name)  # , device=device)

    def generate_impostors(
        self, text: str
    ) -> Dict[str, str]:
        """
        Generates impostors from a pre-defined dataset.
        :param text: Input text to generate impostors for (not used in this implementation)
        :return: dictionary of impostors with keys as ids and values as texts
        """
        ds = self._get_dataset_split_from_path(self.path2imp)
        sampled = ds.shuffle().select(range(min(len(ds), self.n_impostors)))
        # select n_impostors entries which are most similar to the input text in terms of content
        candidate_texts = []
        for _, row in sampled.to_pandas().iterrows():
            entry = row.to_dict()
            assert isinstance(
                entry, dict
            ), "Each entry in the dataset must be a dictionary (generate_impostors)."
            pair = entry.get("pair", [])
            candidate_texts.extend(pair)

        if not candidate_texts:
            raise ValueError("No candidate texts found.")

        # compute embeddings
        text_embedding = self.model.encode(text, convert_to_tensor=True)
        candidate_embeddings = self.model.encode(
            candidate_texts, convert_to_tensor=True
        )

        # compute cosine similarity
        similarities = util.cos_sim(text_embedding, candidate_embeddings)[0]

        #  top-n most similar
        top_indices = similarities.topk(self.n_impostors).indices.tolist()
        selected_texts = [candidate_texts[i] for i in top_indices]

        return {f"impostor_{i}_content": imp for i, imp in enumerate(selected_texts)}
