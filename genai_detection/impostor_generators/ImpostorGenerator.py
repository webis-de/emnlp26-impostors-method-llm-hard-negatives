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

# FIXME: make compatible with new mongodb idea
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from datasets import load_from_disk
from dotenv import load_dotenv

from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.paraphrasing.two_step_paraphrasers import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

load_dotenv()


class BaseImpostorGenerator(ABC):
    """Abstract base class for generating impostors."""

    def __init__(self, n_impostors: int):
        """
        :param n_impostors: number of impostors to generate
        """
        self.n_impostors = n_impostors

class NonLLMImpostorGenerator(BaseImpostorGenerator):
    def __init__(self, n_impostors: int, path2imp: str, split: str = "test"):
        """
        :param n_impostors: Number of impostors to generate.
        :param path2imp: Path to the directory where impostors are sampled from.
        :param split: The split to use when generating paraphrases.
        """
        super().__init__(n_impostors)
        self.split = split
        assert os.path.exists(path2imp), "Path {} does not exist.".format(path2imp)
        self.path2imp = path2imp

    @abstractmethod
    def generate_impostors(
        self, text: str
    ) -> dict:
        """Get a dictionary of impostor texts for the given input text.

        References:
        ===========
        Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
        Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.

        :param text_id: ID in a mongo database containing texts to retrieve impostors from
        :param text: input text to generate impostors for (i.e., the candidate text, NOT the disputed text)
        """
        pass

    def _get_dataset_split_from_path(self, path2imp: str):
        path2imp = Path(path2imp)
        if not path2imp.exists():
            raise FileNotFoundError(f"Data not found at {path2imp}")
        dataset = load_from_disk(
            os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
                path2imp,
            )
        )
        if self.split not in dataset:
            raise ValueError(
                f"Dataset {path2imp} does not contain '{self.split}' split."
            )

        ds = dataset[self.split]
        if len(ds) == 0:
            raise ValueError("Dataset split is empty.")
        return ds


class LLMImpostorGenerator(BaseImpostorGenerator):
    def __init__(
        self, n_impostors: int
    ):
        super().__init__(n_impostors)
        self.mongoDB = ParaphraseMongoDB()

    def _get_text_or_id(self, text: Optional[str], text_id: Optional[str]):
        """
        :param text: input text to generate impostors for (i.e., the candidate text)
        :param text_id: ID in a mongo database containing texts to retrieve impostors from.
        :return: A tuple containing text and text_id (can be none if not yet in collection).
        """
        assert text or text_id, "Either text or text_id must be provided."
        # Get text from mongodb if missing
        if not text:
            # Should only contain one element because _id is the primary key
            text = self.mongoDB.find_document(
                collection=self.mongoDB.original_collection, document_id=text_id
            )[0]["text"]
            assert (
                (text is not None) and type(text) == str and len(text) > 0
            ), f"Text ID {text_id} not found."
        # Save text in mongoDB if not yet present
        if text and text_id:
            existing = self.mongoDB.find_document(
                collection=self.mongoDB.original_collection, document_id=text_id
            )
            if existing is None:
                raise Exception(f"Document ID {text_id} not found.")
                self.mongoDB.insert_document(
                    collection=self.mongoDB.original_collection,
                    insert_data={"text": text},
                )
        return text, text_id
    # def generate_impostors():
        # for i in range(n_imp_to_generate):
        #     # randomly select a paraphraser and a prompt
        #     paraphraser = self.paraphrasers[i % len(self.paraphrasers)]
        #     p_id = random.randint(0, len(self.prompts) - 1)
        #     prompt = self.prompts[p_id]
        #     try:
        #         impostor_texts = paraphraser.paraphrase(text, prompt=prompt)
        #         if (
        #             isinstance(paraphraser, OneStepParaphraser)
        #             and paraphraser.model_id == "qwen3-32b"
        #         ):
        #             # qwen3-32b returns thinking steps and the final answer, separated by </think>
        #             impostor_texts = [
        #                 item.split("</think>")[-1] for item in impostor_texts
        #             ]
        #
        #         for imp in impostor_texts:
        #             if (
        #                 imp and (len(imp.split()) / len(text.split())) >= 0.6
        #             ):  # only non-empty + valid length filter
        #                 # read json requires no " or { }
        #                 imp = imp.replace("{", "(").replace("}", ")").replace('"', "'")
        #                 key = f"impostor_{i}_prompt{p_id}_{paraphraser.model_id}"
        #                 j = i
        #                 while key in list(impostors.keys()):
        #                     j += 1
        #                     key = f"impostor_{j}_prompt{p_id}_{paraphraser.model_id}"
        #                 impostors[key] = imp
        #     except Exception as e:
        #         print(f"Error generating impostor with {paraphraser}: {e}")

        # return impostors


class NonNaiveLLMImpostorGenerator(LLMImpostorGenerator):
    def __init__(self, n_impostors: int):
        """
        Naive LLM-based impostor generator that uses no naive paraphrasers (i.e. only two-step paraphrasers).
        :param n_impostors: number of impostors to generate
        """
        super().__init__(
            n_impostors=n_impostors,
        )


if __name__ == "__main__":
    llm_paraphraser = LLMImpostorGenerator(n_impostors=4)
    imps = llm_paraphraser.generate_impostors(text_id="68f50029edacdf3d5c0279eb", text=None)
    for i, imp in enumerate(imps):
        # $7,886.76 07.11.25, 10.04 Uhr
        # $7,886.75 07.11.25, 10.49 Uhr
        # $7,886.71 07.11.25, 12.49 Uhr
        # FIXME: number of impostors does not work properly, too many and too short paraphrases
        print("imp number ", i, "of length ", len(imp.split()))
