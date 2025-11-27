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
from abc import ABC
from datetime import datetime
from typing import Optional

import nltk

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

nltk.download("punkt_tab")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )


class Paraphraser(ABC):
    """
    Abstract base class for paraphrasing models.
    """
    def __init__(self, n_paraphrases: int, model_id: str = CONFIG.OPENAI_MODEL):
        self.n_paraphrases = n_paraphrases
        self.mongoDB = ParaphraseMongoDB()
        self.original_collection = self.mongoDB.original_collection
        self.paraphrase_collection = self.mongoDB.paraphrase_collection
        self.model_id = model_id

    def paraphrase(
        self, text: str, prompt: str, max_length: int = CONFIG.MAX_LENGTH
    ) -> str:
        """
        Generate a paraphrase of the input text.
        Generate a paraphrase of the input text.

        :param text: The input text to be paraphrased.
        :param prompt: The prompt to be used for paraphrasing.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :return: A paraphrased version of the input text.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def save_paraphrase_in_mongodb(self, original_text_id:str, original_text:str, paraphrased_text:str, extracted_info:Optional[dict], total_costs:Optional[float],temperature:float=1.0, prompt:str="bullet points dspy") -> None:
        """
        Save the paraphrase to the MongoDB database collection called "paraphrase".
        :param original_text_id: The id of the original text in the original_text collection.
        :param original_text: The original text as a string.
        :param paraphrased_text: The paraphrased text as a string.
        :param extracted_info: A dictionary containing the extracted information from the paraphrase.
        :param total_costs: The total cost of the paraphrase (including extracted information and paraphrase generation).
        :return:-
        """
        assert (
            type(paraphrased_text) == str
        ), f"paraphrased_text must be a str, but is of type {type(paraphrased_text)}"
        assert len(paraphrased_text.split()) > 0, "paraphrased_text is empty"

        # Build the new document
        paraphrase_doc = {
            # Do not use text_id as _id since a text will be paraphrased multiple times with different settings
            "text_id": original_text_id,  # ID of the original text document
            "length_original_text": len(original_text.split()),
            "length_paraphrased_text": len(paraphrased_text.split()),
            "intermediate_prompt": "bullet points dspy",
            "prompt": prompt,
            "llm": self.model_id,
            "temperature": temperature,
            "paraphrase": paraphrased_text,
            "extracted_info": extracted_info,
            "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "openai_costs": total_costs,
        }

        # Insert into document into paraphrase collection
        self.paraphrase_collection.insert_one(paraphrase_doc)
        logging.info(f"Inserted paraphrase for document ID: {original_text_id}")


#
# def get_paraphraser_dict() -> Dict[str, Paraphraser]:
#     """
#     Returns a dictionary of paraphrasers with their names as keys and instances as values.
#     """
#     paraphrasers: Dict[str, Paraphraser] = {
#         # often fail with NotImplementedError: Cannot copy out of meta tensor; no data! ...
#         "t5_ChatGPT": T5ChatGPTParaphraser(),
#         "t5_Google_PAWS": T5GooglePAWSParaphraser(),
#         "ollama": OllamaParaphraser(model_id=CONFIG.OLLAMA_MODEL),
#         # works better than t5 models and ollama
#         "qwen3-32b": SAIAParaphraser("qwen3-32b"),
#         "mistral-large-instruct": SAIAParaphraser("mistral-large-instruct"),
#         "openai-gpt-oss-120b": SAIAParaphraser("openai-gpt-oss-120b"),
#         "meta-llama-3.1-8b-instruct": SAIAParaphraser("meta-llama-3.1-8b-instruct"),
#     }
#     two_step_paraphraser = TwoStepParaphraser()
#     translation_paraphraser = TranslationParaphraser(
#         text_extractor=paraphrasers["openai-gpt-oss-120b"],
#         text_generator=paraphrasers["openai-gpt-oss-120b"],
#     )
#     paraphrasers.update(
#         {
#             "two-step": two_step_paraphraser,
#             "translation": translation_paraphraser,
#         }
#     )
#     return paraphrasers
