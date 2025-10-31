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
from typing import List

import nltk

from genai_detection.config import CONFIG

nltk.download("punkt_tab")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)


class Paraphraser(ABC):
    """
    Abstract base class for paraphrasing models.
    """

    def paraphrase(
            self, text: str, prompt: str, max_length: int = CONFIG.MAX_LENGTH
    ) -> List[str]:
        """
        Generate a paraphrase of the input text.

        :param text: The input text to be paraphrased.
        :param prompt: The prompt to be used for paraphrasing.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :return: A paraphrased version of the input text.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def paraphrase_batch(self, texts: list[str], prompt: str = None) -> list[list[str]]:
        """
        Generate paraphrases for a batch of input texts.

        :param texts: A list of input texts to be paraphrased.
        :param prompt: The prompt to be used for paraphrasing. If None, a default prompt will be used.
        :return: A list of paraphrased versions of the input texts.
        """

        return [self.paraphrase(text=text, prompt=prompt) for text in texts]

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
