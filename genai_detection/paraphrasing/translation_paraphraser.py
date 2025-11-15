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
from typing import Optional, List

import deepl

from genai_detection.config import CONFIG
from genai_detection.paraphrasing.one_step_paraphrasers import OneStepParaphraser
from genai_detection.paraphrasing.paraphraser import Paraphraser


class TranslationParaphraser(Paraphraser):
    """
    A paraphrasing model that first extracts the title of the text, tone and genre from the input text using one LLM and then generates a paraphrase based on this information.

    Inspired by the work of:
    C. Zhou, C. Qiu, L. Liang and D. E. Acuna, "Paraphrase Identification With Deep Learning: A Review of Datasets and Methods," in IEEE Access, vol. 13, pp. 65797-65822, 2025, doi: 10.1109/ACCESS.2025.3556899.

    For implementation details, see:
    https://github.com/deeplcom/deepl-python (accessed 25.07.2025)
    """

    def __init__(
        self,
        text_extractor: Paraphraser,
        text_generator: Paraphraser,
        language: str = "French",
    ):
        """
        Initializes the TranslationParaphraser model.
        :param text_extractor: A model or function to translate to foreign languages.
        :param text_generator: A model or function to translate from foreign languages.
        """
        self.model_id = "translation"
        assert isinstance(text_extractor, OneStepParaphraser) and isinstance(
            text_generator, OneStepParaphraser
        ), "Both text_extractor and text_generator must be instances of NaiveParaphraser or its subclasses."
        self.text_generator = text_generator
        self.text_extractor = text_extractor
        self.language = language
        self.extractor_prompt = f"Translate the text above into {self.language}. Do not use direct quotes or newlines. Output only the translated text, without any additional commentary or formatting."
        self.generator_prompt = f"Translate the text above from {self.language} into English. Do not use direct quotes or newlines. Output only the translated text, without any additional commentary or formatting."
        self.deepl_client = deepl.DeepLClient(CONFIG.DEEPL_API_KEY)
        self.text_extractor = text_extractor
        self.text_generator = text_generator

    def paraphrase(
        self, text: str, prompt: Optional[str], max_length: int = CONFIG.MAX_LENGTH
    ) -> List[str]:
        print(
            f"[DEBUG] Using TranslationParaphraser with prompt: {self.extractor_prompt}"
        )
        paraphrased_texts = []
        try:
            translation = self.deepl_client.translate_text(text, target_lang="FR")
            language = (
                "EN-US"
                if translation.detected_source_lang == "EN"
                else translation.detected_source_lang
            )
            res = self.deepl_client.translate_text(
                translation.text, target_lang=language
            )
            paraphrased_texts.append(res.text)
        except Exception as e:
            print(f"[ERROR] Failed to translate text using DeepL: {e}")

        translation = self.text_extractor.paraphrase(
            text=text, prompt=self.extractor_prompt, max_length=max_length
        )
        while not translation:
            print(
                f"[WARNING] No translation returned. Retrying with the same text and prompt: {self.extractor_prompt}"
            )
            translation = self.text_extractor.paraphrase(
                text=text, prompt=self.extractor_prompt, max_length=max_length
            )
        paraphrased_texts.extend(
            self.text_generator.paraphrase(
                text=translation[0], prompt=self.generator_prompt
            )
        )

        return paraphrased_texts
