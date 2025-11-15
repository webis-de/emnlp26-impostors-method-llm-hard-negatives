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

from genai_detection.impostor_generators.ImpostorGenerator import NonNaiveLLMImpostorGenerator
from genai_detection.paraphrasing.one_step_paraphrasers import SAIAParaphraser
from genai_detection.paraphrasing.translation_paraphraser import TranslationParaphraser


class TranslationImpostorGenerator(NonNaiveLLMImpostorGenerator):
    def __init__(self, n_impostors: int, language: str = "French", model_id: Optional[str] = "openai-gpt-oss-120b"):
        super().__init__(n_impostors=n_impostors)
        llm =  SAIAParaphraser(model_id=model_id)
        self.translation_paraphraser = TranslationParaphraser(
                text_extractor=llm,
                text_generator=llm,
                language=language,
            )

    def generate_impostors(
            self, text: Optional[str], text_id: Optional[str]
    )-> List[str]:
        text, text_id = self._get_text_or_id(text, text_id)
        # TODO: Update translation generator logic
        paraphrases = [self.translation_paraphraser.paraphrase(text=text, text_id=text_id) for i in range(self.n_impostors)]
        return paraphrases
