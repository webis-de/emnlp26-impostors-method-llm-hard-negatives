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

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.ImpostorGenerator import LLMImpostorGenerator
from genai_detection.paraphrasing.one_step_paraphrasers import SAIAParaphraser
from genai_detection.paraphrasing.paraphraser import Paraphraser
from genai_detection.paraphrasing.translation_paraphraser import TranslationParaphraser
from genai_detection.paraphrasing.two_step_paraphrasers import (
    TwoStepParaphraser
)


class NaiveImpostorGenerator(LLMImpostorGenerator):
   def __init__(
        self, n_impostors: int, paraphrasers: Optional[List[Paraphraser]] = None
    ):
        super().__init__(n_impostors=n_impostors)
        if paraphrasers is None:
            # self.t5_chatgpt_paraphraser = T5ChatGPTParaphraser()
            # self.t5_google_paws_paraphraser = T5GooglePAWSParaphraser()
            # self.ollama_paraphraser = OllamaParaphraser(model_id=CONFIG.OLLAMA_MODEL)
            self.saiai_paraphraser_llama = SAIAParaphraser(
                model_id="meta-llama-3.1-8b-instruct"
            )
            self.saiai_paraphraser_mistral = SAIAParaphraser(
                model_id="mistral-large-instruct"
            )
            self.saiai_paraphraser_gpt = SAIAParaphraser(
                model_id="openai-gpt-oss-120b"
            )
            self.saiai_paraphraser_qwen = SAIAParaphraser(model_id="qwen3-32b")

            self.two_step_paraphraser = TwoStepParaphraser()
            self.translation_paraphraser = TranslationParaphraser(
                text_extractor=self.saiai_paraphraser_gpt,
                text_generator=self.saiai_paraphraser_gpt,
            )
            self.paraphrasers = [
                # self.t5_chatgpt_paraphraser,
                # self.t5_google_paws_paraphraser,
                # self.ollama_paraphraser,
                self.saiai_paraphraser_llama,
                self.saiai_paraphraser_mistral,
                self.saiai_paraphraser_gpt,
                self.saiai_paraphraser_qwen,  # explanations in the output, separated by </think>
                self.two_step_paraphraser,
                self.translation_paraphraser,
            ]
        else:
            assert all(
                isinstance(p, Paraphraser) for p in paraphrasers
            ), "All paraphrasers must be instances of Paraphraser or its subclasses."
            assert (
                len(paraphrasers) > 0
            ), "At least one paraphraser must be provided."
            self.paraphrasers = paraphrasers
        # FIXME: prompts??
        self.prompts = CONFIG.OPENAI_MODEL
