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
from typing import Optional, List

from genai_detection.impostor_generators.ImpostorGenerator import NonNaiveLLMImpostorGenerator
from genai_detection.paraphrasing.one_step_paraphrasers import SAIAParaphraser
from genai_detection.paraphrasing.translation_paraphraser import TranslationParaphraser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )

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
        # TODO: Update translation generator logic
        impostors, _ = self.obtain_existing_paraphrases(
            collection=self.mongoDB.translation_collection,
            search_args={"text_id": text_id, "language":self.translation_paraphraser.language},
        )

        n_imp_to_generate = self.n_impostors - len(impostors)
        if n_imp_to_generate <= 0:
            logging.info(
                "Number of impostors in mongodb collection: {} for text with ID: {}. No need to generate more impostors, just returning {} impostors.".format(
                    len(impostors), text_id, self.n_impostors
                )
            )
            return impostors[: self.n_impostors]

        new_impostors = [self.translation_paraphraser.paraphrase(text=text) for i in range(n_imp_to_generate)]
        # Save new paraphrases in MongoDB
        for imp in new_impostors:
            self.translation_paraphraser.save_paraphrase_in_mongodb(
                original_text=text,
                original_text_id=text_id,
                paraphrased_text=imp,
                extracted_info={"language": self.translation_paraphraser.language},
                total_costs=0,
                prompt=self.translation_paraphraser.generator_prompt,
                intermediate_prompt=self.translation_paraphraser.extractor_prompt,
            )
        if impostors:
            new_impostors.extend(impostors)
        return new_impostors
