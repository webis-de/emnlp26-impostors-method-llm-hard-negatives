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

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.ImpostorGenerator import LLMImpostorGenerator
from genai_detection.paraphrasing.one_step_paraphrasers import (
    SAIAParaphraser,
    OneStepParaphraser,
)
from genai_detection.paraphrasing.paraphraser import Paraphraser


class NaiveImpostorGenerator(LLMImpostorGenerator):
    def __init__(
        self, n_impostors: int, top_n_freq_words:int, paraphrasers: Optional[List[Paraphraser]] = None
    ):
        super().__init__(n_impostors=n_impostors, top_n_freq_words=top_n_freq_words)
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

            self.paraphrasers = [
                # self.t5_chatgpt_paraphraser,
                # self.t5_google_paws_paraphraser,
                # self.ollama_paraphraser,
                self.saiai_paraphraser_llama,
                self.saiai_paraphraser_mistral,
                self.saiai_paraphraser_gpt,
                self.saiai_paraphraser_qwen,  # explanations in the output, separated by </think>
            ]
        else:
            assert all(
                isinstance(p, Paraphraser) for p in paraphrasers
            ), "All paraphrasers must be instances of Paraphraser or its subclasses."
            assert (
                len(paraphrasers) > 0
            ), "At least one paraphraser must be provided."
            self.paraphrasers = paraphrasers

    def generate_impostors(
        self, text: Optional[str], text_id: Optional[str]
    ) -> List[str]:
        n_imp_to_generate = self.n_impostors

        # TODO: create one method with search generator: No, bc i need if here anyway
        impostors, _ = self.obtain_existing_paraphrases(
            collection=self.mongoDB.naive_paraphrase_collection,
            search_args={"text_id": text_id},
        )

        n_imp_to_generate -= len(impostors)
        if n_imp_to_generate <= 0:
            logging.info(
                "Number of impostors in mongodb collection: {}/{} for text with ID: {}. No need to generate more impostors, just returning {} impostors.".format(
                    len(impostors), self.n_impostors, text_id, self.n_impostors
                )
            )
            return impostors[: self.n_impostors]

        logging.info(
            f"{len(impostors)} precomputed impostors found in mongoDB. Generating {n_imp_to_generate} impostors for text {text[:100]}..."
        )
        for i in range(n_imp_to_generate):
            # randomly select a paraphraser and a prompt
            paraphraser = self.paraphrasers[i % len(self.paraphrasers)]
            try:
                impostor_text = paraphraser.paraphrase(text, prompt=CONFIG.PROMPT)
                if (
                    isinstance(paraphraser, OneStepParaphraser)
                    and paraphraser.model_id == "qwen3-32b"
                ):
                    # qwen3-32b returns thinking steps and the final answer, separated by </think>
                    impostor_text = impostor_text.split("</think>")[-1]

                # automatically inserts model_id
                paraphraser.save_paraphrase_in_mongodb(
                    original_text=text,
                    original_text_id=text_id,
                    paraphrased_text=impostor_text,
                    extracted_info={},
                    total_costs=0,
                    prompt=CONFIG.PROMPT,
                    intermediate_prompt="",
                    collection=self.mongoDB.naive_paraphrase_collection,
                )
                impostors.append(impostor_text)
            except Exception as e:
                logging.warning(f"Error generating impostor with {paraphraser}: {e}")

        return impostors
