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
from genai_detection.paraphrasing.two_step_paraphrasers import TwoStepParaphraser

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )

class TwoStepImpostorGenerator(NonNaiveLLMImpostorGenerator):
    def __init__(self, n_impostors: int):
        """
        Naive LLM-based impostor generator that uses no naive paraphrasers (i.e. only two-step paraphrasers).
        :param n_impostors: number of impostors to generate
        """
        super().__init__(
            n_impostors=n_impostors,
        )
        self.two_step_paraphraser = TwoStepParaphraser()

    def generate_impostors(
        self, text: Optional[str], text_id: Optional[str]
    ) -> List[str]:
        # look at base class
        # text, text_id = self.mongoDB.get_text_or_id_from_orginal_collection(text, text_id)

        # returns a list of impostor texts with n_impostors impostors
        extracted_info = {}
        n_imp_to_generate = self.n_impostors
        impostors = []
        if text_id is not None:
            cursor = self.mongoDB.find_paraphrases(document_id=text_id)
            docs = list(cursor)  # materialize once, safe if the number is small

            impostors = [doc["paraphrase"] for doc in docs if "paraphrase" in doc]
            extracted_info = next(
                (doc["extracted_info"] for doc in docs if "extracted_info" in doc), {}
            )

            n_imp_to_generate -= len(impostors)
            if n_imp_to_generate <= 0:
                logging.info("Number of impostors in mongodb collection: {} for text with ID: {}. No need to generate more impostors, just returning {} impostors.".format(len(impostors), text_id, self.n_impostors))
                return impostors[: self.n_impostors]

        logging.info(
            f"{len(impostors)} precomputed impostors found in mongoDB. Generating {n_imp_to_generate} impostors for text {text[:100]}..."
        )
        self.two_step_paraphraser.set_n_paraphrases(n_paraphrases=n_imp_to_generate)
        if not extracted_info:
            extracted_info, new_impostors, total_cost = self.two_step_paraphraser.paraphrase(text=text)
        else:
            logging.info("Using already extracted information stored in mongoDB for paraphrase generation.")
            new_impostors, total_cost = self.two_step_paraphraser.generate_multiple_paraphrase_based_on_extracted_information(verbose=False, extracted_info=extracted_info)

        assert isinstance(
            extracted_info, dict
        ), f"The extracted_info must be a dictionary but is of type {type(extracted_info)}."
        assert isinstance(
            new_impostors, (list, str)
        ), f"The new_impostors must be a list or a single string but is of type {type(new_impostors)}."
        assert isinstance(
            total_cost, float
        ), f"The total_cost of a paraphrase must be of type float but is of type {type(total_cost)}."
        if isinstance(new_impostors, str):
            new_impostors = [new_impostors]

        # Save new paraphrases in MongoDB
        for imp in new_impostors:
            self.two_step_paraphraser.save_paraphrase_in_mongodb(
                original_text=text,
                original_text_id=text_id,
                paraphrased_text=imp,
                extracted_info=extracted_info,
                total_costs=total_cost / len(new_impostors),
                temperature=1.0,  # Temperature requirements for reasoning models like gpt-5-nano
            )
        if impostors:
            new_impostors.extend(impostors)
        return new_impostors
