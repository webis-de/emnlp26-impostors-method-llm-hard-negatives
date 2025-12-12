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
# FIXME: make compatible with new mongodb idea
import os
import sys
from abc import ABC
from typing import Optional

from dotenv import load_dotenv

from genai_detection.detectors.components.preprocessing import Preprocessor
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )

class BaseImpostorGenerator(ABC):
    """Abstract base class for generating impostors."""

    def __init__(self, n_impostors: int):
        """
        :param n_impostors: number of impostors to generate
        """
        self.n_impostors = n_impostors
        # optional; Koppel et al. (2014) select N random impostors among top M impostors with on-the-fly and in-data
        # impostors generation method
        self.num_potential_impostors = n_impostors

    def set_num_potential_impostors(self, num_potential_impostors: int):
        assert isinstance(num_potential_impostors, int), (f"num_potential_impostors must an integer, "
                                                          f"got {type(num_potential_impostors)}")
        self.num_potential_impostors = max(num_potential_impostors, self.n_impostors)

class MongoDBSavedGenerator(BaseImpostorGenerator):
    def __init__(
        self, n_impostors: int
    ):
        super().__init__(n_impostors)
        self.mongoDB = ParaphraseMongoDB()
        self.text_processor = Preprocessor()

    def generate_impostors(
            self, text: Optional[str], text_id: Optional[str]
    ):
        pass

    def generate_impostors_by_text_id(
            self, text_id: str
    ):
        text, text_id = self.mongoDB.get_text_or_id_from_orginal_collection(
            text=None, text_id=text_id
        )
        return self.generate_impostors(text=text, text_id=text_id)

class GenerativeImpostorGenerator(MongoDBSavedGenerator):
    def __init__(self, n_impostors: int):
        super().__init__(n_impostors)

    def obtain_existing_paraphrases(self, collection, search_args: dict):
        if "text_id" not in search_args.keys():
            logging.info(f"Must provide text_id, but only provides {search_args.keys()}.")
            return [], {}
        else:
            cursor = self.mongoDB.find_document_by_multiple_fields(collection=collection, search_args=search_args)
            docs = list(cursor)  # materialize once, safe if the number is small

            impostor_field_name = "impostor_text" if "index" in search_args.keys() else "paraphrase"
            impostors = [doc[impostor_field_name] for doc in docs]
            extracted_info = next(
                (doc["extracted_info"] for doc in docs if "extracted_info" in doc), {}
            )
            return impostors, extracted_info


class LLMImpostorGenerator(GenerativeImpostorGenerator):
    def __init__(
        self, n_impostors: int
    ):
        super().__init__(n_impostors)


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
    for i in range(3):
        text, text_id = llm_paraphraser.mongoDB.get_text_or_id_from_orginal_collection(
            text=None, text_id="68f50029edacdf3d5c0279ea"
        )
        logging.info("%s", text[:200])
    # imps = llm_paraphraser.generate_impostors(text_id="68f50029edacdf3d5c0279eb", text=None)
    # for i, imp in enumerate(imps):
    #     # $7,886.76 07.11.25, 10.04 Uhr
    #     # $7,886.75 07.11.25, 10.49 Uhr
    #     # $7,886.71 07.11.25, 12.49 Uhr
    #     # FIXME: number of impostors does not work properly, too many and too short paraphrases
    #     logging.info(f"imp number {i} of length {len(imp.split())}")
