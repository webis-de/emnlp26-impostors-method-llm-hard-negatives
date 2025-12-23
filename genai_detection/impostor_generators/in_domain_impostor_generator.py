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

from bson import ObjectId

from genai_detection.impostor_generators.ImpostorGenerator import (
    MongoDBSavedGenerator,
)

logger = logging.getLogger(__name__)

class InDomainImpostorGenerator(MongoDBSavedGenerator):
    def __init__(self, n_impostors: int, dataset_name: str):
        """
        This impostor generator gets in-domain impostors from the dataset the input texts originate from.

        :param n_impostors: Number of impostors to generate.
        :param dataset_name: Name of dataset from which impostors are sampled from.

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’. Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.
        """
        super().__init__(n_impostors=n_impostors)
        unique_dataset_names_saved = self.mongoDB.original_collection.distinct("dataset")
        assert dataset_name in unique_dataset_names_saved, f"{dataset_name} not in {unique_dataset_names_saved}"
        self.dataset_name=dataset_name
        logging.info(f"InDomainImpostorGenerator: Dataset name: {self.dataset_name}")

    def generate_impostors(
        self, text: str, text_id: Optional[str]=None
    ) -> List[str]:
        """
        Generates in-domain impostors from a pre-defined dataset.
        :param text: Input text to generate impostors for (not used in this implementation).
        :param text_id: Optional text ID of the text for which the impostors should be generated.
        :return: Dictionary of impostors with keys as ids and values as texts.
        """
        text_id = ObjectId(text_id)
        print("Run in-domain impostors generator with text_id", text_id, type(text_id))
        original_text = self.mongoDB.original_collection.find_one({"_id": text_id})
        # ensure text ID not same
        search_args = {"dataset": self.dataset_name, "task_description":original_text["task_description"], "id": {"$ne": text_id}}
        print(f"InDomainImpostorGenerator: Search arguments: {search_args}")

        # use this, if the returned impostors should be completely random in-domain texts
        # cursor = self.mongoDB.get_random_matching_documents_from_collection(collection=self.mongoDB.original_collection, search_args=search_args, num_samples=self.n_impostors)
        # impostors = [doc["text"] for doc in cursor]
        # return impostors

        # use this, if the returned impostors should be random among the most similar in-domain texts
        cursor = self.mongoDB.find_document_by_multiple_fields(collection=self.mongoDB.original_collection,search_args=search_args).limit(self.num_potential_impostors)
        impostors = [doc["text"] for doc in cursor]
        print("Num imps", len(impostors))
        return self._select_random_n_imps_among_best_m_potential_impostors(all_impostors=impostors, reference_text=text)
