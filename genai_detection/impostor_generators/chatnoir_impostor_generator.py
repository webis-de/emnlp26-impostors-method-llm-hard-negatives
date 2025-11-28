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
from typing import List, Dict

import requests

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.search_generator_base import SearchImpostorGeneratorBase

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )


class ChatNoirSearchImpostorGenerator(SearchImpostorGeneratorBase):

    def __init__(
        self,
        api_key: str=CONFIG.CHATNOIR_KEY,
        n_impostors: int = 50,
        results_per_query: int = 25,
        max_workers: int = 2,
        n_min_words: int = 3,
        n_max_words: int = 5,
        index_name: str = "msmarco-v2.1"
    ):
        """
        Configuration from Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’. Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954 is:
        - num_queries: 50
        - results_per_query: 25
        - max_workers: -
        - n_min_words: 3
        - n_max_words: 5

        :param api_key: API key for ChatNoir Search API
        :param results_per_query: Number of results to fetch per query (default: 25)
        :param max_workers: Maximum number of threads to use for parallel fetching (default: 2)
        :param n_min_words: Minimum number of words in a query (default: 3)
        :param n_max_words: Maximum number of words in a query (default: 5)
        :param index_name: Name of the index to use (default: `msmarco`).
        """
        super().__init__(n_impostors=n_impostors, api_key=api_key, results_per_query=results_per_query,
         max_workers=max_workers, n_max_words=n_max_words, n_min_words=n_min_words,index_name=index_name)

    def fetch_results(self, query: str) -> List[Dict]:
        """
        Fetches search results for a given query using the ChatNoir Search API.
        :param query: Search query to fetch results for
        :return: List of dictionaries containing search results with keys: 'query', 'title', 'url', 'snippet', and 'position'
        """
        assert query, "No query to fetch results for."
        if not self.api_key:
            raise ValueError("API key for SerpAPI is not provided.")
        try:
            base_url = "https://www.chatnoir.eu/api/v1/_search"
            params = {
                "apikey": self.api_key, "query": query,
                  "size": self.results_per_query,
                  "index": self.index_name,
                  "pretty": True,
                  "minimal": True,
                  "search_method": "default"
              }

            response = requests.get(base_url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.warning(f"Error fetching results for '{query}': {e}")
            return []

    # def generate_impostors(
    #     self, text: Optional[str], text_id:Optional[str]
    # ) -> pd.DataFrame:
    #     """
    #     Generates impostors for the given input text using Google search results.
    #
    #     Steps:
    #     1. Extract medium-frequency words from the input text.
    #     2. Formulate search queries using random combinations of those words.
    #     3. Use the ChatNoir to retrieve search result snippets.
    #     4. Save the results to the mongoDB database collection for queried impostors.
    #     5. Format results into a dictionary with keys as query and position, and values as the full text (or snippets) of the search result.
    #
    #     While Koppel et al. (2014) randomly choose n imposters among the top m imposter, we use all of them.
    #
    #     :param text: Input text to generate impostors for.
    #     :param text_id: The id of the original text in the original text collection.
    #
    #     :return: DataFrame containing search results with columns: 'query', 'title', 'url', 'snippet' (i.e., short content summary of search result), and 'position' (i.e. number of result in the search results)
    #
    #     References:
    #     ===========
    #     Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’. Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.
    #     """
    #     if not isinstance(text, str):
    #         text, text_id = self.text_processor.obtain_text_and_id(value=text)
    #         assert text.strip(), "Input text must be a non-empty string."
    #
    #     impostors = []
    #     if text_id is not None:
    #         cursor = self.mongoDB.find_document_by_multiple_fields(collection=self.mongoDB.on_the_fly_collection,
    #                                                                search_args={"text_id": text_id, "index":self.index_name})
    #         docs = list(cursor)  # materialize once, safe if the number is small
    #
    #         impostors = [doc["impostor_text"] for doc in docs if "impostor_text" in doc]
    #
    #         n_imp_to_generate = self.n_impostors - len(impostors)
    #         if n_imp_to_generate <= 0:
    #             logging.info(
    #                 "Number of impostors in mongodb collection: {} for text with ID: {}. No need to generate more impostors, just returning {} impostors.".format(
    #                     len(impostors), text_id, self.n_impostors
    #                 )
    #             )
    #             return impostors[: self.n_impostors]
    #         self.n_impostors = n_imp_to_generate
    #
    #     medium_frequency_words = self.get_medium_frequency_words(text)
    #     queries = self._generate_queries_based_on_candidate_words(
    #         medium_frequency_words
    #     )
    #     try:
    #         # also saves new impostors in the mongoDB collection
    #         impostors.extend(
    #             self._parallel_fetch(queries=queries, text=text, text_id=text_id)
    #         )
    #         return impostors # list of impostor texts
    #
    #     except Exception as e:
    #         logging.warning(
    #             f"Error during fetching results: {e}"
    #         )
    #         return {"imposter": "Error during fetching results, check logs."}


if __name__ == "__main__":
    chat_noir_retriever = ChatNoirSearchImpostorGenerator(api_key=CONFIG.CHATNOIR_KEY)
    # logging.info("%s", chat_noir_retriever.fetch_results(query="cats"))
    logging.info("%d", len(chat_noir_retriever.generate_impostors(text="cats dogs animals", text_id=None))
                 )
