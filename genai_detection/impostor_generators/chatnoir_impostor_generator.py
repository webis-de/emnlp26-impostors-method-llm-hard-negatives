# Copyright 2026 Klara M. Gutekunst, Webis
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

class ChatNoirSearchImpostorGenerator(SearchImpostorGeneratorBase):

    def __init__(
        self,
        top_n_freq_words:int,
        api_key: str=CONFIG.CHATNOIR_KEY,
        n_impostors: int = 50,
        results_per_query: int = 25,
        max_workers: int = 2,
        n_min_words: int = 3,
        n_max_words: int = 5,
        index_name: str = CONFIG.RETRIEVAL_INDEX_TRANSLATIONS["on_the_fly_chatnoir"]
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
        super().__init__(
            n_impostors=n_impostors,
            top_n_freq_words=top_n_freq_words,
            api_key=api_key,
            results_per_query=results_per_query,
            max_workers=max_workers,
            n_max_words=n_max_words,
            n_min_words=n_min_words,
            index_name=index_name
        )

    def fetch_results(self, query: str) -> List[Dict]:
        """
        Fetches search results for a given query using the ChatNoir Search API.
        Find different indices here: https://www.chatnoir.eu/docs/api-general#indices (12.03.2026)
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
            return response.json()["results"]
        except Exception as e:
            logging.warning(f"Error fetching results for '{query}' with ChatNoir: {e}")
            return []


if __name__ == "__main__":
    chat_noir_retriever = ChatNoirSearchImpostorGenerator(api_key=CONFIG.CHATNOIR_KEY)
    # logging.info("%s", chat_noir_retriever.fetch_results(query="cats"))
    logging.info("%d", len(chat_noir_retriever.generate_impostors(text="cats dogs animals", text_id=None))
                 )
