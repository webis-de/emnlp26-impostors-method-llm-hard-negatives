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

import serpapi

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.search_generator_base import SearchImpostorGeneratorBase

logger = logging.getLogger(__name__)

class GoogleSearchImpostorGenerator(SearchImpostorGeneratorBase):

    def __init__(
        self,
        top_n_freq_words:int,
        api_key: str=CONFIG.SERPAPI_KEY,
        num_queries: int = 1,
        results_per_query: int = 25,
        max_workers: int = 2,
        n_min_words: int = 3,
        n_max_words: int = 5,
    ):
        """
        n_impostors <= num_queries * results_per_query

        Configuration from Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’. Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954 is:
        - num_queries: 50
        - results_per_query: 25
        - max_workers: -
        - n_min_words: 3
        - n_max_words: 5

        :param api_key: API key for SerpAPI Google Search API
        :param num_queries: number of queries to generate (default: 2)
        :param results_per_query: number of results to fetch per query (default: 25)
        :param max_workers: maximum number of threads to use for parallel fetching (default: 2)
        :param n_min_words: minimum number of words in a query (default: 3)
        :param n_max_words: maximum number of words in a query (default: 5)
        :param real_time_generation: whether to use real time generation (default: False)
        """
        super().__init__(
            n_impostors=num_queries * results_per_query,
            top_n_freq_words=top_n_freq_words,
            api_key=api_key,
            results_per_query=results_per_query,
            max_workers=max_workers,
            n_max_words=n_max_words,
            n_min_words=n_min_words,
            index_name=CONFIG.RETRIEVAL_INDEX_TRANSLATIONS["on_the_fly_serpapi"],
        )

    def fetch_results(self, query: str) -> List[Dict]:
        """
        Fetches search results for a given query using the SerpAPI Google Search API.
        :param query: Search query to fetch results for
        :return: List of dictionaries containing search results with keys: 'query', 'title', 'url', 'snippet', and 'position'
        """
        assert query, "No query to fetch results for."
        if not self.api_key:
            raise ValueError("API key for SerpAPI is not provided.")
        try:
            params = {
                "q": query,  # the search query, eg. "coffee"
                "num": self.results_per_query,  # default: 10
                "api_key": self.api_key,
                "engine": "google_light",  # might return fewer results than 'google' engine, but is faster
                "safe": "off",  # disbale filtering out adult content
                "nfpr": 0,  # include results from auto-corrected query for misspellings
                "devices": "desktop",  # use desktop results
            }
            search = serpapi.search(params)
            logging.info("Obtained SERPAPI search results.")
            results = search.as_dict()
            return [
                {
                    "query": query,
                    "target_uri": res.get("link"),
                    "snippet": res.get("snippet"),
                }
                for res in results.get("organic_results", [])
            ]
        except Exception as e:
            logging.warning(f"Error fetching results for '{query}': {e}")
            return []

if __name__ == "__main__":
    google_retriever = GoogleSearchImpostorGenerator(api_key=CONFIG.SERPAPI_KEY)
    # logging.info("%s", google_retriever.fetch_results(query="cats"))
    logging.info("%d", len(google_retriever.generate_impostors(text="cats dogs animals", text_id=None))                )
