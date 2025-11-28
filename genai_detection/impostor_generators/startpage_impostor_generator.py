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

from startpage import StartPage

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.search_generator_base import SearchImpostorGeneratorBase

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )


class StartPageSearchImpostorGenerator(SearchImpostorGeneratorBase):

    def __init__(
        self,
        api_key: str=CONFIG.CHATNOIR_KEY,
        n_impostors: int = 50,
        results_per_query: int = 25,
        max_workers: int = 2,
        n_min_words: int = 3,
        n_max_words: int = 5,
        index_name: str = "Google-Startpage"
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
            api_key=api_key,
            results_per_query=results_per_query,
            max_workers=max_workers,
            n_max_words=n_max_words,
            n_min_words=n_min_words,
            index_name=index_name
        )
        self.max_page = 1 # page 1 until max_page

    def fetch_results(self, query: str) -> List[Dict]:
        """
        Fetches search results for a given query using the ChatNoir Search API.
        :param query: Search query to fetch results for
        :return: List of dictionaries containing search results with keys: 'query', 'title', 'url', 'snippet', and 'position'
        """
        assert query, "No query to fetch results for."
        task = StartPage()
        task.search(word=query, page=self.max_page)
        logging.info(f"Fetched {len(task.results)} results for {query}:\n{task.results}\n\n")
        return task.results.values()


# FIXME: Does not work
if __name__ == "__main__":
    task = StartPage()
    task.search(word="Hello World", page=1)
    logging.info(
        f"Fetched results for Hello World:\n{task.results}\n\n"
    )

    startpage_google_retriever = StartPageSearchImpostorGenerator(api_key=CONFIG.CHATNOIR_KEY)
    logging.info("%s", startpage_google_retriever.fetch_results(query="Hello World"))
    # logging.info("%d", len(startpage_google_retriever.generate_impostors(text="cats dogs animals", text_id=None))
    #              )
