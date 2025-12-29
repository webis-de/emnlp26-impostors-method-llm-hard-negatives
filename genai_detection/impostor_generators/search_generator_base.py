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
import random
import re
from collections import Counter
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import requests
import spacy
from bs4 import BeautifulSoup
from nltk import download

from genai_detection.impostor_generators.ImpostorGenerator import GenerativeImpostorGenerator

logger = logging.getLogger(__name__)

class SearchImpostorGeneratorBase(GenerativeImpostorGenerator):

    def __init__(
        self,
        api_key: str,
        index_name:str,
        n_impostors: int = 50,
        results_per_query: int = 25,
        max_workers: int = 2,
        n_min_words: int = 3,
        n_max_words: int = 5,
    ):
        """
        Configuration from Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’. Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954 is:
        - num_queries: 50
        - results_per_query: 25
        - max_workers: -
        - n_min_words: 3
        - n_max_words: 5

        :param api_key: API key for ChatNoir Search API
        :param index_name: Index name of MongoDB collection
        :param results_per_query: Number of results to fetch per query (default: 25)
        :param max_workers: Maximum number of threads to use for parallel fetching (default: 2)
        :param n_min_words: Minimum number of words in a query (default: 3)
        :param n_max_words: Maximum number of words in a query (default: 5)

        """
        super().__init__(n_impostors=n_impostors)
        self.api_key = api_key
        self.results_per_query = results_per_query
        assert isinstance(results_per_query, int) and results_per_query > 1, f"results_per_query must be a positive integer: {results_per_query}"
        self.num_queries = max(1, self.num_potential_impostors // results_per_query)
        self.max_workers = max_workers
        assert isinstance(n_min_words, int) and isinstance(n_max_words, int), f"Type of n_min_words or n_max_words is not int, but n_min_words: {type(n_min_words)}/ n_max_words: {type(n_max_words)}"
        assert (
            n_min_words < n_max_words
        ), "n_min_words must be less than n_max_words"
        self.n_min_words = n_min_words
        self.n_max_words = n_max_words
        self.index_name = index_name

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning(
                "Spacy model 'en_core_web_sm' not found. Downloading it now. This may take a while."
            )
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def get_medium_frequency_words(
        self, text: str, lower_pct: int = 30, upper_pct: int = 70
    ) -> List[str]:
        """
        Extracts medium-frequency words from the input text.
        Medium frequency words are defined as those that fall between the lower and upper percentiles of word frequencies in the text.
        :param text: Input text to extract medium frequency words from
        :param lower_pct: Lower percentile threshold for word frequency (default: 40)
        :param upper_pct: Upper percentile threshold for word frequency (default: 60)
        :return: List of medium frequency words
        """
        doc = self.nlp(text.lower())
        words = [
            token.text for token in doc if token.is_alpha and not token.is_stop
        ]  # filter out stop words and non-alphabetic tokens
        freq_counter = Counter(words)
        freqs = np.array(list(freq_counter.values()))

        # Compute dynamic thresholds
        low_thresh = np.percentile(freqs, lower_pct)
        high_thresh = np.percentile(freqs, upper_pct)
        return [
            word
            for word, count in freq_counter.items()
            if low_thresh <= count <= high_thresh
        ]  # medium frequency words

    def _generate_queries_based_on_candidate_words(
        self, candidate_words: List[str]
    ) -> List[str]:
        """
        Generates a list of queries using random combinations of candidate words. Each query is a string and will have a random number of words between n_min_words and n_max_words separated by whitespaces.
        :param candidate_words: list of candidate words to use for generating queries
        :return: list of generated queries
        """
        queries = []
        assert (
            len(candidate_words) >= self.n_min_words
        ), f"Not enough candidate words to generate queries, at {self.n_min_words} necessary."
        for _ in range(self.num_queries):
            n_words = min(
                len(candidate_words), random.randint(self.n_min_words, self.n_max_words)
            )
            query_words = random.sample(candidate_words, n_words)
            queries.append(" ".join(query_words))
        return queries

    def _extract_text_from_url(self, url: str) -> str:
        """
        Extracts and cleans main textual content from a webpage.

        This function fetches the content of a given URL, removes non-informative
        HTML elements such as <script>, <style>, <header>, <footer>, and normalizes
        the text by collapsing all whitespace (tabs, newlines, multiple spaces) into
        single spaces. It only keeps the text found within paragraph <p> tags.

        Punctuation is preserved, but all sequences of whitespace are reduced to a
        single space to make the text layout-agnostic and suitable for further
        processing.

        :param url: The URL of the webpage to fetch and process.
        :return: Cleaned textual content from the webpage, with only useful body text
             preserved and whitespace normalized.
        """
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            # Remove unwanted tags
            for tag in soup(
                [
                    "script",
                    "style",
                    "header",
                    "footer",
                    "nav",
                    "aside",
                    "form",
                    "noscript",
                ]
            ):
                tag.decompose()

            # Extract paragraph text
            paragraphs = soup.find_all("p")
            full_text = " ".join(p.get_text() for p in paragraphs)
            # Normalize all whitespace (tabs, newlines, multiple spaces) to a single space
            cleaned_text = re.sub(r"\s+", " ", full_text)
            return cleaned_text.strip()
        except Exception as e:
            logging.warning(f"Error fetching from URL {url}: {e}")
            return ""

    def fetch_results(self, query: str) -> List[Dict]:
        """
        Fetches search results for a given query using the SerpAPI Google Search API.
        :param query: Search query to fetch results for
        :return: List of dictionaries containing search results with keys: 'query', 'title', 'url', 'snippet', and 'position'
        """
        pass

    def _save_on_the_fly_in_mongodb(self, original_text_id:str, original_text:str, impostor_text:str, query:str, uri:str) -> None:
        """
        Save the paraphrase to the MongoDB database collection called "paraphrase".
        :param original_text_id: The id of the original text in the original_text collection.
        :param original_text: The original text as a string.
        :param impostor_text: The text of the impostor as a string.
        :param query: The query as a string.
        :param uri: The uri to the crawled impostor text.
        :param index: The index of the crawled impostor text (Google if SerpAPI).
        :return:-
        """
        assert (
            type(impostor_text) == str
        ), f"paraphrased_text must be a str, but is of type {type(impostor_text)}"
        assert len(impostor_text.split()) > 0, "paraphrased_text is empty"

        MAX_BYTES = 14 * 2**20  # 14 MB buffer
        # mongoDB collection allows at most 16 MB per document (puffer for features other than text)
        impostor_text_bytes = impostor_text.encode("utf-8")[:MAX_BYTES]
        impostor_text_safe = impostor_text_bytes.decode("utf-8", errors="ignore")

        # Build the new document
        paraphrase_doc = {
            # Do not use text_id as _id since a text will be paraphrased multiple times with different settings
            "text_id": original_text_id,  # ID of the original text document
            "length_original_text": len(original_text.split()),
            "length_impostor_text": len(impostor_text.split()),
            "query": query,
            "impostor_text": impostor_text_safe,
            "uri": uri,
            "index": self.index_name,
            "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        }

        # Insert into document into paraphrase collection
        self.mongoDB.on_the_fly_collection.insert_one(paraphrase_doc)


    def _parallel_fetch(
        self, queries: List[str], text: Optional[str], text_id: Optional[str]
    ) -> List[Dict]:
        """
        Fetches results for multiple queries in parallel using a thread pool.
        Impostor texts are crawled websites, stripped of HTML tags if possible and provided snippets by search engine if crawling was not possible.
        :param queries: List of search queries to fetch results for
        :param text: Text of the original text (used to derive the query from), only used as metadata when storing result in mongoDB collection.
        :param text_id: Id of the original text (used to derive the query from), only used as metadata when storing result in mongoDB collection.
        :return: List of impostor texts.
        """
        all_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.fetch_results, query): query for query in queries
            }
            for future in as_completed(futures):
                query = futures[future]  # Retrieve which query produced this future
                new_imps = future.result()

                for new_imp in new_imps:
                    if ("lang" not in new_imp.keys()) or (new_imp["lang"] == "en"):
                        uri = new_imp["target_uri"]
                        extracted_text = self._extract_text_from_url(uri)
                        impostor_text = (
                            new_imp["snippet"]
                            if extracted_text == ""
                            else extracted_text
                        )
                        self._save_on_the_fly_in_mongodb(
                            original_text=text,
                            original_text_id=text_id,
                            impostor_text=impostor_text,
                            query=query,
                            uri=uri,
                        )
                        all_results.append(impostor_text)
                logging.info(f"Inserted {len(new_imps)} search results for document ID: {text_id} on index {self.index_name}")
        return all_results

    def generate_impostors(
        self, text: Optional[str], text_id:Optional[str]
    ) -> List[str]:
        """
        Generates impostors for the given input text using Google search results.

        Steps:
        1. Extract medium-frequency words from the input text.
        2. Formulate search queries using random combinations of those words.
        3. Use the ChatNoir to retrieve search result snippets.
        4. Save the results to the mongoDB database collection for queried impostors.
        5. Format results into a dictionary with keys as query and position, and values as the full text (or snippets) of the search result.

        While Koppel et al. (2014) randomly choose n imposters among the top m imposter, we use all of them.

        :param text: Input text to generate impostors for.
        :param text_id: The id of the original text in the original text collection.

        :return: DataFrame containing search results with columns: 'query', 'title', 'url', 'snippet' (i.e., short content summary of search result), and 'position' (i.e. number of result in the search results)

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’. Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.
        """
        if not isinstance(text, str):
            text, text_id = self.text_processor.obtain_text_and_id(value=text_id)
            assert text.strip(), "Input text must be a non-empty string."

        impostors, _ = self.obtain_existing_paraphrases(
            collection=self.mongoDB.on_the_fly_collection,
            search_args={"text_id": text_id, "index": self.index_name},
        )
        n_imp_to_generate = self.n_impostors - len(impostors)
        print(f"Found {len(impostors)}/{self.n_impostors} impostors.")
        if n_imp_to_generate <= 0:
            logging.info(
                "Number of impostors in mongodb collection: {} for text with ID: {}. No need to generate more impostors, just returning {} impostors.".format(
                    len(impostors), text_id, self.n_impostors
                )
            )
            return impostors[: self.n_impostors]
        # self.n_impostors = n_imp_to_generate  # no good idea, the call after will generate to few

        medium_frequency_words = self.get_medium_frequency_words(text)
        queries = self._generate_queries_based_on_candidate_words(
            medium_frequency_words
        )
        try:
            # also saves new impostors in the mongoDB collection
            logger.info("About to fetch results.")
            new_imps = self._parallel_fetch(queries=queries, text=text, text_id=text_id)
            impostors.extend(new_imps)
            logger.info(f"Fetched {len(new_imps)} results (total of {len(impostors)}/{self.n_impostors} impostors), "
                        f"need to subsample: {len(impostors) > self.n_impostors}.")
            # list of impostor texts
            return self._select_random_n_imps_among_best_m_potential_impostors(all_impostors=impostors,
                                                                               reference_text=text)

        except Exception as e:
            logging.warning(
                f"Error during fetching results: {e}"
            )
            return impostors
