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
import random
import re
from collections import Counter
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup
from nltk import download

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.ImpostorGenerator import (
    BaseImpostorGenerator,
)
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB


class ChatNoirSearchImpostorGenerator(BaseImpostorGenerator):

    def __init__(
        self,
        api_key: str=CONFIG.CHATNOIR_KEY,
        num_queries: int = 50,
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
        :param num_queries: Number of queries to generate (default: 2)
        :param results_per_query: Number of results to fetch per query (default: 25)
        :param max_workers: Maximum number of threads to use for parallel fetching (default: 2)
        :param n_min_words: Minimum number of words in a query (default: 3)
        :param n_max_words: Maximum number of words in a query (default: 5)
        :param index_name: Name of the index to use (default: `msmarco`).
        """
        super().__init__(n_impostors=num_queries * results_per_query)
        self.api_key = api_key
        self.num_queries = num_queries
        self.results_per_query = results_per_query
        self.max_workers = max_workers
        self.n_min_words = n_min_words
        self.n_max_words = n_max_words
        self.index_name = index_name
        assert (
            self.n_min_words < self.n_max_words
        ), "n_min_words must be less than n_max_words"
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print(
                "Spacy model 'en_core_web_sm' not found. Downloading it now. This may take a while."
            )
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        self.mongoDB = ParaphraseMongoDB()

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
            print(f"Error fetching from URL {url}: {e}")
            return ""

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
            print(f"Error fetching results for '{query}': {e}")
            return []

    def _parallel_fetch(self, queries: List[str], text:Optional[str], text_id:Optional[str]) -> List[Dict]:
        """
        Fetches results for multiple queries in parallel using a thread pool.
        Impostor texts are crawled websites, stripped of HTML tags if possible and provided snippets by search engine if crawling was not possible.
        :param queries: List of search queries to fetch results for
        :param text: Text of the original text (used to derive the query from), only used as metadata when storing result in mongoDB collection.
        :param text_id: Id of the original text (used to derive the query from), only used as metadata when storing result in mongoDB collection.
        :return: List of impostor texts. # TODO: delete this: dictionaries containing search results for all queries
        """
        all_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.fetch_results, query): query for query in queries
            }
            for future in as_completed(futures):
                query = futures[future]   # Retrieve which query produced this future
                new_imps = future.result()["results"]

                for new_imp in new_imps:
                #     for key in [
                #         "warc_id",
                #         "score",
                #         "cache_uri",
                #         "target_hostname",
                #         "crawl_date",
                #         "page_rank",
                #         "spam_rank",
                #         "content_type",
                #     ]:
                #         new_imp.pop(key, None)  # removes key if exists, ignores if not
                    # lang not in keys if the retrieval flag "minimal" is true
                    if ("lang" not in new_imp.keys()) or (new_imp["lang"] == "en"):
                        uri = new_imp["target_uri"]
                        extracted_text = self._extract_text_from_url(uri)
                        impostor_text = new_imp["snippet"] if extracted_text == "" else extracted_text
                        self._save_on_the_fly_in_mongodb(
                            original_text=text,
                            original_text_id=text_id,
                            impostor_text=impostor_text,
                            query=query,
                            index=new_imp["index"],
                            uri=uri
                        )
                        all_results.append(impostor_text)
        return all_results

    def _save_on_the_fly_in_mongodb(self, original_text_id:str, original_text:str, impostor_text:str, query:str, index:str, uri:str) -> None:
        """
        Save the paraphrase to the MongoDB database collection called "paraphrase".
        :param original_text_id: The id of the original text in the original_text collection.
        :param original_text: The original text as a string.
        :param impostor_text: The text of the impostor as a string.
        :param query: The query as a string.
        :param index: The index of the crawled impostor text.
        :param uri: The uri to the crawled impostor text.
        :return:-
        """
        assert (
            type(impostor_text) == str
        ), f"paraphrased_text must be a str, but is of type {type(impostor_text)}"
        assert len(impostor_text.split()) > 0, "paraphrased_text is empty"

        # Build the new document
        paraphrase_doc = {
            # Do not use text_id as _id since a text will be paraphrased multiple times with different settings
            "text_id": original_text_id,  # ID of the original text document
            "length_original_text": len(original_text.split()),
            "length_impostor_text": len(impostor_text.split()),
            "query": query,
            "impostor_text": impostor_text,
            "index": index,
            "uri": uri,
            "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }

        # Insert into document into paraphrase collection
        self.mongoDB.on_the_fly_collection.insert_one(paraphrase_doc)
        print(f"Inserted paraphrase for document ID: {original_text_id}")

    def generate_impostors(
        self, text: Optional[str], text_id:Optional[str]
    ) -> pd.DataFrame:
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
        # TODO: get text from text ID if not given
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        medium_frequency_words = self.get_medium_frequency_words(text)
        queries = self._generate_queries_based_on_candidate_words(
            medium_frequency_words
        )
        try:
            # also saves new impostors in the mongoDB collection
            return self._parallel_fetch(queries=queries, text=text, text_id=text_id) # list of impostor texts
            # result_df = pd.DataFrame(self._parallel_fetch(queries))
            # result_df.drop_duplicates(subset="target_uri", inplace=True)
            #
            # result_df.drop(axis="columns", columns=["warc_id", "score", "cache_uri", "target_hostname", "crawl_date", "page_rank", "spam_rank", "content_type"], inplace=True, errors="ignore")
            # result_df = result_df[result_df["lang"] == "en"]
            # print(f"Found {len(result_df)} results.")
            # result_df.to_csv('out.csv', index=False)
            # print(result_df.head())
            #
            # if result_df.empty:
            #     print("Warning: No results fetched. CSV not saved.")
            #     return result_df

        except Exception as e:
            print(
                f"Error during fetching results: {e}"
            )
            return {"imposter": "Error during fetching results, check logs."}

        # return result_df


if __name__ == "__main__":
    chat_noir_retriver = ChatNoirSearchImpostorGenerator(api_key=CONFIG.CHATNOIR_KEY)
    # print(chat_noir_retriver.fetch_results(query="cats"))
    print(chat_noir_retriver.generate_impostors(text="cats dogs animals", text_id=None))
