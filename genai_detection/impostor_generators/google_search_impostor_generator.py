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
import datetime
import logging
import random
import re
from collections import Counter
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import serpapi
import spacy
from bs4 import BeautifulSoup
from google.auth.transport import requests
from nltk import download

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.ImpostorGenerator import (
    BaseImpostorGenerator,
)

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )

class GoogleSearchImpostorGenerator(BaseImpostorGenerator):

    def __init__(
        self,
        api_key: str,
        num_queries: int = 1,
        results_per_query: int = 25,
        max_workers: int = 2,
        n_min_words: int = 3,
        n_max_words: int = 5,
        real_time_generation: bool = False,
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
        super().__init__(n_impostors=num_queries * results_per_query)
        self.api_key = api_key
        self.num_queries = num_queries
        self.results_per_query = results_per_query
        self.max_workers = max_workers
        self.n_min_words = n_min_words
        self.n_max_words = n_max_words
        self.real_time_generation = real_time_generation
        assert (
            self.n_min_words < self.n_max_words
        ), "n_min_words must be less than n_max_words"
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
        :param text: input text to extract medium frequency words from
        :param lower_pct: lower percentile threshold for word frequency (default: 30)
        :param upper_pct: upper percentile threshold for word frequency (default: 70)
        :return: list of medium frequency words
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
        :param query: search query to fetch results for
        :return: list of dictionaries containing search results with keys: 'query', 'title', 'url', 'snippet', and 'position'
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
            results = search.as_dict()
            return [
                {
                    "query": query,
                    "title": res.get("title"),
                    "url": res.get("link"),
                    "snippet": res.get("snippet"),
                    "rich_snippet": res.get("rich_snippet", ""),
                    "author": res.get("author", "unknown"),
                    "position": res.get("position"),
                    "full_text": self._extract_text_from_url(res.get("link")),
                }
                for res in results.get("organic_results", [])
            ]
        except Exception as e:
            logging.warning(f"Error fetching results for '{query}': {e}")
            return []

    def _parallel_fetch(self, queries: List[str]) -> List[Dict]:
        """
        Fetches results for multiple queries in parallel using a thread pool.
        :param queries: list of search queries to fetch results for
        :return: list of dictionaries containing search results for all queries
        """
        all_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.fetch_results, query): query for query in queries
            }
            for future in as_completed(futures):
                all_results.extend(future.result())
        return all_results

    def generate_impostors(
        self, text: str
    ) -> pd.DataFrame:
        """
        Generates impostors for the given input text using Google search results.

        Steps:
        1. Extract medium-frequency words from the input text.
        2. Formulate search queries using random combinations of those words.
        3. Use the SerpAPI to retrieve search result snippets.
        4. Save the results to a CSV file.
        5. Format results into a dictionary with keys as query and position, and values as the full text (or snippets) of the search result.

        While Koppel et al. (2014) randomly choose n imposters among the top m imposter, we use all of them.

        :param text: Input text to generate impostors for.

        :return: DataFrame containing search results with columns: 'query', 'title', 'url', 'snippet' (i.e. short content summary of search result), and 'position' (i.e. number of result in the search results)

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’. Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        if self.real_time_generation:
            logging.info("Generating impostors in real-time.")
            medium_frequency_words = self.get_medium_frequency_words(text)
            queries = self._generate_queries_based_on_candidate_words(
                medium_frequency_words
            )
            try:
                result_df = pd.DataFrame(self._parallel_fetch(queries))
                result_df.drop_duplicates(subset="url", inplace=True)

                if result_df.empty:
                    logging.info("Warning: No results fetched. CSV not saved.")
                    return result_df
                # TODO: use mongo collection instead
                if path2imp is None:
                    path2imp = (
                        Path(__file__).resolve().parents[2]
                        / CONFIG.PATH2GENERIC_ON_FLY_IMP
                    )
                else:
                    path2imp = Path(path2imp)
                if path2imp.suffix != ".csv" or path2imp.is_dir():
                    if path2imp.is_file():
                        path2imp = path2imp.with_suffix(".csv")
                    else:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        path2imp = (
                            path2imp / f"google_on_fly_impostor_results_{timestamp}.csv"
                        )

                path2imp.parent.mkdir(parents=True, exist_ok=True)
                result_df.to_csv(path2imp, index=False)
            except Exception as e:
                logging.warning(
                    f"Error during fetching results (probabily no more free API calls): {e}"
                )
                return {"imposter": "Error during fetching results, check logs."}
        else:  # use precomputed results
            logging.info("Using precomputed results from path2imp. No real-time generation.")
            if "PUCK" in text:  # Midsummer Night's Dream
                path2imp = (
                    path2imp
                    / "impostor_A_Midsummer_Nights_Dream_William_Shakespeare_results_20250608_201352.csv"
                )
            elif "Frankenstein" in text:  # Frankenstein
                path2imp = (
                    path2imp
                    / "impostor_Frankenstein_Mary_Wollstonecraft_(Godwin)_Shelley_results_20250608_145350.csv"
                )
            elif "unlineal" in text:  # Macbeth
                path2imp = (
                    path2imp
                    / "impostor_Macbeth_William_Shakespeare_results_20250608_211827.csv"
                )
            elif "auld" in text:  # Orthello
                path2imp = (
                    path2imp
                    / "impostor_Othello_the_Moor_of_Venice_William_Shakespeare_results_20250608_212040.csv"
                )
            else:  # A Lovers Complaint
                path2imp = (
                    path2imp
                    / "impostor_A_Lovers_Complaint_William_Shakespeare_results_20250608_202134.csv"
                )
            if not Path(path2imp).exists():
                return {
                    "imposter": f"No real time generation and Path {path2imp} does not exist."
                }
            result_df = pd.read_csv(path2imp)

        # aggregate results' texts, preferably using full_text, if empty use snippet and return a list of texts
        impostor_texts = {
            f"{re.sub(' ', '_', string=row['query'])}_{row['position']}": row[
                "full_text"
            ]
            for _, row in result_df.iterrows()
            if pd.notnull(row.get("full_text"))
        }

        # full_text is missing but snippet is present
        for _, row in result_df.iterrows():
            if pd.isna(row.get("full_text")) and pd.notnull(row.get("snippet")):
                key = f"{re.sub(' ', '_', row['query'])}_{row['position']}"
                impostor_texts[key] = row["snippet"]

        return impostor_texts
