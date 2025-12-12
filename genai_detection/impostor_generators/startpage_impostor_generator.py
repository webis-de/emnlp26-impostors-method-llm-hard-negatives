# Copyright 2025 Janek Bevendorff and Klara M. Gutekunst, Webis
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
"""Find original code base: https://git.webis.de/code-research/web-search/affiliate-marketing-and-search/-/blob/main/serp_crawler/serp_crawler/crawler.py?ref_type=heads (28.11.2025)"""
import logging
import random
from random import choice
from typing import List, Dict
from urllib import parse as urlparse

import httpx
from resiliparse.parse.html import HTMLTree

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.search_generator_base import SearchImpostorGeneratorBase

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )

# List of user agent strings (has to be rotated regularly to circumvent blacklisting).
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/125.0.6422.33 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/125.0.6422.33 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.53 Mobile Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:126.0) Gecko/20100101 Firefox/126.0',
    'Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0',
    'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/126.0 Mobile/15E148 Safari/605.1.15',
    'Mozilla/5.0 (Android 14; Mobile; rv:126.0) Gecko/126.0 Firefox/126.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:115.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/124.0.2478.109',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/124.0.2478.109',
    'Mozilla/5.0 (Linux; Android 10; HD1913) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.53 Mobile Safari/537.36 EdgA/124.0.2478.87',
    'Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.53 Mobile Safari/537.36 EdgA/124.0.2478.87',
    'Mozilla/5.0 (Linux; Android 10; Pixel 3 XL) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.53 Mobile Safari/537.36 EdgA/124.0.2478.87',
    'Mozilla/5.0 (Linux; Android 10; ONEPLUS A6003) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.53 Mobile Safari/537.36 EdgA/124.0.2478.87',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 EdgiOS/124.2478.105 Mobile/15E148 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 OPR/110.0.0.0',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 OPR/110.0.0.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 OPR/110.0.0.',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 OPR/110.0.0.0',
    'Mozilla/5.0 (X11; CrOS x86_64 15633.69.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.212 Safari/537.36',
    'Mozilla/5.0 (X11; CrOS armv7l 15633.69.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.212 Safari/537.36',
    'Mozilla/5.0 (X11; CrOS aarch64 15633.69.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.212 Safari/537.36',
    'Mozilla/5.0 (X11; CrOS x86_64 15633.69.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.6045.212 Safari/537.36',
    'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.53 Mobile Safari/537.36',
    'Mozilla/5.0 (Linux; Android 14; SM-A205U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.53 Mobile Safari/537.36',
    'Mozilla/5.0 (Linux; Android 14; SM-A102U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.53 Mobile Safari/537.36',
    'Mozilla/5.0 (Linux; Android 14; SM-G960U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.53 Mobile Safari/537.36',
    'Mozilla/5.0 (Android 14; Mobile; rv:68.0) Gecko/68.0 Firefox/126.0',
    'Mozilla/5.0 (Android 14; Mobile; LG-M255; rv:126.0) Gecko/126.0 Firefox/126.0'
]


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
        Issue HTTP request for a search result page from StartPage using a random user agent string.

        :param query: search query
        :param user_agent_string: set fixed UA string instead of choosing one at random
        :return: structured list of search results
        """
        assert query, "No query to fetch results for."

        request_headers = {
            "User-Agent": choice(USER_AGENTS),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.8",
            "Accept-Encoding": "gzip,deflate",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": "https://www.startpage.com/",
            "Origin": "https://www.startpage.com",
            "Cookie": "preferences=date_timeEEEworldN1Ndisable_family_filterEEE0N1Ndisable_open_in_new_window"
            "EEE0N1Nenable_post_methodEEE1N1Nenable_proxy_safety_suggestEEE1N1Nenable_stay_control"
            "EEE1N1Ninstant_answersEEE1N1Nlang_homepageEEEs%2Fdevice%2FenN1NlanguageEEEenglish"
            "N1Nlanguage_uiEEEenglishN1Nnum_of_resultsEEE20N1Nsearch_results_regionEEEall"
            "N1NsuggestionsEEE1N1Nwt_unitEEEcelsius",
        }

        try:
            resp = httpx.post(
                "https://www.startpage.com/sp/search",
                content=urlparse.urlencode(
                    dict(query=query, t="device", lui="english", cat="web")
                ).encode(),
                headers=request_headers,
            )
            response_bytes = resp.read()

            if not response_bytes:
                logger.error("Invalid server response")
                return []

            tree = HTMLTree.parse_from_bytes(response_bytes, "utf-8")
            logging.info(f"Fetched results from {query}")
            result_list = []
            for qr in tree.body.query_selector_all("#main > .w-gl .result"):
                result_a = qr.query_selector(".result-title.result-link")
                if not result_a:
                    logger.error("Result has no title link.")
                    continue
                result_url = result_a["href"]
                snippet = qr.query_selector(".description")
                result_list.append(
                    dict(
                        query=query,
                        target_uri=result_url,
                        snippet=snippet.text.strip() if snippet else "",
                    )
                )
            while not result_list:
                logger.error(f"No results found. Random changing word order for query '{query}'")
                random.shuffle(query.split())
                result_list.append(self.fetch_results(query=" ".join(query)))
            logging.info(f"Fetched {len(result_list)} results from {query}")
            return result_list
        except Exception as e:
            logger.error("Connection error while fetching results.")
            logger.exception(e)
            return []


if __name__ == "__main__":
    startpage_google_retriever = StartPageSearchImpostorGenerator(api_key=CONFIG.CHATNOIR_KEY)
    logging.info("%s", startpage_google_retriever.fetch_results(query="Hello World"))
