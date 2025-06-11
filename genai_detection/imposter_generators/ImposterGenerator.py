from abc import ABC, abstractmethod
import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import List, Dict
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util
import random
import numpy as np
import pandas as pd
import spacy
import requests
from bs4 import BeautifulSoup
from datasets import load_from_disk
import serpapi
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import CONFIG

load_dotenv() 


class BaseImposterGenerator(ABC):
    """Abstract base class for generating imposters."""
    def __init__(self, n_impostors: int, split:str='test'):
        """
        :param n_impostors: number of impostors to generate
        :param split: dataset split to use (default: 'test')
        """
        self.n_impostors = n_impostors
        self.split = split

    @abstractmethod
    def generate_imposters(self, text:str, path2imp:str=None, real_time_generation:bool=False) -> dict:
        """Get a dictionary of impostor texts for the given input text.
        
        References:
        ===========
        Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
        Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.

        :param text: input text to generate impostors for (i.e., the candidate text, NOT the disputed text)
        """
        pass

    def _get_dataset_split_from_path(self, path2imp:str):
        path2imp = Path(path2imp)     
        if not path2imp.exists():
            raise FileNotFoundError(f"Training data not found at {path2imp}")
        dataset = load_from_disk(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), path2imp))  # TODO: for notebook ..
        if self.split not in dataset:
            raise ValueError(f"Dataset {path2imp} does not contain '{self.split}' split.")
        
        ds = dataset[self.split]
        if len(ds) == 0:
            raise ValueError("Dataset split is empty.")
        return ds


class ContentImposterGenerator(BaseImposterGenerator):
    def __init__(self, n_impostors: int, model_name: str = "all-MiniLM-L6-v2", split:str='test'):
        """
        :param n_impostors: number of impostors to generate
        :param model_name: embedding model from sentence-transformers
        :param split: dataset split to use (default: 'test')
        """
        super().__init__(n_impostors=n_impostors, split=split)
        self.model = SentenceTransformer(model_name)

    def generate_imposters(self, text: str, path2imp:str=None, real_time_generation:bool=False) -> List[str]:
        """
        Generates imposters from a pre-defined dataset.
        :param text: input text to generate imposters for (not used in this implementation)
        :param path2imp: path to the dataset file containing imposters, i.e. fixed Huggingface dataset
        :param real_time_generation: not used in this implementation, but kept for interface consistency
        :return: dictionary of imposters with keys as ids and values as texts
        """
        ds = self._get_dataset_split_from_path(path2imp)
        sampled = ds.shuffle().select(range(min(len(ds), self.n_impostors)))
        # select n_impostors entries which are most similar to the input text in terms of content
        candidate_texts = []
        for entry in sampled:
            pair = entry.get("pair", [])
            candidate_texts.extend(pair)

        if not candidate_texts:
            raise ValueError("No candidate texts found.")

        # compute embeddings
        # TODO: preprocess text to remove special characters, punctuation, etc.?
        text_embedding = self.model.encode(text, convert_to_tensor=True)
        candidate_embeddings = self.model.encode(candidate_texts, convert_to_tensor=True)

        # compute cosine similarity
        similarities = util.cos_sim(text_embedding, candidate_embeddings)[0]

        #  top-n most similar
        top_indices = similarities.topk(self.n_impostors).indices.tolist()
        selected_texts = [candidate_texts[i] for i in top_indices]

        return {f"imposter_{i}": imp for i, imp in enumerate(selected_texts)}


class GoogleSearchImposterGenerator(BaseImposterGenerator):
    def __init__(self, api_key: str, num_queries: int = 2, results_per_query: int = 25, max_workers: int = 2, n_min_words: int = 3, n_max_words: int = 5):
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
        """
        super().__init__(n_impostors=num_queries * results_per_query)
        self.api_key = api_key
        self.num_queries = num_queries
        self.results_per_query = results_per_query
        self.max_workers = max_workers
        self.n_min_words = n_min_words
        self.n_max_words = n_max_words
        assert self.n_min_words < self.n_max_words, "n_min_words must be less than n_max_words"
        self.nlp = spacy.load("en_core_web_sm")

    def get_medium_frequency_words(self, text: str, lower_pct: int = 30, upper_pct: int = 70) -> List[str]:
        """
        Extracts medium frequency words from the input text.
        Medium frequency words are defined as those that fall between the lower and upper percentiles of word frequencies in the text.
        :param text: input text to extract medium frequency words from
        :param lower_pct: lower percentile threshold for word frequency (default: 30)
        :param upper_pct: upper percentile threshold for word frequency (default: 70)
        :return: list of medium frequency words
        """
        doc = self.nlp(text.lower())
        words = [token.text for token in doc if token.is_alpha and not token.is_stop]   # filter out stop words and non-alphabetic tokens
        freq_counter = Counter(words)
        freqs = np.array(list(freq_counter.values()))
    
        # Compute dynamic thresholds
        low_thresh = np.percentile(freqs, lower_pct)
        high_thresh = np.percentile(freqs, upper_pct)
        return [word for word, count in freq_counter.items() if low_thresh <= count <= high_thresh]   # medium frequency words

    def _generate_queries_based_on_candidate_words(self, candidate_words: List[str]) -> List[str]:
        """
        Generates a list of queries using random combinations of candidate words. Each query is a string and will have a random number of words between n_min_words and n_max_words separated by whitespaces.
        :param candidate_words: list of candidate words to use for generating queries
        :return: list of generated queries
        """
        queries = []
        assert len(candidate_words) >= self.n_min_words, f"Not enough candidate words to generate queries, at {self.n_min_words} necessary."
        for _ in range(self.num_queries):
            n_words = min(len(candidate_words), random.randint(self.n_min_words, self.n_max_words))
            query_words = random.sample(candidate_words, n_words)
            queries.append(" ".join(query_words))
        return queries

    def _extract_text_from_url(self, url:str) -> str:
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
            for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form", "noscript"]):
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
        :param query: search query to fetch results for
        :return: list of dictionaries containing search results with keys: 'query', 'title', 'url', 'snippet', and 'position'
        """
        assert query, "No query to fetch results for."
        if not self.api_key:
            raise ValueError("API key for SerpAPI is not provided.")
        try:
            params = {
                "q": query, # the search query, eg. "coffee"
                "num": self.results_per_query,  # default: 10
                "api_key": self.api_key,
                "engine": "google_light",   # might return fewer results than 'google' engine, but is faster
                "safe": "off", # disbale filtering out adult content
                "nfpr": 0, # include results from auto-corrected query for misspellings
                "devices": "desktop",  # use desktop results
            }
            search = serpapi.search(params)
            results = search.as_dict()
            return [
                {"query": query, "title": res.get("title"), "url": res.get("link"), "snippet": res.get("snippet"), "rich_snippet": res.get("rich_snippet", ""),
                 "author": res.get("author", "unknown"), "position": res.get("position"), "full_text": self._extract_text_from_url(res.get("link")),}
                for res in results.get("organic_results", [])
            ]
        except Exception as e:
            print(f"Error fetching results for '{query}': {e}")
            return []

    def _parallel_fetch(self, queries: List[str]) -> List[Dict]:
        """
        Fetches results for multiple queries in parallel using a thread pool.
        :param queries: list of search queries to fetch results for
        :return: list of dictionaries containing search results for all queries
        """
        all_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.fetch_results, query): query for query in queries}
            for future in as_completed(futures):
                all_results.extend(future.result())
        return all_results

    def generate_imposters(self, text: str, path2imp:str=None, real_time_generation:bool=False) -> pd.DataFrame:
        """
        Generates imposters for the given input text using Google search results.

        Steps:
        1. Extract medium-frequency words from the input text.
        2. Formulate search queries using random combinations of those words.
        3. Use the SerpAPI to retrieve search result snippets.
        4. Save the results to a CSV file.
        5. Format results into a dictionary with keys as query and position, and values as the full text (or snippets) of the search result.

        :param text (str): input text to generate impostors for
        :param path2imp (str or Path, optional): Path to save the CSV. If a directory or None, appends a timestamped filename.
        :param real_time_generation (bool): If True, generates queries and fetches results in real-time. If False, uses precomputed results from the specified path.

        :return: DataFrame containing search results with columns: 'query', 'title', 'url', 'snippet' (i.e. short content summary of search result), and 'position' (i.e. number of result in the search results)

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’. Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        if real_time_generation:
            medium_frequency_words = self.get_medium_frequency_words(text)
            queries = self._generate_queries_based_on_candidate_words(medium_frequency_words)
            result_df = pd.DataFrame(self._parallel_fetch(queries))
            result_df.drop_duplicates(subset="url", inplace=True)


            if result_df.empty:
                print("Warning: No results fetched. CSV not saved.")
                return result_df
        
            if path2imp is None:
                path2imp = Path(CONFIG.PATH2GENERIC_ON_FLY_IMP) 
            else:
                path2imp = Path(path2imp)
            if path2imp.suffix != ".csv" or path2imp.is_dir():
                if path2imp.is_file():
                    path2imp = path2imp.with_suffix(".csv")
                else:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    path2imp = path2imp / f"google_on_fly_imposter_results_{timestamp}.csv"

            path2imp.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(path2imp, index=False)
        else:   # use precomputed results
            # TODO for debugging purposes, delete later:
            if 'PUCK' in text:  # Midsummer Night's Dream
                path2imp = path2imp / 'imposter_A_Midsummer_Nights_Dream_William_Shakespeare_results_20250608_201352.csv'
            elif 'Frankenstein' in text:  # Frankenstein
                path2imp = path2imp / 'imposter_Frankenstein_Mary_Wollstonecraft_(Godwin)_Shelley_results_20250608_145350.csv'
            elif 'unlineal' in text:  # Macbeth
                path2imp = path2imp / 'imposter_Macbeth_William_Shakespeare_results_20250608_211827.csv'
            elif 'auld' in text:  # Orthello
                path2imp = path2imp / 'imposter_Othello_the_Moor_of_Venice_William_Shakespeare_results_20250608_212040.csv'
            else: # A Lovers Complaint
                path2imp = path2imp / 'imposter_A_Lovers_Complaint_William_Shakespeare_results_20250608_202134.csv'
            result_df = pd.read_csv(path2imp)

        
        # aggregate results' texts, preferably using full_text, if empty use snippet and return a list of texts
        imposter_texts = {
            f"{re.sub(' ', '_', string=row['query'])}_{row['position']}": row['full_text']
            for _, row in result_df.iterrows()
            if pd.notnull(row.get('full_text'))
        }

        # full_text is missing but snippet is present
        for _, row in result_df.iterrows():
            if pd.isna(row.get('full_text')) and pd.notnull(row.get('snippet')):
                key = f"{re.sub(' ', '_', row['query'])}_{row['position']}"
                imposter_texts[key] = row['snippet']


        return imposter_texts
        
    
class TextLenImposterGenerator(BaseImposterGenerator):
    def __init__(self, n_impostors: int, split:str='test'):
        super().__init__(n_impostors=n_impostors, split=split)

    def generate_imposters(self, text: str, path2imp:str=None, real_time_generation:bool=False, valid_relative_text_len_dif:float=0.3) -> List[str]:
        ds = self._get_dataset_split_from_path(path2imp)
        max_subset_size = min(self.n_impostors, len(ds))  # generate around twice as many impostors as requested, to ensure diversity
        sampled = ds.shuffle(seed=42).select(range(max_subset_size))
        candidate_texts = []
        for entry in sampled:
            pair = entry.get("pair", [])
            candidate_texts.extend(pair)
        text_len = len(text)
        threshold = text_len * valid_relative_text_len_dif
        # filter candidates based on length
        filtered_candidates = [s for s in candidate_texts if abs(len(s) - text_len) < threshold]
        if not filtered_candidates:
            return self.generate_imposters(text=text, path2imp=path2imp, real_time_generation=path2imp, valid_relative_text_len_dif=min(1, valid_relative_text_len_dif * 2))  # try again with a larger threshold
        
        lengths = np.array([len(s) for s in filtered_candidates])
        diffs = np.abs(lengths - text_len)
        probs = 1 / (1 + diffs)
        probs /= probs.sum()

        num_to_sample = min(self.n_impostors, len(filtered_candidates))
        selected = np.random.choice(filtered_candidates, size=num_to_sample, replace=False, p=probs)

        return {f"imposter_{i}": imp for i, imp in enumerate(selected)}


class NDocsImposterGenerator(BaseImposterGenerator):
    def __init__(self, n_impostors: int):
        """
        References:
        ===========
        Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.
        """
        super().__init__(n_impostors=n_impostors)

    def generate_imposters(self, text: str, path2imp:str=None, real_time_generation:bool=False) -> List[str]:
        pass


class LLMImposterGenerator(BaseImposterGenerator):
    def __init__(self, n_impostors: int):
        self.n_impostors = n_impostors

    def generate_imposters(self, text: str, path2imp:str=None, real_time_generation:bool=False) -> List[str]:
        pass


class FixedImposterGenerator(BaseImposterGenerator):
    def __init__(self, n_impostors: int, split:str='test'):
        """
        :param n_impostors: number of imposters to generate
        :param split: dataset split to use (default: 'test')

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’. Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.
        """
        super().__init__(n_impostors=n_impostors, split=split)
 

    def generate_imposters(self, text: str, path2imp:Path=None, real_time_generation:bool=False) -> List[str]:
        """
        Generates imposters from a pre-defined dataset.
        :param text: input text to generate imposters for (not used in this implementation)
        :param path2imp: path to the dataset file containing imposters, i.e. fixed Huggingface dataset
        :param real_time_generation: not used in this implementation, but kept for interface consistency
        :return: dictionary of imposters with keys as ids and values as texts
        """
        ds = self._get_dataset_split_from_path(path2imp)
        sampled = ds.shuffle().select(range(min(len(ds), self.n_impostors//2)))
        imposters = {}
        for i, entry in enumerate(sampled):
            if "pair" not in entry:
                continue
            key = entry.get("id", f"imposter_{i}")
            imposters[f"{key}_left"] = entry["pair"][0]
            imposters[f"{key}_right"] = entry["pair"][1]

        if not imposters:
            raise ValueError("No imposters found with 'pair' field.")

        return imposters


class BlogImposterGenerator(FixedImposterGenerator):
    def __init__(self, n_impostors: int, split:str='test'):
        super().__init__(n_impostors=n_impostors, split=split) 
        
    def generate_imposters(self, text: str, path2imp:str=None, real_time_generation:bool=False) -> List[str]:
        """Generates imposters from the Blog dataset.
        :param text: input text to generate imposters for (not used in this implementation)
        :param path2imp: not used in this implementation, but kept for interface consistency
        :param real_time_generation: not used in this implementation, but kept for interface consistency
        :return: dictionary of imposters with keys as ids and values as texts"""
        return super().generate_imposters(text=text, path2imp=os.path.join(os.path.abspath(".."), CONFIG.PATH2BLOG), real_time_generation=real_time_generation)

        
       
# Example usage
if __name__ == "__main__":
    # artwork_name = "Frankenstein_Mary_Wollstonecraft_(Godwin)_Shelley.txt"
    # #"A_Midsummer_Nights_Dream_William_Shakespeare.txt"#"A_Lovers_Complaint_William_Shakespeare.txt"
    # path2lovers_shakespeare = Path(CONFIG.PATH2GUTENBERG) / artwork_name
    # with open(path2lovers_shakespeare) as f:
    #     input_text = f.read()

    generator = GoogleSearchImposterGenerator(api_key='CONFIG.SERPAPI_KEY', num_queries=2, results_per_query=25, max_workers=2, n_min_words=3, n_max_words=5)
    # imposters = generator.generate_imposters(input_text, path2imp=Path(CONFIG.PATH2GENERIC_ON_FLY_IMP) / f"imposter_{artwork_name.split('.')[0]}_results.csv")
    # print(f"Generated {len(imposters)} imposters for {artwork_name}:")

    # for imposter_name, imposter_text in imposters.items():
    #     print(f"Imposter {imposter_name}: {imposter_text[:100]}...")  # Print first 100 characters of each imposter

