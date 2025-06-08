from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import List, Dict
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import numpy as np
import pandas as pd
import spacy
from serpapi import GoogleSearch


class BaseImposterGenerator(ABC):
    """Abstract base class for generating imposters."""

    @abstractmethod
    def generate_imposters(self, tetx:str) -> dict:
        """Get a dictionary of impostor texts for the given input text.
        
        References:
        ===========
        Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
        Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.

        :param text: input text to generate impostors for (i.e., the candidate text, NOT the disputed text)
        """
        pass


class GoogleSearchImposterGenerator(BaseImposterGenerator):
    def __init__(self, api_key: str, num_queries: int = 50, results_per_query: int = 25, max_workers: int = 10, n_min_words: int = 3, n_max_words: int = 5):
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
            query_words = random.sample(candidate_words, random.randint(self.n_min_words, self.n_max_words))
            queries.append(" ".join(query_words))
        return queries

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
                "num": self.results_per_query,
                "api_key": self.api_key,
                "engine": "google_light",
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            return [
                {"query": query, "title": res.get("title"), "url": res.get("link"), "snippet": res.get("snippet"), "position": res.get("position")}
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

    def generate_imposters(self, text: str) -> pd.DataFrame:
        """
        Generates impostors for the given input text by fetching search results based on medium frequency words extracted from the text.
        :param text: input text to generate impostors for
        :return: DataFrame containing search results with columns: 'query', 'title', 'url', 'snippet' (i.e. short content summary of search result), and 'position' (i.e. number of result in the search results)
        """
        medium_frequency_words = self.get_medium_frequency_words(text)
        queries = self._generate_queries_based_on_candidate_words(medium_frequency_words)
        results = self._parallel_fetch(queries)
        return pd.DataFrame(results)
    
    
class TextLenImposterGenerator(BaseImposterGenerator):
    def __init__(self, n_imposter: int):
        self.n_imposter = n_imposter

    def generate_imposters(self, text: str) -> List[str]:
        # TODO: Add path to training data
        # pan23-dataset-converted/train/
        path2_training_data = Path("../data/datasets/pan23-authorship-verification/pan23-authorship-verification-training-dataset/pairs.jsonl")  # Placeholder path
        #path2_training_data = Path("../data/datasets/pan20-authorship-verification/pan20-authorship-verification-training-dataset/pan20-authorship-verification-training-small.jsonl")
        
        if not path2_training_data.exists():
            raise FileNotFoundError(f"Training data not found at {path2_training_data}")
        
        # TODO: Ensure not same author as imposter (difficult, bc during inference, we don't know the author of the input text)
        # FIXME: for PAN20 or other big datasets, this will produce OOM errors
        with open(path2_training_data, "r", encoding="utf-8") as f:
            # TODO: Omit enumeration an limit of 500 pairs later
            tr_data = [json.loads(line).get('pair',[]) for i,line in enumerate(f) if i < 500] 
            candidates = [item for sublist in tr_data for item in sublist if abs(len(item)- len(text)) < len(text) * 0.3]  # flatten and filter by length
            probs = [1 / (1 + abs(len(s) - len(text))) for s in candidates]
            total = sum(probs)
            probs = [prob / total for prob in probs]

        if not candidates:
            raise ValueError("No suitable impostor candidates found.")
        
         # select n random texts of similar length
        selected = np.random.choice(a=candidates, p=probs, size=min(self.n_imposter, len(candidates)), replace=False)
        # create a dictionary of impostors
        return {f"impostor_{i}": selected[i] for i in range(len(selected))}
    

    
class NDocsImposterGenerator(BaseImposterGenerator):
    def __init__(self, n_imposter: int):
        """
        References:
        ===========
        Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.
        """
        self.n_imposter = n_imposter

    def generate_imposters(self, text: str) -> List[str]:
        pass


class LLMImposterGenerator(BaseImposterGenerator):
    def __init__(self, n_imposter: int):
        self.n_imposter = n_imposter

    def generate_imposters(self, text: str) -> List[str]:
        pass


class FixedImposterGenerator(BaseImposterGenerator):
    def __init__(self, n_imposter: int, imposter_file:Path):
        self.n_imposter = n_imposter
        self.imposter_file = imposter_file
        if not self.imposter_file.exists():
            raise FileNotFoundError(f"Imposter file not found at {self.imposter_file}")

    def generate_imposters(self, text: str) -> List[str]:
        pass


class BlogImposterGenerator(BaseImposterGenerator):
    def __init__(self, n_imposter: int):
        self.n_imposter = n_imposter

    def generate_imposters(self, text: str) -> List[str]:
        pass


        

        
       
# Example usage
if __name__ == "__main__":
    with open("input_text.txt") as f:
        input_text = f.read()

    generator = GoogleSearchImposterGenerator(api_key="YOUR_SERPAPI_KEY")
    result_df = generator.generate_imposters(input_text)
    result_df.to_csv("imposter_results.csv", index=False)
    print(result_df.head())
