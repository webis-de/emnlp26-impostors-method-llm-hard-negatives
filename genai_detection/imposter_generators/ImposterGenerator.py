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
    def __init__(self, api_key: str, num_queries: int = 50, results_per_query: int = 25, max_workers: int = 10):
        self.api_key = api_key
        self.num_queries = num_queries
        self.results_per_query = results_per_query
        self.max_workers = max_workers
        self.nlp = spacy.load("en_core_web_sm")

    def extract_features(self, text: str, low: int = 2, high: int = 10) -> List[str]:
        doc = self.nlp(text.lower())
        words = [token.text for token in doc if token.is_alpha and not token.is_stop]
        freq = Counter(words)
        return [word for word, count in freq.items() if low <= count <= high]

    def _generate_imposters(self, features: List[str]) -> List[str]:
        queries = []
        for _ in range(self.num_queries):
            query_words = random.sample(features, random.randint(3, 5))
            queries.append(" ".join(query_words))
        return queries

    def fetch_results(self, query: str) -> List[Dict]:
        try:
            params = {
                "q": query,
                "num": self.results_per_query,
                "api_key": self.api_key,
                "engine": "google",
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            return [
                {"query": query, "title": res.get("title"), "url": res.get("link")}
                for res in results.get("organic_results", [])
            ]
        except Exception as e:
            print(f"Error fetching results for '{query}': {e}")
            return []

    def _parallel_fetch(self, queries: List[str]) -> List[Dict]:
        all_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.fetch_results, query): query for query in queries}
            for future in as_completed(futures):
                all_results.extend(future.result())
        return all_results

    def generate_imposters(self, text: str) -> pd.DataFrame:
        features = self.extract_features(text)
        queries = self._generate_imposters(features)
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
