from collections import Counter, defaultdict
from itertools import batched
import itertools
import json
from pathlib import Path
from random import sample
import re
from typing import Iterable, List, Literal
import heapq
from nltk.stem.snowball import SnowballStemmer

import numpy as np
from nltk import ngrams
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from genai_detection.detectors.detector_base import DetectorBase

__all__ = ["ImpostorDetector"]


class ImpostorDetector(DetectorBase):
    """
    The Imposter method extends the ngram-unmasking method.
    It uses saves the most similar author to the disputed text for each of multiple random feature selection rounds, 
    where the disputed text is compared not only to the candidate text, but alos to a set of imposter texts.
    The final prediction is made based on of how often an author is predicted after each feature-elimination step.

    The input is a list of texts where text ``i`` and text ``i+1`` belong to a pair.
    The output for one document pair is a score for the disputed text and the candidate text (i.e. author).

    References:
    ===========
    Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
    Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.

    Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.
    """

    def __init__(
        self,
        rounds=100,
        top_n=100000,
        portion_delete=0.5,
        tokenizer=None,
        shared_vocab_only=True,
        tfidf_freqs=True,
        n_impostors=25,
        threshold=0.1,
    ):
        """
        :param rounds: number of random feature selection rounds, Koppel et Al. (2014) use 100
        :param top_n: number of top space-free character 4-grams to consider, Koppel et Al. (2014) use 100,000
        :param portion_delete: portion of features to eliminate in each round (reset in each round); Koppel et Al. (2014) use 50% of features
        :param tokenizer: custom tokenizer function (must accept exactly one parameter, defaults to space-free character 4-grams cf. Koppel et Al. (2014))
        :param shared_vocab_only: restrict analysis to shared vocabulary across pairs of texts (Koppel et Al. (2014): all texts in the corpus, i.e. shared)
        :param tfidf_freqs: use tfidf term frequencies (Koppel et Al. (2014) use tfidf)
        :param n_impostors: number of impostors to use for each candidate TODO: allow specification type of LLM impostors; Koppel et Al. (2014) use 25 impostors
        :param threshold: threshold for the minimum similarity score to consider two texts same-author, TODO: not used yet, Koppel et Al. (2014) use 0.1
        """

        self.rounds = rounds
        self.top_n = top_n
        self.shared_vocab_only = shared_vocab_only
        self.portion_delete = portion_delete
        self.n_impostors = n_impostors
        self.tfidf_freqs = tfidf_freqs
        self.tokenizer = tokenizer or self.tokenize_char_ngrams
        self.threshold = threshold

    def get_scores(self, text: Iterable[str]) -> List[float]:
        """
        Get scores for text pairs. A higher score indicates that the input text pair is more likely to be authored by the same author.
        The algorithm stems from Koppel et Al. (2014)[, where some details are adapted from Kocher et Al. (2015)]:
        Each text from the pair is the disputed text and the candidate text once.
        For the candidate text, a set of impostors is generated.
        For each round, a portion of features is randomly deleted, and the most similar candidate text is determined.
        The final score is the number of rounds where the candidate text was the most similar to the disputed text.
        The final score for a pair is the average of the scores for both directions (disputed text vs. candidate text and vice versa).

        While Koppel et Al. (2014) use (1) a fixed set of imposter documents without realtion to document pair, 
        (2) on-the-fly generated same content imposter via Google search, 
        (3) Blogs to obtain same genre imposters, and Kocher et Al. (2015) use (4) a set of imposter documents based on the number of documents written by the author,
        we define different techniques to generate impostors, which can be specified via the `technique` parameter in the `_get_imposters` method:
        We currently support:
        (1) `text_len`: generate impostors of similar length from a predefined dataset (default, see `_get_imposters` method).
        (2) `llm`: use LLMs to generate impostors (TODO: not implemented yet), extension of Koppel et Al. (2014).
        (3) `n_docs`: generate impostors based on the number of documents written by the author (TODO: not implemented yet), cf. Kocher et Al. (2015).
        
        TODO: If the score is above a certain threshold, the input text is classified as same-author, which is not implemented yet/ not the purpose of this method.

        Koppel et Al. (2014) exclude texts shorter than 500 words.
        Kocher et Al. (2015) exclude words appearing only once to prevent overfitting to words occuring only once.
        Koppel et Al. (2014) select m most similar imposters in terms of min-max similarity as imposter candidates and then, 
        randomly select n actual imposters among potential imposters (because it has proven superior to using the top n imposters). 
        They claim the approach is not sensitive to the choice of m and n.
        Koppel et Al. (2014) compare using min-max and cosine simialrity.

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
        Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.

        Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.

        :param text: input text or batch of input texts
        :return: score indicating whether the input text is machine-generated, i.e. close 1 means machine-generated, close 0 means human-written
        """
        text = list(text)  # convert to tuple to list

        scores_per_pair = defaultdict(
            int
        )  # id is index of pair (i.e, length is half of the input text list)
        for i, t in enumerate(batched(text, 2, strict=True)):
              # TODO: check text length, if too short, i.e. less than 500 `words`(unclear which unit from Koppel et. Al. (2014)), skip?
            if len(self.tokenize_whitespace(t[0])) + len(self.tokenize_whitespace(t[1])) < 1000:
                continue
          
            # TODO: preprocessing: remove punctuation, lowercasing, remove html tags (e.g., <nl>), etc.?
            # Koppel et Al. (2014) do not normalize text pairs, but without normalization, the results are terrible. 
            # Does not make sense, bc 	idiosyncrasies of authors are not captured when using stemmed text.
            # Kontrolliere Situation
            tokens_left = self.tokenizer(self.normalize_text(t[0]))
            tokens_right = self.tokenizer(self.normalize_text(t[1]))

            if len(tokens_left) == 0 or len(tokens_right) == 0:
                print("Skipping empty text pair: Left: {}, Right: {}".format(t[0], t[1]))
                continue

            # frequencies as Counter (subclass of defaultdict(int))
            freqs_left = Counter(tokens_left)
            freqs_right = Counter(tokens_right)

            # Kocher et Al. (2015) exclude words appearing only once
            freqs_left = Counter({k: v for k, v in freqs_left.items() if v > 1})
            freqs_right = Counter({k: v for k, v in freqs_right.items() if v > 1})

            if self.shared_vocab_only:  # TODO: over all corpus documents
                shared_tokens = freqs_left.keys() & freqs_right.keys()
            else:
                shared_tokens = freqs_left.keys() | freqs_right.keys()
           
            top_tokens = heapq.nlargest(
                self.top_n,
                shared_tokens,
                key=lambda x: freqs_left[x] + freqs_right[x]
            )

            # TODO: muss man TFIDF gemeinsam (left, right, imposters) berechnen, wegen Dataset Normalierung?
            x_left = self.tokens_to_matrix(tokens_left, top_tokens)
            x_right = self.tokens_to_matrix(tokens_right, top_tokens)

            store = {
                # TODO: if I knew >=1 author, i could choose imposters based on similar number of documents written (like paper)
                "left": {"tfidf": x_left, "tokens": tokens_left, "text": t[0], "author": "unknown"},
                "right": {"tfidf": x_right, "tokens": tokens_right, "text": t[1], "author": "unknown"},
            }

            # two iterations, generating imposters for each candidate once
            for j, (unknown, candidate) in enumerate(
                itertools.permutations(list(store.keys()), 2)
            ):
                scores_over_different_rounds = 0
                # get imposters for the candidate text, NOT the disputed text
                impostor_candidates = self._get_imposters(
                    store[candidate]["text"], self.n_impostors, technique="text_len"
                )  # returns dict of model:text pairs
                tmp_store = {
                    impostor_name: {
                        "tfidf": self.tokens_to_matrix(
                            self.tokenizer(impostor_text), top_tokens
                        ),
                        "text": impostor_text,
                        "tokens": self.tokenizer(impostor_text),
                    }
                    for impostor_name, impostor_text in impostor_candidates.items()
                }
                tmp_store[candidate] = store[candidate]  # add actual candidate

                # for different rounds, randomly delete a portion of features (reset in each round)
                for _ in range(self.rounds):
                    # feature selection: randomly delete a portion of features
                    rand_feat_to_keep_ids = sample(
                        range(len(top_tokens)),
                        int(len(top_tokens) * (1 - self.portion_delete)),
                    )
                    scores = {
                        c: self.minmax_similarity(
                            store[unknown]["tfidf"][:, rand_feat_to_keep_ids],  # disputed text
                            tmp_store[c]["tfidf"][:, rand_feat_to_keep_ids],
                        )
                        for c in list(tmp_store.keys())
                    }
                    # increase score if the most similar candidate is the actual candidate
                    max_similar_candidate = max(scores, key=scores.get)
                    scores_over_different_rounds += max_similar_candidate == candidate
                # average after second loop
                scores_per_pair[i] += scores_over_different_rounds
                scores_per_pair[i] /= j + 1

        # one elmenent = averaged score of X,Y and Y,X pair (score=number of rounds where the candidate was the most similar)
        # TODO: threshold is in [0,1], maybe normalize by rounds?
        return list(scores_per_pair.values())
        # return [v / self.rounds for v in scores_per_pair.values()]
    
    def normalize_text(self, text):
        """
        Normalize input text by lowercasing and stemming.
        Koppel et Al. (2014) do (explicitly) not normalize text pairs, but without normalization, the results are terrible.
        Kocher et Al. (2015) use isolated words without stemming but with punctuation symbols.
        """
        stemmer = SnowballStemmer('english')
        return ' '.join(stemmer.stem(w) for w in text.lower().split())

    def get_prediction(self, text: Iterable[str]) -> List[bool]:
        """
        Predict if the input text(s) were written by a the same author TODO: machine.

        :param text: input text or batch of input texts
        :return: boolean classifications of whether inputs are likely same author TODO: machine-generated
        """
        scores = self.get_scores(text)
        return [score > self.threshold for score in scores]

    def tokens_to_matrix(self, tokens, top_token_list):
        """
        Transform list of tokens into matrix of term tfidf values of the top tokens.
        Koppel et Al. (2014) use space-free character 4-grams tfidf values to represent each document as a numerical vector.

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.

        :param tokens: list of input tokens (e.g., space-free character 4-grams)
        :param top_token_list: list of top tokens to include in the matrix
        :return: Numpy array of term tfidf values, `shape = (len(tokens), len(top_token_list))`
        """
        if len(top_token_list) == 0:
            return np.zeros((1, self.top_n), dtype=np.float32)  # return empty matrix if no top tokens
        
        # avoid fitting a new vectorizer every time (costly)
        if not hasattr(self, "_vectorizer") or self._vectorizer_vocab != top_token_list:
            self._vectorizer = TfidfVectorizer(vocabulary=top_token_list, input="content", dtype=np.float32)
            self._vectorizer_vocab = top_token_list

        tfidf_matrix = self._vectorizer.fit_transform(
            [" ".join(tokens)]
        )  # format (n_samples=1, n_features=self.top_n)

        assert tfidf_matrix.shape[1] == len(
            top_token_list
        ), "TFIDF matrix shape mismatch with top token list length."
        assert (
            tfidf_matrix.shape[0] == 1
        ), "TFIDF matrix should have one row for the single input document."

        return tfidf_matrix.toarray()

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Calculate cosine similarity between two vectors.
        Koppel et Al. (2014) have use cosine similarity as a baseline.
        """
        return (
            cosine_similarity(vec1, vec2).flatten()[0]
            if vec1 is not None and vec2 is not None
            else 0.0
        )

    def minmax_similarity(self, vec1, vec2):
        """
        Calculate min-max similarity between two vectors in TFIDF format.
        Koppel et Al. (2014) use min-max similarity.
        """
        if vec1 is None or vec2 is None:
            return 0.0
        assert len(vec1) == len(vec2), "Vectors must be of the same length."
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        numerator = np.minimum(vec1, vec2).sum()
        denominator = np.maximum(vec1, vec2).sum()
        return 0.0 if denominator == 0 else numerator / denominator
    
    @staticmethod
    def tokenize_whitespace(text: str, normalize_ws: bool = True):
        """
        Tokenize input text by any whitespace character (including \n \r \t \f and spaces).
        Kocher et Al. (2015) use isolated words without stemming but with punctuation symbols.

        References:
        ===========
        Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.

        :param text: input text
        :param normalize_ws: collapse whitespace before tokenization
        :return: list of tokens
        """
        if normalize_ws:
            text = re.sub(r"\s+", " ", text)
        return text.split() 

    @staticmethod
    def tokenize_char_ngrams(
        text: str, n: int = 4, normalize_ws: bool = True, space_free: bool = True
    ):
        """
        Tokenize input text into character n-grams.
        Koppel et Al. (2014) use space-free character 4-grams tfidf values to represent each document as a numerical vector.
        A space-free n-grams is a (1) sequence of n characters without any whitespace in it, (2) a sequence of <= n characters surrounded by spaces.

        References:
        ===========
        Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
        Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.

        :param text: input text
        :param n: n-gram order
        :param normalize_ws: collapse whitespace before tokenization
        :return: list of n-gram tokens
        """
        # remove first and last whitespace
        text = text.strip()
        if normalize_ws:
            text = re.sub(r"\s+", " ", text)
        if space_free:
            # of size n without spaces
            n_grams = [
                text[i : i + n]
                for i in range(0, len(text) - n + 1)
                if " " not in text[i : i + n]
            ]
            # add m-grams with spaces, where m < n
            # TODO: should be >2 whitespaces to pad be allowed? I don't think so, produces: 'to  ', '  to'
            for token in text.split():
                if (len(token) < n) and ((n - 2) <= len(token)):
                    # add all n-grams options with spaces
                    # TODO: results in most common ngrams ('the ', 22), (' the', 22), 'to  ': 10, ' to ': 10, '  to': 10,
                    n_grams.extend(
                        " " * i + token + " " * (n - len(token) - i)
                        for i in range(n - len(token) + 1)
                    )
            return n_grams

        else:
            return [text[i : i + n] for i in range(0, len(text) - n + 1)]

    def _get_imposters(self, text: str, n: int, technique: Literal["llm", "text_len", "n_docs"]="text_len") -> dict:
        """
        Get a dictionary of impostor texts for the given input text.

        This method generates impostors based on the input text. If `technique` is llm, it should use LLMs to generate impostors (not implemented yet). 
        If `technique` is text_len, it generates impostors of similar length from a predefined dataset.
        If `technique` is n_docs, it generates impostors based on the number of documents written by the author; as proposed by Kocher et Al. (2015).
        
        References:
        ===========
        Kocher, Mirco, and Jacques Savoy. ‘UniNE at CLEF 2015: Author Identification’, 2015.

        :param text: input text to generate impostors for (i.e., the candidate text, NOT the disputed text)
        :param n: number of impostors to generate
        :param use_llms: whether to use LLMs to generate impostors. If not, imposters are texts of similar length. TODO: reference to paper
        :return: dictionary of model names and their corresponding impostor texts
        """
        # TODO: Placeholder for actual implementation
        # In practice, this should return a dict with model names as keys and generated texts as values.
        if technique == "llm":
            raise NotImplementedError("LLM-based impostor generation is not implemented.")
        if technique == "n_docs":
            # TODO: I need author names for this
            raise NotImplementedError("n-docs impostor generation is not implemented.")
        else:            
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
            selected = np.random.choice(a=candidates, p=probs, size=min(n, len(candidates)), replace=False)
            # create a dictionary of impostors
            return {f"impostor_{i}": selected[i] for i in range(len(selected))}
