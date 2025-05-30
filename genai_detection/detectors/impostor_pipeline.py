from collections import Counter

import numpy as np
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfTransformer


class ImpostorPipeline:
    """
    Impostors method by Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
    Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.
    """
    def __init__(self):
        self.top_four_grams = None
        self._fitted = False

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """ Calculate cosine similarity between two vectors. """
        if not vec1 or not vec2:
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = sum(a ** 2 for a in vec1) ** 0.5
        norm_b = sum(b ** 2 for b in vec2) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def minmax_similarity(self, vec1, vec2):
        """ Calculate min-max similarity between two vectors. """
        if not vec1 or not vec2:
            return 0.0
        # tfidf vectors
        assert self.top_four_grams is not None, "Model must be fitted before calculating similarity."
        count_matrix = np.array([
            [vec1.get(term, 0) for term in self.top_four_grams],
            [vec2.get(term, 0) for term in self.top_four_grams]
        ])
        # Apply TF-IDF
        transformer = TfidfTransformer()
        tfidf_matrix = transformer.fit_transform(count_matrix)
        tfidf_array = tfidf_matrix.toarray()
        return sum(min(a, b) for a, b in zip(tfidf_array[0], tfidf_array[1])) / sum(max(a, b) for a, b in zip(tfidf_array[0], tfidf_array[1]))



    def embed(self, text:str):
        """ Embed the text into spacefree 4-grams like original paper."""
        # TODO: ensure spacefree 4-grams
        if not self._fitted:
            raise RuntimeError("The model must be fitted before embedding text.")

        four_grams = ngrams(text.split(), 4)
        filtered_grams = [gram for gram in four_grams if gram in self.top_four_grams]
        # keep 100,000 most frequent 4-grams in corpus, i.e., saved in self.top_four_grams
        count_four_grams = Counter(filtered_grams)
        return count_four_grams.values()


    def fit(self, corpus):
        """
        Fit the model on a corpus of documents.

        Parameters:
        corpus (list of str): A list of document texts to fit the model.
        """
        if not isinstance(corpus, list) or not all(isinstance(doc, str) for doc in corpus):
            raise ValueError("Corpus must be a list of strings (documents).")
        # compare space-free character 4-gram frequencies in the corpus
        four_gram_counter = Counter()
        for text in corpus:
            # Generate space-free character 4-grams
            for token in text.split():
                # FIXME: not correct, should include words < 4 characters
                four_grams = [''.join(gram) for gram in ngrams(token, 4,pad_both_ends=True)]
                four_gram_counter.update(four_grams)
        self.top_four_grams = dict(four_gram_counter.most_common(100000))

    @staticmethod
    def predict(doc1, doc2):
        """
        Predict if two documents are written by the same author.

        Parameters:
        doc1 (str): The first document text.
        doc2 (str): The second document text.

        Returns:
        bool: True if both documents are likely written by the same author, False otherwise.
        """
        # True if doc1 is more similar to doc2 than to imposters
        # tasks: select imposters, calculate similarity

        # assert >= 500 words in each document
        if len(doc1.split()) < 500 or len(doc2.split()) < 500:
            raise ValueError("Both documents must contain at least 500 words.")
