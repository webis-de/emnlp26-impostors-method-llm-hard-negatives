from typing import List

import scipy
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfFeatureExtractor:
    def __init__(self):
        # char_wb: n-grams only from text inside word boundaries; n-grams at the edges of words are padded with space.
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html (14.11.2025)
        # min_df: Kocher et al. (2015) exclude words appearing only once
        self.vectorizer = TfidfVectorizer(ngram_range=(4, 4), analyzer="char_wb", min_df=2)

    def fit_transform(self, corpus: List[str]) -> scipy.sparse.csr_matrix:
        """
        Returns a numpy array with shape (len(corpus), len(feature_extractor.vectorizer.vocabulary_))
        :param corpus: List of strings; each string is a document.
        :return: The sparse matrix with shape (len(corpus), len(feature_extractor.vectorizer.vocabulary_)) with tfidf features.
        """
        assert len(corpus) > 0, "The corpus cannot be empty."
        assert all([type(doc)==str for doc in corpus]), "The corpus cannot contain non-string."
        return self.vectorizer.fit_transform(corpus)
