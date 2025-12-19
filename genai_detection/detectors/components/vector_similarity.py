import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity as cs


def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    Koppel et al. (2014) have use cosine similarity as a baseline.
    """
    if vec1 is None or vec2 is None:
        return 0.0

        # Ensure both are numpy arrays
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    # Reshape only if 1D
    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1:
        vec2 = vec2.reshape(1, -1)

    # Compute cosine similarity (returns 2D array)
    return cs(vec1, vec2).flatten()[0]


def minmax_similarity(vec1, vec2):
    """
    Calculate min-max similarity between two vectors in TFIDF format.
    Koppel et al. (2014) use min-max similarity.
    """
    if vec1 is None or vec2 is None:
        return 0.0
    # Ensure vectors are 1D numpy arrays
    if isinstance(vec1, csr_matrix):
        vec1 = vec1.toarray().flatten()
    if isinstance(vec2, csr_matrix):
        vec2 = vec2.toarray().flatten()
    assert len(vec1) == len(vec2), f"Vectors must be of the same length but have {len(vec1)} and {len(vec2)}."
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    numerator = np.minimum(vec1, vec2).sum()
    denominator = np.maximum(vec1, vec2).sum()
    return 0.0 if denominator == 0 else numerator / denominator
