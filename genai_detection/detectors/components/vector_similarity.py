import numpy as np
from more_itertools import argmax
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

def extended_minmax_similarity(candidate, disputed, candidate_fillers):
    """
    Calculate extended min-max similarity between two vectors in TFIDF format given a background set of fillers.
    Khonji & Iraqi (2014) use this version of the min-max similarity.

    We ensure it is a valid metric between 0 and 1.
    """
    def get_most_similar_to_reference(reference, fillers):
        filler_similarities = [
            minmax_similarity(reference, filler) for filler in fillers
        ]
        return fillers[argmax(filler_similarities)]
    if not candidate_fillers:
        return 0.0
    max_sim_imp_to_candidate = get_most_similar_to_reference(reference=candidate, fillers=candidate_fillers)
    max_sim_imp_to_disputed = get_most_similar_to_reference(
        reference=disputed, fillers=candidate_fillers
    )
    candidate_disputed_min_max = minmax_similarity(candidate, disputed)
    denominator = minmax_similarity(candidate, max_sim_imp_to_candidate)*minmax_similarity(disputed, max_sim_imp_to_disputed)

    if denominator == 0:
        return 0.0
    return min(1.0, candidate_disputed_min_max**2/denominator)

def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Manhattan (L1) distance between two vectors.

    Inputs:
        a: np.ndarray A.
        b: np.ndarray B.

    Returns:
        L1 distance as a float.
    """
    return float(np.abs(a - b).sum())
