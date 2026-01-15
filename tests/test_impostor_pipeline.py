import unittest
from unittest.mock import patch

import numpy as np

from genai_detection.detectors.impostor import ImpostorDetector


class DummyMongo:
    def __init__(self, *args, **kwargs):
        self.impostor_output_collection = "impostor_output"
        self.inserted = []

    def find_document_by_multiple_fields(self, *args, **kwargs):
        return []

    def insert_document(self, collection, insert_data):
        self.inserted.append((collection, insert_data))


class DummyGenerator:
    def __init__(self):
        self.num_potential_impostors = 3

    def generate_impostors_by_text_id(self, text_id):
        return ["impostor_one", "impostor_two"]


class DummyPairProcessor:
    def filter_pairs(self, text_list):
        return [
            {
                "left": {"id": "L", "original_text": "lefttext"},
                "right": {"id": "R", "original_text": "righttext"},
            }
        ]


class DummyFeatureExtractor:
    def __init__(self):
        self.vectorizer = type("Vectorizer", (), {"vocabulary_": {"f": 0}})()
        self.corpus = None

    def fit_transform(self, corpus):
        self.corpus = corpus
        rows = len(corpus)
        data = np.arange(rows, dtype=float).reshape(rows, 1)
        return DummyMatrix(data)


class DummyMatrix:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx):
        return DummyRow(self._data[idx])


class DummyRow:
    def __init__(self, row):
        self._row = row

    def toarray(self):
        return np.asarray([self._row], dtype=float)


class DummyScorer:
    def __init__(self, expected_left_idx, expected_right_idx, expected_left_imps, expected_right_imps):
        self.expected_left_idx = expected_left_idx
        self.expected_right_idx = expected_right_idx
        self.expected_left_imps = expected_left_imps
        self.expected_right_imps = expected_right_imps

    def score_pair(self, pair, vectorizer):
        self._assert_pair_vectors(pair)
        p_values = {
            "left_disputed_right_candidate_uncorrected_p_value": 0.04,
            "right_disputed_left_candidate_uncorrected_p_value": 0.06,
        }
        return 1.0, p_values

    def _assert_pair_vectors(self, pair):
        self._assert_vector(pair["left"]["tfidf"], self.expected_left_idx)
        self._assert_vector(pair["right"]["tfidf"], self.expected_right_idx)

        left_imps = [int(v[0]) for v in pair["left"]["impostors_tfidf"]]
        right_imps = [int(v[0]) for v in pair["right"]["impostors_tfidf"]]
        self._assert_list(left_imps, self.expected_left_imps)
        self._assert_list(right_imps, self.expected_right_imps)

    def _assert_vector(self, vector, expected_idx):
        if int(vector[0]) != expected_idx:
            raise AssertionError(
                f"Expected vector index {expected_idx}, got {vector[0]}"
            )

    def _assert_list(self, values, expected):
        if values != expected:
            raise AssertionError(f"Expected {expected}, got {values}")


class TestImpostorPipeline(unittest.TestCase):
    @patch("genai_detection.detectors.impostor_base.ParaphraseMongoDB", DummyMongo)
    @patch("genai_detection.detectors.impostor.create_impostor_generator", return_value=DummyGenerator())
    def test_pipeline_index_mapping(self, _mock_generator):
        detector = ImpostorDetector(rounds=1, portion_delete=0.0, n_impostors=2, min_n_tokens=1, upsample=False)
        detector.mongoDB = DummyMongo()
        detector.pair_processor = DummyPairProcessor()
        detector.feature_extractor = DummyFeatureExtractor()
        detector.scorer = DummyScorer(
            expected_left_idx=0,
            expected_left_imps=[1, 2],
            expected_right_idx=3,
            expected_right_imps=[4, 5],
        )
        with patch("genai_detection.detectors.impostor.binom_test", return_value=0.5):
            scores = detector.get_score(["lefttext", "righttext"])

        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertEqual(len(detector.mongoDB.inserted), 1)


if __name__ == "__main__":
    unittest.main()
