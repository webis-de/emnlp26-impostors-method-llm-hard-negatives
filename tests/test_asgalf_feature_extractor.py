import unittest

from ablations.impostor_asgalf import ASGALFFeatureExtractor


class TestASGALFFeatureExtractor(unittest.TestCase):
    def test_ngram_bounds_invalid_min_zero(self):
        with self.assertRaises(ValueError):
            ASGALFFeatureExtractor(ngram_min=0, ngram_max=3)

    def test_ngram_bounds_invalid_min_greater_than_max(self):
        with self.assertRaises(ValueError):
            ASGALFFeatureExtractor(ngram_min=4, ngram_max=3)

    def test_ngram_bounds_valid(self):
        extractor = ASGALFFeatureExtractor(ngram_min=1, ngram_max=3)
        self.assertEqual(extractor.ngram_min, 1)
        self.assertEqual(extractor.ngram_max, 3)


if __name__ == "__main__":
    unittest.main()
