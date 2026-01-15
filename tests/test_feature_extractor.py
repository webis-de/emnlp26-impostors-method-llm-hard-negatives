import unittest

from genai_detection.detectors.components.feature_extractor import TfidfFeatureExtractor


class TestTfidfFeatureExtractor(unittest.TestCase):
    def test_space_free_char_ngrams_basic(self):
        text = "hello to"
        expected = ["hell", "ello", "to  "]
        tokens = TfidfFeatureExtractor._space_free_char_ngrams(text, n=4)
        self.assertEqual(tokens, expected)

    def test_vectorizer_vocab_space_free(self):
        extractor = TfidfFeatureExtractor(top_n_freq_words=1000, min_df=1)
        corpus = ["hello to"]
        extractor.fit_transform(corpus)
        vocab = extractor.vectorizer.vocabulary_
        self.assertIn("hell", vocab)
        self.assertIn("ello", vocab)
        self.assertIn("to  ", vocab)
        self.assertNotIn(" hel", vocab)
        self.assertNotIn("llo ", vocab)
        self.assertNotIn(" to ", vocab)


if __name__ == "__main__":
    unittest.main()
