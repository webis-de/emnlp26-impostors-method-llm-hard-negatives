# Copyright 2024 Klara M. Gutekunst, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, List, Optional, Sequence

import evaluate
import gensim.downloader
import nltk
import numpy as np
import torch
from nltk.translate import bleu_score, meteor_score
from sentence_transformers import SentenceTransformer
from word_mover_distance import model  # https://pypi.org/project/word-mover-distance/

logger = logging.getLogger(__name__)


class WMDReadyKeyedVectors:
    """Adapter so gensim vectors expose the interface expected by `word_mover_distance`."""

    def __init__(self, keyed_vectors):
        self.model = keyed_vectors

    def __getitem__(self, key):
        return self.model[key]

    def __contains__(self, key):
        return key in self.model

    def keys(self):
        return self.model.key_to_index.keys()


class ParaphraseMetricCalculator:
    """
    Compute lexical and semantic paraphrase quality metrics.
    """

    _ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")
    _DEFAULT_METRICS = [
        "bleu_score",
        "meteor_score",
        "rouge1",
        "rouge2",
        "rougeL", # if no line-breaks exists, rougeLsum behaves like rougeL
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "sbert_wms",
        "sbert_cos",
        "sem_sim_avg",
        "syn_sim_avg",
        "gohsen_delta",
    ]

    def __init__(
        self,
        include_meteor: bool = True,
        include_wmd: bool = True,
        include_sbert_cos: bool = True,
    ):
        self.include_meteor = include_meteor
        self.include_wmd = include_wmd
        self.include_sbert_cos = include_sbert_cos
        self.rouge_score = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")

        if self.include_meteor:
            self._ensure_wordnet_available()
        self.sbert_model = (
            self._load_sentence_transformer() if self.include_sbert_cos else None
        )
        self.wmd_model = self._load_wmd_model() if self.include_wmd else None

    @staticmethod
    def _ensure_wordnet_available() -> None:
        """Ensure METEOR dependencies are available without failing import time."""
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            try:
                nltk.download("wordnet", quiet=True)
            except Exception as e:
                logger.warning("Could not download NLTK WordNet corpus for METEOR: %s", e)

    @staticmethod
    def _load_sentence_transformer():
        """Load SBERT encoder used for cosine similarity."""
        try:
            return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning("Failed to load SentenceTransformer model: %s", e)
            return None

    @staticmethod
    def _load_wmd_model():
        """Load vectors for Word Mover's Distance with retry and graceful fallback."""
        logger.info("Loading pre-trained word vectors for WMD...")
        tries = 0
        pretr_word_model = None
        while not pretr_word_model and tries < 10:
            try:
                pretr_word_model = WMDReadyKeyedVectors(
                    gensim.downloader.load("glove-twitter-25")
                )
            except Exception as e:
                logger.warning("Failed to load gensim glove-twitter-25. Retrying... %s", e)
                tries += 1
        if pretr_word_model is None:
            logger.warning("WMD vectors unavailable after retries; WMD metric will default to 0.0.")
            return None
        return model.WordEmbedding(model=pretr_word_model)

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return float(np.mean(np.asarray(values, dtype=float)))

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        """Ensure prompts consistently carry the `<TEXT>` placeholder exactly once."""
        prompt = str(prompt)
        return prompt if "<TEXT>" in prompt else f"{prompt} <TEXT>"

    def _meteor(self, original_tokens: list[list[str]], paraphrase_tokens: list[list[str]]) -> list[float]:
        """Compute METEOR and return 0.0 if resources are missing or scoring fails."""
        return [meteor_score.single_meteor_score(references=or_tokens, hypothesis=par_tokens) for
                or_tokens, par_tokens in zip(original_tokens, paraphrase_tokens)]

    def _sbert_cosine(self, original_texts: List[str], paraphrases: List[str]) -> list[float]:
        """Compute normalized SBERT cosine similarity in [0, 1]."""
        if not self.include_sbert_cos or self.sbert_model is None:
            return 0.0
        def _sbert_cosine_aux(t:str, p:str):
            cos_sim = torch.cosine_similarity(
                    self.sbert_model.encode(t, convert_to_tensor=True),
                    self.sbert_model.encode(p, convert_to_tensor=True),
                    dim=0,
                    ).item()
            return float((cos_sim + 1.0) / 2.0)
        return [_sbert_cosine_aux(original_text, paraphrase) for original_text, paraphrase in zip(original_texts, paraphrases)]

    def _wmd_similarity(
        self, original_tokens: list[str], paraphrase_tokens: list[str]
    ) -> float:
        """
        Convert WMD distance into similarity with exp(-distance).

        Returns 0.0 when vectors are unavailable or distance computation fails.
        """
        if not self.include_wmd or self.wmd_model is None:
            return 0.0
        try:
            distance = self.wmd_model.wmdistance(
                [tok.lower() for tok in original_tokens],
                [tok.lower() for tok in paraphrase_tokens],
            )
            if not np.isfinite(distance):
                return 0.0
            return float(np.exp(-distance))
        except Exception as e:
            logger.warning("WMD computation failed, defaulting to 0.0: %s", e)
            return 0.0

    def _bleu(self, original_tokens: list[str], paraphrase_tokens: list[str]) -> float:
        """Compute BLEU score."""
        assert original_tokens, "original_tokens must not be empty"
        assert paraphrase_tokens, "paraphrase_tokens must not be empty"
        return float(bleu_score.sentence_bleu(
            references=[original_tokens],
            hypothesis=paraphrase_tokens,
            smoothing_function=bleu_score.SmoothingFunction().method1,
        ))

    def _rouge(self, original_text: str, paraphrase: str) -> dict[str, float]:
        """Compute ROUGE score."""
        assert original_text, "original_tokens must not be empty"
        assert paraphrase, "paraphrase_tokens must not be empty"
        score = self.rouge_score.compute(predictions=[paraphrase], references=[original_text])
        return {key: float(score.get(key, 0.0)) for key in self._ROUGE_KEYS}

    def _compute_lexical_metrics(
        self, original_texts: list[str], paraphrases: list[str], original_texts_tokens: list[list[str]], paraphrases_tokens: list[list[str]]
    ) -> list[dict[str, float]]:
        """Compute token-overlap metrics."""
        # bleu scores: List of floats
        bleu_scores = [
                self._bleu(original_tokens=or_tokens, paraphrase_tokens=par_tokens) for or_tokens, par_tokens in zip(
                original_texts_tokens, paraphrases_tokens)
                ]

        # rouge scores: List of dicts
        rouge_scores = [
                self._rouge(original_text=original_text, paraphrase=paraphrase) for original_text, paraphrase in zip(
                        original_texts, paraphrases)
                ]

        return [{"bleu": bleu_scores[i], **rouge_scores[i]} for i in range(len(original_texts))]

    def _compute_semantic_metrics(
        self,
        original_sentences: list[str],
        paraphrases: list[str],
        original_texts_tokens: list[list[str]],
        paraphrases_tokens: list[list[str]],
    ) -> list[dict[str, float]]:
        """Compute embedding-based semantic metrics for each original/paraphrase pair."""
        out = self._bertscore(
            paraphrases=paraphrases,
            references=original_sentences,
        )

        if self.include_sbert_cos:
            sbert_cosine_scores = self._sbert_cosine(
                original_texts=original_sentences,
                paraphrases=paraphrases,
            )
            out = [
                {**row, "sbert_cos": score}
                for row, score in zip(out, sbert_cosine_scores)
            ]

        if self.include_wmd:
            sbert_wms = [
                self._wmd_similarity(
                    original_tokens=or_tokens,
                    paraphrase_tokens=par_tokens,
                )
                for or_tokens, par_tokens in zip(
                    original_texts_tokens, paraphrases_tokens
                )
            ]
            out = [{**row, "sbert_wms": score} for row, score in zip(out, sbert_wms)]

        if self.include_meteor:
            meteor_scores = self._meteor(original_tokens=original_texts_tokens,
                    paraphrase_tokens=paraphrases_tokens,)
            out = [{**row, "meteor_score": score} for row, score in zip(out, meteor_scores)]

        return out

    def _semantic_average_values(self, semantic_metrics: dict) -> list[float]:
        """Return only enabled semantic components for aggregate averaging."""
        values = [
            semantic_metrics["bertscore_precision"],
            semantic_metrics["bertscore_recall"],
            semantic_metrics["bertscore_f1"],
        ]
        if self.include_wmd:
            values.append(semantic_metrics["sbert_wms"])
        if self.include_sbert_cos:
            values.append(semantic_metrics["sbert_cos"])
        return values

    def _bertscore(
        self, paraphrases: list[str], references: list[str]
    ) -> list[dict[str, float]]:
        scores = self.bertscore.compute(
            predictions=paraphrases,
            references=references,
            model_type="distilbert-base-uncased",
            lang="en",
            batch_size=64,
            idf=False,
            verbose=False,
        )

        return [
            {"precision": p, "recall": r, "f1": f}
            for p, r, f in zip(scores["precision"], scores["recall"], scores["f1"])
        ]

    def get_paraphrase_metrics(
        self,
        paraphrases: List[str],
        original_texts: List[str],
    ) -> list[dict[Any, Any]]:
        """
        Compute lexical and semantic similarity metrics for paraphrase/reference pairs.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries, where each dictionary contains all computed
            metric values for one paraphrase/reference pair.
        """
        assert paraphrases, "paraphrase must not be empty"
        assert original_texts, "original_text must not be empty"
        assert isinstance(paraphrases, List), f"paraphrase must be a List[str], but is of type {type(paraphrases)}"
        assert isinstance(original_texts, List), f"original_text must be a List[str], but is of type {type(original_texts)}"
        assert len(paraphrases) == len(
            original_texts
        ), "paraphrases and original texts must have the same length"

        original_texts_tokens = [original_text.split() for original_text in original_texts]
        paraphrases_tokens = [paraphrase.split() for paraphrase in paraphrases]

        # list of dicts with lexical and semantic metrics
        lexical_metrics = self._compute_lexical_metrics(
            original_texts=original_texts,
            paraphrases=paraphrases,
            original_texts_tokens=original_texts_tokens,
            paraphrases_tokens=paraphrases_tokens
        )
        semantic_metrics = self._compute_semantic_metrics(
            original_sentences=original_texts,
            paraphrases=paraphrases,
            original_texts_tokens=original_texts_tokens,
            paraphrases_tokens=paraphrases_tokens
        )

        results: list[dict[str, Any]] = []
        for lexical, semantic in zip(lexical_metrics, semantic_metrics):
            sem_avg = self._mean(list(semantic.values()))
            syn_avg = self._mean(list(lexical.values()))

            results.append(
                {
                    **lexical,
                    **semantic,
                    "sem_sim_avg": sem_avg,
                    "syn_sim_avg": syn_avg,
                    "gohsen_delta": sem_avg - syn_avg,
                }
            )

        return results

    def get_metric_names(self) -> List[str]:
        """Return currently active metric columns in deterministic order."""
        metric_names = list(self._DEFAULT_METRICS)
        if not self.include_meteor:
            metric_names.remove("meteor_score")
        if not self.include_wmd:
            metric_names.remove("sbert_wms")
        if not self.include_sbert_cos:
            metric_names.remove("sbert_cos")
        return metric_names
