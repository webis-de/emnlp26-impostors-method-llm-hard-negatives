# Copyright 2026 Klara M. Gutekunst, Webis
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
import random
import re
from typing import Optional, List, Dict, Iterable

from bson import ObjectId

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.ImpostorGenerator import LLMImpostorGenerator
from genai_detection.paraphrasing.one_step_paraphrasers import (DipperParaphraser, SAIAParaphraser,
                                                                OneStepParaphraser, )
from genai_detection.paraphrasing.paraphraser import Paraphraser


class NaiveImpostorGenerator(LLMImpostorGenerator):
    def __init__(
        self, n_impostors: int, top_n_freq_words:int, paraphrasers: Optional[List[Paraphraser]] = None,  use_dipper:
            bool = True
    ):
        super().__init__(n_impostors=n_impostors, top_n_freq_words=top_n_freq_words)
        if paraphrasers is None:
            # self.t5_chatgpt_paraphraser = T5ChatGPTParaphraser()
            # self.t5_google_paws_paraphraser = T5GooglePAWSParaphraser()
            # self.ollama_paraphraser = OllamaParaphraser(model_id=CONFIG.OLLAMA_MODEL)

            # self.saiai_paraphraser_llama = SAIAParaphraser(
            #     model_id="meta-llama-3.1-8b-instruct"
            # )
            # self.saiai_paraphraser_mistral = SAIAParaphraser(
            #     model_id="mistral-large-instruct"
            # )
            # self.saiai_paraphraser_gpt = SAIAParaphraser(
            #     model_id="openai-gpt-oss-120b"
            # )
            # self.saiai_paraphraser_qwen = SAIAParaphraser(model_id="qwen3-32b")

            self.paraphrasers = [
                # self.t5_chatgpt_paraphraser,
                # self.t5_google_paws_paraphraser,
                # self.ollama_paraphraser,

                # self.saiai_paraphraser_llama,
                # self.saiai_paraphraser_mistral,
                # self.saiai_paraphraser_gpt,
                # self.saiai_paraphraser_qwen,  # explanations in the output, separated by </think>
            ]
            if use_dipper:
                self.dipper_paraphraser = DipperParaphraser()
                self.paraphrasers.append(self.dipper_paraphraser)
        else:
            assert all(
                isinstance(p, Paraphraser) for p in paraphrasers
            ), "All paraphrasers must be instances of Paraphraser or its subclasses."
            assert (
                len(paraphrasers) > 0
            ), "At least one paraphraser must be provided."
            self.paraphrasers = paraphrasers
        self.prompt_bank = [
            CONFIG.PROMPT,
            "Rewrite the text to preserve meaning but restructure sentences. Avoid copying original phrasing. Output only the paraphrase.",
            "Paraphrase the text with different syntax and sentence order. Keep all facts and named entities. No explanations.",
            "Express the same ideas with different wording and clause structure. Keep tone and meaning. Output only the paraphrase.",
        ]
        self.decoding_param_bank = [
            {"temperature": min(CONFIG.TEMPERATURE + 0.2, 1.3), "top_p": 0.95, "frequency_penalty": 0.2},
            {"temperature": min(CONFIG.TEMPERATURE + 0.4, 1.5), "top_p": 0.9, "frequency_penalty": 0.3},
        ]
        self.max_candidates_per_impostor = 3
        self.min_len_ratio = 0.6
        self.max_len_ratio = 3.5
        self.min_token_jaccard = 0.12
        self.max_token_jaccard = 0.65
        self.max_char_ngram_overlap = 0.75
        self.char_ngram_n = 4

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9']+", text.lower())

    def _char_ngrams(self, text: str) -> set:
        text = re.sub(r"\s+", " ", text.lower()).strip()
        n = self.char_ngram_n
        if len(text) < n:
            return set()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    def _prepare_text_stats(self, text: str) -> Dict:
        tokens = self._tokenize(text)
        return {
            "tokens": tokens,
            "token_set": set(tokens),
            "token_len": len(tokens),
            "char_ngrams": self._char_ngrams(text),
            "numbers": {t for t in tokens if t.isdigit()},
        }

    def _normalize_candidate(self, candidate: str, paraphraser: Paraphraser) -> str:
        if (
            isinstance(paraphraser, OneStepParaphraser)
            and paraphraser.model_id == "qwen3-32b"
            and "</think>" in candidate
        ):
            candidate = candidate.split("</think>")[-1]
        candidate = re.sub(r"\s+", " ", candidate).strip()
        return candidate

    def _score_candidate(self, candidate: str, stats: Dict) -> Optional[float]:
        tokens = self._tokenize(candidate)
        if not tokens:
            return None
        if stats["token_len"] == 0:
            return None

        length_ratio = len(tokens) / stats["token_len"]
        if not (self.min_len_ratio <= length_ratio <= self.max_len_ratio):
            return None

        cand_numbers = {t for t in tokens if t.isdigit()}
        if stats["numbers"] and not stats["numbers"].issubset(cand_numbers):
            return None

        token_set = set(tokens)
        union = stats["token_set"] | token_set
        jaccard = len(stats["token_set"] & token_set) / max(len(union), 1)
        if not (self.min_token_jaccard <= jaccard <= self.max_token_jaccard):
            return None

        cand_ngrams = self._char_ngrams(candidate)
        if not cand_ngrams or not stats["char_ngrams"]:
            return None
        char_overlap = len(cand_ngrams & stats["char_ngrams"]) / max(
            len(stats["char_ngrams"]), 1
        )
        if char_overlap > self.max_char_ngram_overlap:
            return None

        return (1.0 - char_overlap) + (0.3 * jaccard)

    def _select_best_candidate(
        self,
        candidates: Iterable[str],
        paraphraser: Paraphraser,
        stats: Dict,
        seen: set,
    ) -> Optional[str]:
        best = None
        best_score = None
        for cand in candidates:
            cand = self._normalize_candidate(cand, paraphraser)
            if not cand:
                continue
            cand_key = cand.lower()
            if cand_key in seen:
                continue
            score = self._score_candidate(cand, stats)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best = cand
                best_score = score
        return best

    def generate_impostors(
        self, text: Optional[str], text_id: Optional[ObjectId]
    ) -> List[str]:
        n_imp_to_generate = self.n_impostors
        if text is None:
            text = self.mongoDB.get_text_or_id_from_orginal_collection(text_id=text_id, text=text)

        impostors, _ = self.obtain_existing_paraphrases(
            collection=self.mongoDB.naive_paraphrase_collection,
            search_args={"text_id": text_id},
        )
        logging.info("Obtained {} existing impostors from naive paraphrases mongodb collection".format(len(impostors)))

        n_imp_to_generate -= len(impostors)
        if n_imp_to_generate <= 0:
            logging.info(
                "Number of impostors in mongodb collection: {}/{} for text with ID: {}. No need to generate more impostors, just returning {} impostors.".format(
                    len(impostors), self.n_impostors, text_id, self.n_impostors
                )
            )
            return impostors[: self.n_impostors]

        if isinstance(text, tuple):
            text = text[0]
        logging.info(
            f"{len(impostors)} precomputed impostors found in mongoDB. Generating {n_imp_to_generate} impostors for text {text[:100]}..."
        )
        stats = self._prepare_text_stats(text)
        seen = {i.lower() for i in impostors}
        max_attempts = max(n_imp_to_generate * self.max_candidates_per_impostor, 1)
        attempts = 0
        logging.info("Start generating {} impostors with {} attempts.".format(n_imp_to_generate,max_attempts))
        while len(impostors) < self.n_impostors and attempts < max_attempts:
            paraphraser = random.choice(self.paraphrasers)
            prompt = random.choice(self.prompt_bank)
            decoding = random.choice(self.decoding_param_bank)
            try:
                if isinstance(paraphraser, SAIAParaphraser):
                    raw = paraphraser.paraphrase(text, prompt=prompt, **decoding)
                else:
                    raw = paraphraser.paraphrase(text, prompt=prompt)

                candidates = raw if isinstance(raw, list) else [raw]
                impostor_text = self._select_best_candidate(
                    candidates, paraphraser, stats, seen
                )
                if not impostor_text:
                    attempts += 1
                    continue

                # automatically inserts model_id
                paraphraser.save_paraphrase_in_mongodb(
                    original_text=text,
                    original_text_id=text_id,
                    paraphrased_text=impostor_text,
                    extracted_info={
                        "prompt": prompt,
                        "decoding": decoding,
                    },
                    total_costs=0,
                    temperature=decoding.get("temperature", CONFIG.TEMPERATURE),
                    prompt=prompt,
                    dataset_name=self.dataset_name,
                    intermediate_prompt="",
                    collection=self.mongoDB.naive_paraphrase_collection,
                )
                impostors.append(impostor_text)
                seen.add(impostor_text.lower())
                logging.info(f"Generated impostor: {len(impostors)}/{self.n_impostors}")
            except Exception as e:
                logging.warning(f"Error generating impostor with {paraphraser}: {e}")
            attempts += 1

        return impostors
