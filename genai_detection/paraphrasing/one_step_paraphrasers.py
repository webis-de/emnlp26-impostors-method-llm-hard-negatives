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
import json
import logging
import os
import random
import re
import dspy
from typing import List, Optional

import openai
import torch
from nltk import sent_tokenize
from openai import OpenAI
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, PegasusForConditionalGeneration, PegasusTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer, )

from genai_detection.config import CONFIG
from genai_detection.paraphrasing.paraphraser import Paraphraser

logger = logging.getLogger(__name__)

class OneStepParaphraser(Paraphraser):
    """
    Abstract base class for a paraphrasing model.
    This paraphrasing model receives a prompt and an input text and instructs an LM to alter the text according to the prompt.
    """

    @staticmethod
    def _sentence_tokenized_chunks(
        tokenizer, input_text: str, max_tokens: int = CONFIG.MAX_LENGTH
    ) -> List[str]:
        sentences = sent_tokenize(input_text)
        chunks = []
        current_chunk = ""
        current_len = 0

        for sentence in sentences:
            tokenized = tokenizer.encode(sentence, add_special_tokens=False)
            token_len = len(tokenized)

            # If adding this sentence would exceed limit, start a new chunk
            if current_len + token_len > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_len = token_len
            else:
                current_chunk += " " + sentence
                current_len += token_len

        # Append final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


class T5ChatGPTParaphraser(OneStepParaphraser):
    """
    A paraphrasing model based on the T5 architecture.
    """

    def __init__(self):
        """
        Initializes the T5 paraphraser model and tokenizer.
        The model was trained on our ChatGPT paraphrase dataset.


        References
        ===========
        - T5 Model: https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base (12.06.2025)
        - ChatGPT paraphrase dataset: https://huggingface.co/datasets/humarin/chatgpt-paraphrases (12.06.2025)
        @inproceedings{chatgpt_paraphraser,
            author={Vladimir Vorobev, Maxim Kuznetsov},
            title={A paraphrasing model based on ChatGPT paraphrases},
            year={2023}
            }

        """
        self.model_id = "humarin/chatgpt_paraphraser_on_T5_base"
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Apple Silicon GPU
        elif torch.cuda.is_available() and torch.version.cuda is not None:
            self.device = torch.device("cuda")  # NVIDIA GPU
        else:
            self.device = torch.device("cpu")
        logging.info("Using T5ChatGPTParaphraser")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        logging.info("Loaded T5ChatGPTParaphraser tokenizer")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, device_map=None  # force CPU load
        ).to(self.device)
        logging.info("Loaded T5ChatGPTParaphraser model")

    def paraphrase(
        self, text: str, prompt: str, max_length=CONFIG.MAX_LENGTH
    ) -> List[str]:
        logging.info("T5 model")
        chunks = self._sentence_tokenized_chunks(
            input_text=text, max_tokens=max_length, tokenizer=self.tokenizer
        )
        results = []

        for i, chunk in enumerate(chunks):
            input_ids = self.tokenizer(
                f"{chunk}\n{prompt.strip()}",
                return_tensors="pt",
                padding="longest",
                max_length=max_length,
                truncation=True,
            ).input_ids.to(self.device)

            outputs = self.model.generate(
                input_ids,
                repetition_penalty=10.0,
                num_return_sequences=5,
                no_repeat_ngram_size=2,
                num_beams=5,
                num_beam_groups=5,
                max_length=max_length,
                diversity_penalty=3.0,
                trust_remote_code=True,
            )
            if len(results) > 0:
                results = torch.cat([results, outputs], dim=1)
            else:
                results = outputs

        res = self.tokenizer.batch_decode(results, skip_special_tokens=True)
        return res


class T5GooglePAWSParaphraser(OneStepParaphraser):
    """
    A paraphrasing model based on the T5 architecture.
    """

    def __init__(self):
        """
        Initializes the T5 paraphraser model and tokenizer.
        The model was trained on our GooglePAWS paraphrase dataset.

        References
        ===========
        - T5 Model: https://huggingface.co/Vamsi/T5_Paraphrase_Paws (12.06.2025)
        - Google PAWS dataset: https://github.com/google-research-datasets/paws (12.06.2025)
        """
        self.model_id = "Vamsi/T5_Paraphrase_Paws"
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Apple Silicon GPU
        elif torch.cuda.is_available() and torch.version.cuda is not None:
            self.device = torch.device("cuda")  # NVIDIA GPU
        else:
            self.device = torch.device("cpu")
        logging.info("Using T5GooglePAWSParaphraser")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        logging.info("Loaded T5GooglePAWSParaphraser tokenizer")
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(
                self.device
            )
        except Exception as e:
            logging.warning(f"[ERROR] Failed to load T5GooglePAWSParaphraser model: {e}")
            self.model = None

        logging.info("Loaded T5GooglePAWSParaphraser model")

    def paraphrase(
        self, text: str, prompt: str, max_length: int = CONFIG.MAX_LENGTH
    ) -> List[str]:
        logging.info("T5 model")
        chunks = self._sentence_tokenized_chunks(
            input_text=text, max_tokens=max_length, tokenizer=self.tokenizer
        )
        results = []

        for i, chunk in enumerate(chunks):
            encoding = self.tokenizer.encode_plus(
                f"{chunk}\n{prompt.strip()}</s>",
                padding="max_length",
                return_tensors="pt",
            )

            input_ids, attention_masks = encoding["input_ids"].to(
                self.device
            ), encoding["attention_mask"].to(self.device)

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                max_length=max_length,
                do_sample=True,
                top_k=120,
                top_p=0.95,
                num_return_sequences=1,
                trust_remote_code=True,
            )
            if len(results) > 0:
                try:
                    results = torch.cat([results, outputs], dim=1)

                except Exception as e:
                    logging.warning(e)
                    logging.warning("failed text chunk: %s %s", results, outputs)
            else:
                results = outputs

        outputs = results

        res = []
        for output in outputs:
            line = self.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            res.append(line)
        return res


class FinetunedParaphraser(OneStepParaphraser):
    """
    Local seq2seq paraphraser backed by a fine-tuned Pegasus model.

    The default model, ``tuner007/pegasus_paraphrase``, works best on short
    sentence-level inputs. Longer texts are split into sentence windows and each
    window is paraphrased independently. To avoid repeated identical paraphrases,
    generation samples multiple candidates per window and returns one complete
    sampled variant.

    References
    ==========
    - Pegasus paraphrase model: https://huggingface.co/tuner007/pegasus_paraphrase
    """

    def __init__(
        self,
        model_id: str = "tuner007/pegasus_paraphrase",
        sent_interval: int = 1,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.sent_interval = sent_interval

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available() and torch.version.cuda is not None:
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logging.info("Using %s Paraphraser on %s", self.model_id, self.device)
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_id)

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=dtype,
                low_cpu_mem_usage=True, ).to(self.device)
        self.max_model_input_length = min(
            getattr(self.tokenizer, "model_max_length", 1024),
            getattr(self.model.config, "max_position_embeddings", 1024),
        )
        self.max_model_output_length = getattr(
            self.model.config, "max_position_embeddings", self.max_model_input_length
        )

        self.model.eval()
        logging.info("Loaded %s paraphraser model", self.model_id)

    def paraphrase(
        self,
        text: str,
        prompt: str = "",
        max_length: int = 128,#CONFIG.MAX_LENGTH,
        lexical_diversity: Optional[int] = None,
        order_diversity: Optional[int] = None,
        sent_interval: Optional[int] = 1,
        do_sample: bool = True,
        temperature: float = 1.1,
        top_p: float = 0.98,
        top_k: Optional[int] = 120,
        num_return_sequences: int = 8,
    ) -> str:
        """
        Generate one sampled paraphrase with the local Pegasus paraphraser.

        :param text: The input text to be paraphrased.
        :param prompt: Accepted for interface compatibility, but ignored because
            Pegasus expects plain source text rather than chat instructions.
        :param max_length: Maximum number of output tokens per generation step.
        :param lexical_diversity: Accepted for compatibility with the old DIPPER
            interface, but unused by Pegasus.
        :param order_diversity: Accepted for compatibility with the old DIPPER
            interface, but unused by Pegasus.
        :param sent_interval: Number of sentences paraphrased per generation step.
        :param do_sample: Whether to sample during generation.
        :param temperature: Sampling temperature.
        :param top_p: Nucleus sampling value.
        :param top_k: Top-k sampling value.
        :param num_return_sequences: Number of sampled candidates per sentence window.
        :return: A paraphrased version of the input text.
        """
        interval = self.sent_interval if sent_interval is None else sent_interval
        if interval < 1:
            raise ValueError("sent_interval must be at least 1.")

        sentences = sent_tokenize(text)

        if not sentences:
            return ""

        paraphrase_variants = ["" for _ in range(max(num_return_sequences, 1))]
        input_max_length = min(max_length, self.max_model_input_length)
        output_max_length = min(max_length, self.max_model_output_length)

        for sent_idx in range(0, len(sentences), interval):
            # Process short sentence windows because the Pegasus paraphrase model is
            # trained for sentence-level inputs and has a limited position budget.
            sentence_window = " ".join(sentences[sent_idx: sent_idx + interval])

            # Pegasus expects ordinary text, not chat instructions or DIPPER control tokens.
            input_text = sentence_window

            # Truncate to the model's real input limit to avoid CUDA embedding
            # asserts when a sentence window is longer than Pegasus can encode.
            tokenized = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True,
                    max_length=input_max_length, ).to(self.device)

            # Sample multiple candidates per window and discourage copied or
            # repetitive wording so repeated calls produce more diverse outputs.
            generation_args = {
                    "max_length": output_max_length,
                    "do_sample": do_sample,
                    "num_return_sequences": max(num_return_sequences, 1),
                    "repetition_penalty": 1.15,
                    "no_repeat_ngram_size": 3,
                    }

            if do_sample:
                generation_args["temperature"] = temperature
                generation_args["top_p"] = top_p

                if top_k is not None:
                    generation_args["top_k"] = top_k

            # inference_mode disables gradient bookkeeping and keeps GPU memory
            # usage lower during generation.
            with torch.inference_mode():
                generated = self.model.generate(**tokenized, **generation_args, )

            # Decode all sampled sequences; they are stitched by candidate index
            # below to keep full-text variants internally consistent.
            decoded = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Keep candidate index stable across windows so each variant is one
            # coherent full-text paraphrase, then randomly choose among variants.
            for variant_idx, candidate in enumerate(decoded):
                candidate = candidate.strip()
                paraphrase_variants[variant_idx] = (
                    f"{paraphrase_variants[variant_idx]} {candidate}".strip()
                )

        candidates = [candidate for candidate in paraphrase_variants if candidate]
        return random.choice(candidates) if candidates else ""


class SAIAParaphraser(OneStepParaphraser):
    """
    SAIA paraphrasing model hosted by GWDG (Gesellschaft für wissenschaftliche Datenverarbeitung mbH Göttingen).
    SAIA is the Scalable Artificial Intelligence (AI) Accelerator that hosts our AI services.

    For more information, see https://docs.hpc.gwdg.de/services/saia/index.html#api-request (09.08.2025).
    """

    def __init__(self, model_id: str = CONFIG.SAIA_MODEL):
        self._init_client()
        self.model_id = model_id

    def _init_client(self, saia_api_key:str= CONFIG.SAIA_KEY):
        if not saia_api_key:
            saia_api_key = CONFIG.SAIA_KEY if os.path.exists("/Users/klara") else os.environ["SAIA_KEY"]

        self.client = OpenAI(
            base_url=(
                CONFIG.SAIA_URL
                if os.path.exists("/Users/klara")
                else os.environ["SAIA_URL"]
            ),
            api_key=saia_api_key,
        )

    def paraphrase(
        self,
        text: str,
        prompt: str,
        max_length: float = CONFIG.MAX_LENGTH,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
    ) -> str:
        """
        Generate paraphrased versions of the input text.

        :param text: The input text to be paraphrased.
        :param prompt: The prompt to be used for paraphrasing. This model allows for JSON structured ouput, hence, specify here the prompt to be used for paraphrasing.
        The prompt is inserted after the text to enforce its importance in the LLM's context when working on long texts.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :return: A paraphrased version of the input text.
        """
        # max token differs across models but usually at least 40k tokens, so we crop at less to be safe
        body = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a paraphrasing assistant. Output only the final paraphrased text.",
                },
                {"role": "user", "content": f"{text[:30000]}\n{prompt.strip()}"},
            ],
            "temperature": CONFIG.TEMPERATURE if temperature is None else temperature,
        }
        if top_p is not None:
            body["top_p"] = top_p
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        try:
            response = self.client.chat.completions.create(**body)
            resp = response.choices[0].message.content
            resp = re.sub("'", " ", resp)  # replace single quotes with double quotes
            resp = re.sub(r"\s+", " ", resp)  # remove excessive whitespaces

            try:
                data = json.loads(resp)
                return data
            except json.JSONDecodeError:
                return response.choices[0].message.content

        except openai.RateLimitError as e:
            logging.warning(
                f"[ERROR] Rate limit exceeded for {self.model_id}: {e}. Sleeping 1 minute and trying again..."
            )
            return self.paraphrase(text=text, prompt=prompt, max_length=max_length)


class OllamaParaphraser(SAIAParaphraser):
    """
    Ollama paraphrasing model hosted by Webis.
    """

    def __init__(self, model_id: str = CONFIG.OLLAMA_MODEL):
        super().__init__(model_id=model_id)
        self.client = OpenAI(
            base_url=CONFIG.OLLAMA_URL if os.path.exists("/Users/klara") else os.environ['OLLAMA_URL'],
            api_key=CONFIG.OLLAMA_KEY if os.path.exists("/Users/klara") else os.environ['OLLAMA_KEY'],
        )




class DSPyOneStepSignature(dspy.Signature):
    """Rewrite the input text as a faithful paraphrase."""

    text: str = dspy.InputField(desc="The original text to paraphrase.")
    prompt: str = dspy.InputField(desc="Additional paraphrasing instruction.")
    paraphrase: str = dspy.OutputField(
        desc="Only the final paraphrased text, preserving meaning and tone."
    )


class DSPyOneStepParaphraser(OneStepParaphraser):
    def __init__(
        self,
        model_id: str = CONFIG.OPENAI_MODEL,
        temperature: float = CONFIG.TEMPERATURE,
    ):
        super().__init__(n_paraphrases=1, model_id=model_id)

        is_reasoning_model = any(name in model_id.lower() for name in ["gpt-5"])

        self.lm = dspy.LM(
            model_id,
            api_base=CONFIG.OPENAI_URL if os.path.exists("/Users/klara") else os.environ["OPENAI_URL"],
            api_key=CONFIG.OPENAI_KEY if os.path.exists("/Users/klara") else os.environ["OPENAI_KEY"],
            model_type="chat",
            cache=False,
            temperature=1.0 if is_reasoning_model else temperature,
            max_tokens=16000 if is_reasoning_model else CONFIG.MAX_LENGTH,
        )
        dspy.configure(lm=self.lm)
        self.generator = dspy.Predict(DSPyOneStepSignature)

    def paraphrase(
        self,
        text: str,
        prompt: str = CONFIG.PROMPT,
        max_length: int = CONFIG.MAX_LENGTH,
    ) -> str:
        result = self.generator(text=text, prompt=prompt)
        return result.paraphrase.strip()
