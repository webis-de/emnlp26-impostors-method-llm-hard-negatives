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
import re
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


class DipperParaphraser(OneStepParaphraser):
    """
    DIPPER paraphrasing model.

    The model expects a short context/prefix plus one sentence to paraphrase, marked
    with ``<sent>...</sent>``. Longer texts are processed sentence by sentence, feeding
    the previously generated output back as context.

    References
    ==========
    - DIPPER Model: https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl
    - DIPPER paper: https://aclanthology.org/2023.eacl-main.117/
    """

    def __init__(
        self,
        model_id: str = "tuner007/pegasus_paraphrase",#"kalpeshk2011/dipper-paraphraser-xxl",
        tokenizer_id: str = "google/t5-v1_1-xxl",
        lexical_diversity: int = 60,
        order_diversity: int = 0,
        sent_interval: int = 1,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.lexical_diversity = lexical_diversity
        self.order_diversity = order_diversity
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

        ## DIPPER
        # self.tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_id)
        # logging.info("Loaded DipperParaphraser tokenizer")
        # # load in FP16
        # dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        # self.model = T5ForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=dtype,
        #         low_cpu_mem_usage=True, ).to(self.device)


        ## smaller paraphraser
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_id)

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=dtype,
                low_cpu_mem_usage=True, ).to(self.device)

        self.model.eval()
        logging.info("Loaded %s paraphraser model", self.model_id)



    @staticmethod
    def _validate_diversity(value: int, name: str) -> int:
        if value not in {0, 20, 40, 60, 80, 100}:
            raise ValueError(f"{name} must be one of 0, 20, 40, 60, 80, 100.")
        return 100 - value

    def paraphrase(
        self,
        text: str,
        prompt: str = "",
        max_length: int = CONFIG.MAX_LENGTH,
        lexical_diversity: Optional[int] = None,
        order_diversity: Optional[int] = None,
        sent_interval: Optional[int] = None,
        do_sample: bool = True,
        top_p: float = 0.75,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Generate a paraphrase with DIPPER.

        :param text: The input text to be paraphrased.
        :param prompt: Optional context prepended before the text.
        :param max_length: Maximum number of output tokens per generation step.
        :param lexical_diversity: DIPPER lexical diversity control in {0, 20, 40, 60, 80, 100}.
        :param order_diversity: DIPPER order diversity control in {0, 20, 40, 60, 80, 100}.
        :param sent_interval: Number of sentences paraphrased per generation step.
        :param do_sample: Whether to sample during generation.
        :param top_p: Nucleus sampling value.
        :param top_k: Top-k sampling value.
        :return: A paraphrased version of the input text.
        """
        lex_code = self._validate_diversity(
            self.lexical_diversity if lexical_diversity is None else lexical_diversity,
            "lexical_diversity",
        )
        order_code = self._validate_diversity(
            self.order_diversity if order_diversity is None else order_diversity,
            "order_diversity",
        )
        interval = self.sent_interval if sent_interval is None else sent_interval
        if interval < 1:
            raise ValueError("sent_interval must be at least 1.")

        sentences = sent_tokenize(text)
        prefix = prompt.strip()
        output_text = ""

        for sent_idx in range(0, len(sentences), interval):
            curr_sent_window = " ".join(sentences[sent_idx : sent_idx + interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            tokenized = self.tokenizer(
                [final_input_text],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            tokenized = {
                key: value.to(self.device) for key, value in tokenized.items()
            }

            generation_args = {
                "max_length": max_length,
                "do_sample": do_sample,
                "top_p": top_p,
            }
            if top_k is not None:
                generation_args["top_k"] = top_k

            with torch.no_grad():
                generated = self.model.generate(**tokenized, **generation_args)
            decoded = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()

            output_text = f"{output_text} {decoded}".strip()
            prefix = f"{prefix} {decoded}".strip()

        return output_text


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
