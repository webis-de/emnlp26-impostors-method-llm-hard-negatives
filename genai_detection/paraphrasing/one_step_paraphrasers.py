# Copyright 2025 Klara M. Gutekunst, Webis
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
from typing import List

import openai
import torch
from nltk import sent_tokenize
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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


class SAIAParaphraser(OneStepParaphraser):
    """
    SAIA paraphrasing model hosted by GWDG (Gesellschaft für wissenschaftliche Datenverarbeitung mbH Göttingen).
    SAIA is the Scalable Artificial Intelligence (AI) Accelerator that hosts our AI services.

    For more information, see https://docs.hpc.gwdg.de/services/saia/index.html#api-request (09.08.2025).
    """

    def __init__(self, model_id: str = CONFIG.SAIA_MODEL):
        self.client = OpenAI(
            base_url=CONFIG.SAIA_URL if os.path.exists("/Users/klara") else os.environ['SAIA_URL'],
            api_key=CONFIG.SAIA_KEY if os.path.exists("/Users/klara") else os.environ['SAIA_KEY'],
        )
        self.model_id = model_id

    def paraphrase(
        self, text: str, prompt: str, max_length: float = CONFIG.MAX_LENGTH
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
            "temperature": CONFIG.TEMPERATURE,
        }
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
