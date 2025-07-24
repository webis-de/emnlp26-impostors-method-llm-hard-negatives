from abc import ABC
import ast
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, List, Literal, Optional, get_args
import unicodedata
import openai
from pydantic import BaseModel
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from openai import OpenAI
import dirtyjson
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import CONFIG


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)

# cf. https://sdlaml.pages.jsc.fz-juelich.de</ai/guides/blablador_api_access/ (15.06.2025)
ModelName = Literal[
    "1 - Llama3 405 the best general model and big context size",
    "1 - Ministral 8b - the fast model",
    "1 - Teuken-7B-instruct-research-v0.4 - The OpenGPT-X model",
    "10 Mistral-Nemo-Instruct-2407 - Our fast-experimental - with a large context size",
    # "2 - QwenLong L1 32B - A long context reasoning model from 28.05.2025", # specially trained for reasoning, not available om 16.06.2025
    "3 - DeepCoder-14B-Preview - the code model from 09.04.2025",
    "5 - GritLM-7B - For Chat AND Text Embeddings",
    "alias-code",  # specially trained for code
    "alias-embeddings",  # specially made for embeddings
    "alias-fast",  # high throughput
    "alias-fast-experimental",
    "alias-llama3-huge",  # most accurate, but slowest?
    "alias-opengptx",
    "gpt-3.5-turbo",
    "text-davinci-003",
    "text-embedding-ada-002",
]


class TopicSchema(BaseModel):
    topic: str
    genre: str
    tone: str
    time_period: str
    language_register: str


class BulletSchema(BaseModel):
    bullet_points: list[str]
    genre: str
    tone: str


class TaskSchema(BaseModel):
    task: str
    genre: str
    tone: str
    time_period: str
    language_register: str


class TitleSchema(BaseModel):
    title: str
    genre: str
    tone: str
    time_period: str
    language_register: str


class Paraphraser(ABC):
    """
    Abstract base class for paraphrasing models.
    """

    def paraphrase(
        self,
        text: str,
        prompt: str,
        n_responses: int = 5,
        max_length: int = CONFIG.MAX_LENGTH,
        response_schema: Optional[dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate a paraphrase of the input text.

        :param text: The input text to be paraphrased.
        :param prompt: The prompt to be used for paraphrasing.
        :param n_responses: The number of paraphrases to generate.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :return: A paraphrased version of the input text.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def paraphrase_batch(self, texts: list[str], prompt: str = None) -> list[str]:
        """
        Generate paraphrases for a batch of input texts.

        :param texts: A list of input texts to be paraphrased.
        :param prompt: The prompt to be used for paraphrasing. If None, a default prompt will be used.
        :return: A list of paraphrased versions of the input texts.
        """
        return [self.paraphrase(text=text, prompt=prompt) for text in texts]

    def _post_process_llm_response(self, response: str, key: str) -> str:
        resp = unicodedata.normalize("NFKC", response)
        logger.info(f"\n[DEBUG] Response for {key} extraction: {resp}\n")
        match = re.search(r"\{.*?\}", resp)
        if match:
            resp = match.group(0)
        try:
            return ast.literal_eval(resp).get(key, "")
        except Exception as e:
            try:
                return dirtyjson.loads(resp).get(key, "")
            except Exception as e:
                return resp

    def get_tone(self, text: str) -> str:
        """
        Extract the tone of the input text.

        :param text: The input text from which to extract the tone.
        :return: The tone of the text as a string.
        """
        resp = self.paraphrase(
            text=text,
            n_responses=1,
            prompt="Extract the tone (i.e. quality in the voice that expresses the speaker's feelings or thoughts) of the text. Respond ONLY with a JSON object (i.e. no chain-of-thought, no explanations) in the following format: {'tone':'<tone>'}. Text to extract tone from:",
        )[0]
        return self._post_process_llm_response(resp, key="tone")

    def get_genre(self, text: str) -> str:
        """
        Extract the genre of the input text.

        :param text: The input text from which to extract the genre.
        :return: The genre of the text as a string.
        """
        resp = self.paraphrase(
            text=text,
            n_responses=1,
            prompt="Extract the genre (i.e. subject or style of literature) of the text. Respond ONLY with a JSON object (i.e. no chain-of-thought, no explanations) in the following format: {'genre':'<genre>'}. Text to extract genre from:",
        )[0]
        return self._post_process_llm_response(resp, key="genre")

    def get_time_period(self, text: str) -> str:
        """
        Extract the time period of the input text.

        :param text: The input text from which to extract the time period.
        :return: The time period of the text as a string.
        """
        resp = self.paraphrase(
            text=text,
            n_responses=1,
            prompt="Extract the time period (i.e. when the events in the text take place) of the text. Respond ONLY with a JSON object (i.e. no chain-of-thought, no explanations) in the following format: {'time_period':'<time_period>'}. Text to extract time period from:",
        )[0]
        return self._post_process_llm_response(resp, key="time_period")

    def get_register(self, text: str) -> str:
        """
        Extract the register of the input text.

        :param text: The input text from which to extract the register.
        :return: The register of the text as a string.
        """
        resp = self.paraphrase(
            text=text,
            n_responses=1,
            prompt="Extract the register (i.e. level of formality or informality) of the text. Respond ONLY with a JSON object (i.e. no chain-of-thought, no explanations) in the following format: {'language_register':'<register>'}. Text to extract register from:",
        )[0]
        return self._post_process_llm_response(resp, key="register")


class NaiveParaphraser(Paraphraser):
    """
    A naive paraphrasing model that cannot extract tone or genre at the same time as paraphrasing, since models inherenting from the class proved unable to consistently return a valid JSON object.
    This class is a placeholder and does not implement actual paraphrasing logic.
    """


class NonNaiveParaphraser(Paraphraser):
    """
    Abstract base class for paraphrasing models.
    """

    def __init__(self, text_extractor=None, text_generator=None):
        """
        Initializes the BulletPointParaphraser model.
        This model is a placeholder and does not implement actual paraphrasing logic.

        :param text_extractor: A model or function to extract bullet points, tone and genre from the input text.
        :param text_generator: A model or function to generate text based on the extracted bullet points, tone and genre.
        """
        assert isinstance(
            text_extractor, Paraphraser
        ), f"{text_extractor.__name__} must inherit from Paraphraser"
        assert isinstance(
            text_generator, Paraphraser
        ), f"{text_generator.__name__} must inherit from Paraphraser"
        assert hasattr(
            text_extractor, "paraphrase"
        ), "text_extractor must implement a 'paraphrase' method"
        assert hasattr(
            text_generator, "paraphrase"
        ), "text_generator must implement a 'paraphrase' method"

        self.text_extractor = text_extractor
        self.text_generator = text_generator

    def paraphrase(
        self,
        text: str,
        prompt: str,
        n_responses: int = 5,
        max_length: int = CONFIG.MAX_LENGTH,
        response_schema: Optional[dict[str, Any]] = None,
        ground_truth: Optional[
            dict
        ] = None,  # if any ground truth is available, use it rather than the LLM extracted text
    ) -> List[str]:
        """
        Generate a paraphrase of the input text.

        :param text: The input text to be paraphrased.
        :param prompt: The prompt to be used for paraphrasing.
        :param n_responses: The number of paraphrases to generate.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :param response_schema: Optional schema to validate the response.
        :param ground_truth: Optional ground truth to use instead of the LLM extracted text.
        :return: A paraphrased version of the input text.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class T5ChatGPTParaphraser(NaiveParaphraser):
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
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Apple Silicon GPU
        elif torch.cuda.is_available() and torch.version.cuda is not None:
            self.device = torch.device("cuda")  # NVIDIA GPU
        else:
            self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "humarin/chatgpt_paraphraser_on_T5_base"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "humarin/chatgpt_paraphraser_on_T5_base"
        ).to(self.device)

    def paraphrase(
        self,
        text: str,
        prompt: str,
        num_beams=5,
        num_beam_groups=5,
        n_responses: int = 5,
        repetition_penalty=10.0,
        diversity_penalty=3.0,
        no_repeat_ngram_size=2,
        max_length: int = CONFIG.MAX_LENGTH,
        temperature: float = CONFIG.TEMPERATURE,
        response_schema: Optional[dict[str, Any]] = None,
    ) -> List[str]:
        input_ids = self.tokenizer(
            f"{text}\n{prompt.strip()}",
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        ).input_ids.to(self.device)

        outputs = self.model.generate(
            input_ids,
            repetition_penalty=repetition_penalty,
            num_return_sequences=n_responses,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            max_length=max_length,
            diversity_penalty=diversity_penalty,
        )

        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res


class T5GooglePAWSParaphraser(NaiveParaphraser):
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
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Apple Silicon GPU
        elif torch.cuda.is_available() and torch.version.cuda is not None:
            self.device = torch.device("cuda")  # NVIDIA GPU
        else:
            self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "Vamsi/T5_Paraphrase_Paws"
        ).to(self.device)

    def paraphrase(
        self,
        text: str,
        prompt: str,
        n_responses: int = 5,
        max_length: int = CONFIG.MAX_LENGTH,
        temperature: float = CONFIG.TEMPERATURE,
        response_schema: Optional[dict[str, Any]] = None,
    ) -> List[str]:
        # TODO: no duplication penalty, and thus, there are duplicates in the output
        # print(f"[DEBUG] Using T5GooglePAWSParaphraser with prompt: {prompt}")
        encoding = self.tokenizer.encode_plus(
            f"{text}\n{prompt.strip()}</s>", padding="max_length", return_tensors="pt"
        )

        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding[
            "attention_mask"
        ].to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            max_length=max_length,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            num_return_sequences=n_responses,
        )

        res = []
        for output in outputs:
            line = self.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            res.append(line)

        return res


class OllamaParaphraser(NaiveParaphraser):
    """
    Ollama paraphrasing model hosted by Webis.
    """

    def __init__(self, model_id: str = "default:latest"):
        self.client = OpenAI(
            base_url="https://llm.web.webis.de/api",
            api_key=CONFIG.OPENAI_KEY,
        )
        # custom (non-OpenAI) endpoint: Use requests library
        response = requests.get(
            "https://llm.web.webis.de/ollama/api/tags",
            headers={"Authorization": f"Bearer {CONFIG.OPENAI_KEY}"},
        )
        models = response.json()["models"]
        assert model_id in [model["name"] for model in models]
        self.model_id = model_id

    def paraphrase(
        self,
        text: str,
        prompt: str = 'Paraphrase the following text. Respond ONLY with a JSON object in the following format: {"genre":"<genre>","tone":"<tone>","paraphrase":"<paraphrased version of the text>"}. Text to paraphrase:',
        max_length: int = CONFIG.MAX_LENGTH,
        temperature: float = CONFIG.TEMPERATURE,
        n_responses: int = 1,
        response_schema: Optional[
            dict[str, Any]
        ] = None,  # optional pydantic schema to validate the response
    ) -> List[str]:
        """
        Generate paraphrased versions of the input text.

        :param text: The input text to be paraphrased.
        :param prompt: The prompt to be used for paraphrasing. This model allows for JSON structured ouput, hence, specify here the prompt to be used for paraphrasing.
        The prompt is inserted after the text to enforce its importance in the LLM's context when working on long texts.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :param temperature: Controls the randomness of the output. Lower values make the output more deterministic
        :param n_responses: The number of paraphrases to generate.
        :return: A list of paraphrased versions of the input text.
        """
        # if response_schema is None:
        #     print("[WARNING] No response format specified. Using default JSON object format.")
        # print(f"[DEBUG] paraphrase of Ollama: Response format: {format.model_json_schema()}/{type(format)}")
        responses = []
        for i in range(
            n_responses
        ):  # directly using parameter n does not return n responses, but only one response
            body = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": f"{text}\n{prompt.strip()}"}],
                "n": 1,
                "max_tokens": max_length,
                "temperature": temperature,
            }
            # if response_schema:
            #      # text={"format": {"type": "json_object"}},  # FIXME: keyword unknwon even though: https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#json-mode
            #     body['response_format'] = {"type": "json_schema", "json_schema": response_schema}  # use pydantic schema to validate the response, https://ollama.com/blog/structured-outputs
            try:
                response = self.client.chat.completions.create(**body)
            except openai.InternalServerError as e:
                print(
                    f"[ERROR] Failed to generate paraphrase with Ollama: {e}. Skipping..."
                )
                continue
            # print(f"[DEBUG] Response from Ollama paraphraser: {response.choices[0].message.content}")
            # print(f"[DEBUG] Response format: {format.model_json_schema()}/{type(format)}")
            resp = response.choices[0].message.content
            resp = re.sub("'", " ", resp)  # replace single quotes with double quotes
            resp = re.sub(r"\s+", " ", resp)  # remove excessive whitespaces

            try:
                data = json.loads(resp)
                responses.append(data)
            except json.JSONDecodeError:
                responses.append(response.choices[0].message.content)

        return responses


class BlabladorParaphraser(NaiveParaphraser):
    """
    Blablador paraphrasing model hosted by Jülich/ Helmholtz AI.
    """

    def __init__(self, model_id: ModelName = "1 - Ministral 8b - the fast model"):
        self.base_url = "https://api.helmholtz-blablador.fz-juelich.de/v1"
        self.headers = {
            "Authorization": f"Bearer {CONFIG.BLABLADOR_KEY}",
            "Accept": "application/json",
        }
        assert model_id in self._get_available_models(
            verbose=False
        ), f"Model {model_id} is not available. Please choose from the available models."
        self.model_id = model_id

    def _get_available_models(self, verbose: bool = True) -> list[str]:
        """
        Fetch the list of available models from the Blablador API.

        :return: A list of model IDs.
        """
        response = requests.get(f"{self.base_url}/models", headers=self.headers)

        if response.status_code == 200:
            models = [model["id"] for model in response.json()["data"]]
            if verbose:
                print("Available Blablador models:")
                for model in models:
                    print(model)
            return models
        else:
            raise Exception(
                f"Error fetching models: {response.status_code} - {response.text}"
            )

    def paraphrase(
        self,
        text: str,
        prompt: str,
        verbose: bool = False,
        max_length: int = CONFIG.MAX_LENGTH,
        temperature: float = CONFIG.TEMPERATURE,
        n_responses: int = 5,
        response_schema: Optional[dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate a paraphrase of the input text.

        :param text: The input text to be paraphrased.
        :param verbose: If True, prints the paraphrased text.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :param temperature: Controls the randomness of the output. Lower values make the output more deterministic.
        :return: A paraphrased version of the input text.
        """
        payload = {
            "model": self.model_id,  # model ID
            "prompt": f"{prompt.strip()} {text}",
            "max_tokens": max_length,
            "temperature": temperature,
            "n": n_responses,
            "use_beam_search": True,
            "presence_penalty": 2.0,  # encourage diversity, new content
            "frequency_penalty": 2.0,  # discourage repetition, word-level redundancy
        }
        # use completions endpoint for paraphrasing, bc we don't need a multi-turn role-based instruction chat
        response = requests.post(
            f"{self.base_url}/completions",
            headers={**self.headers, "Content-Type": "application/json"},
            data=json.dumps(payload),
        )

        if response.status_code == 200:
            paraphrased_texts = (
                [choice["text"] for choice in response.json()["choices"]]
                if n_responses > 1
                else [response.json()["choices"][0]["text"]]
            )
            if verbose:
                print("Paraphrased text(s):\n", paraphrased_texts)
            return paraphrased_texts
        else:
            print("Error:", response.status_code, response.text)


class BulletPointParaphraser(NonNaiveParaphraser):
    """
    A paraphrasing model that first extracts bullet points, tone and genre from the input text using one LLM and then generates a paraphrase based on this information.
    """

    def __init__(self, text_extractor=None, text_generator=None):
        """
        Initializes the BulletPointParaphraser model.
        This model is a placeholder and does not implement actual paraphrasing logic.

        :param text_extractor: A model or function to extract bullet points, tone and genre from the input text.
        :param text_generator: A model or function to generate text based on the extracted bullet points, tone and genre.
        """
        super().__init__(text_extractor=text_extractor, text_generator=text_generator)
        self.extractor_prompt = 'Summarize the text above in five to six short bullet points. Respond ONLY with a JSON object in the following format: {"bullet_points":"<list of bullet points>","tone":"<tone>","time_period":<time_period>,"language_register":<register>,"target_audience":"<target_audience>","genre":"<genre>"}. Do not use direct quotes.'

    def add_tailoring_quotes(self, value):
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            return value  # already quoted properly
        value = value.strip("'").strip('"').strip()
        return f'"{value}"'

    def _extract_bullet_points(
        self,
        text: str,
        prompt: Optional[str] = None,
        temperature: float = CONFIG.TEMPERATURE,
        key: str = "bullet_points",
        response_schema: Optional[dict[str, Any]] = None,
    ) -> tuple[List[str], str, str, str, str, str]:
        """
        Extract bullet points, tone, and genre from the input text.
        Currently, the text extractor is instructed to avoid direct quotes from the original text, to
        (1) Avoid text duplicates in the paraphrased text, and
        (2) Because (escaping) quotes impede parsing the result using a Python library like ast.

        :param text: The input text.
        :param prompt: Optional custom prompt.
        :param temperature: Controls the randomness of the output. Lower values make the output more deterministic.
        :key: The key to extract from the response. Usually 'bullet_points', but can be customized, because some paraphraser inherent from the class.
        :return: Tuple of bullet points, tone, genre, time period, and register.
        """
        assert self.text_extractor is not None, "Text extractor must be provided."
        if prompt is None:
            prompt = self.extractor_prompt

        logger.info(f"\n[DEBUG] Using text extractor with prompt: {prompt}")
        try:
            res = self.text_extractor.paraphrase(
                text=text,
                prompt=prompt,
                n_responses=1,
                max_length=CONFIG.MAX_LENGTH,
                temperature=temperature,
                response_schema=response_schema,
            )[0]
        except IndexError as e:
            logger.error(f"Error extracting bullet points: {e}")
            return [], "", "", "", "", ""

        logger.info(f"\n[DEBUG] Response from text extractor: {res}\n")

        if isinstance(res, str):
            # res = re.sub(
            #     r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE
            # )

            res = unicodedata.normalize("NFKC", res)  # normalize unicode characters
            match = re.search(
                r"\{.*?\}", res, flags=re.DOTALL
            )  # keep only the JSON part
            if match:
                res = match.group()
            else:
                res = res.strip()
                if not res.endswith("}"):
                    res += "}"
                if not res.startswith("{"):
                    res = "{" + res
        try:
            resp = (
                ast.literal_eval(res) if isinstance(res, str) else res
            )  # ensure res is a dictionary
        except Exception as e:
            try:
                resp = dirtyjson.loads(
                    res
                )  # fallback to dirtyjson.loads if ast.literal_eval fails
            except Exception as e:
                # resp = {key: [res]}
                # TODO: try again, until valid JSON is returned
                print("Again")
                return self._extract_bullet_points(
                    text=text,
                    prompt=prompt,
                    temperature=temperature,
                    key=key,
                    response_schema=response_schema,
                )
        assert isinstance(
            resp, dict
        ), f"Expected a Dictionary response, got {type(resp)}"

        logger.info(
            f"\n[DEBUG] Extracted {key}, tone, genre, time period, and register: {res}\n"
        )
        bullet_points, tone, genre, time_period, register, target_audience = (
            resp.get(key, res),
            resp.get("tone", ""),
            resp.get("genre", ""),
            resp.get("time_period", ""),
            resp.get("language_register", ""),
            resp.get("target_audience", ""),
        )
        return bullet_points, tone, genre, time_period, register, target_audience

    def _generate_paraphrase_from_bullet_points(
        self,
        bullet_points: List[str],
        prompt: Optional[str] = None,
        n_responses: int = 3,
        temperature: float = CONFIG.TEMPERATURE,
    ) -> tuple[List[str], str, str]:
        """
        Generate a paraphrase using the extracted bullet points, tone, and genre.

        :param bullet_points: List of key points.
        :param prompt: Optional prompt to guide generation.
        :param n_responses: Number of paraphrased versions to generate.
        :return: List of paraphrased texts.
        """
        assert self.text_extractor is not None, "Text extractor must be provided."
        if prompt is None:
            # PAN24: (fallback)
            text = (
                bullet_points[0] * 5
            )  # TODO: only bc currently no access to text in this function
            prompt = (
                f"Write a text of about {len(text)} words which covers the following items:"
                + "\n".join(f"- {bp}" for bp in bullet_points)
            )
        return self.text_generator.paraphrase(
            text="",
            prompt=prompt,
            n_responses=n_responses,
            max_length=CONFIG.MAX_LENGTH,
            temperature=temperature,
        )

    def paraphrase(
        self,
        text: str,
        prompt: Optional[str] = None,
        n_responses: int = 5,
        max_length: int = CONFIG.MAX_LENGTH,
        temperature: float = CONFIG.TEMPERATURE,
        response_schema: Optional[dict[str, Any]] = None,
        ground_truth: Optional[dict] = None,
    ) -> List[str]:
        """
        Generate a paraphrase of the input text by first extracting bullet points.

        :param text: The input text to be paraphrased.
        :param prompt: The prompt to be used for extracting the bulletpoints, tone, genre, time period, register (default works well).
        :param n_responses: The number of paraphrases to generate.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :param temperature: Controls the randomness of the output. Lower values make the output more deterministic.
        :param response_schema: Optional schema to validate the response.
        :param ground_truth: Optional ground truth to use instead of the LLM extracted text.
        :return: A paraphrased version of the input text.
        """
        bullet_points, tone, genre, time_period, register, target_audience = (
            self._extract_bullet_points(
                text=text,
                prompt=prompt,
                temperature=temperature,
                response_schema=BulletSchema.model_json_schema(),
            )
        )
        genre = ground_truth.get("genre", genre) if ground_truth else genre
        time_period = (
            ground_truth.get("time_period", time_period)
            if ground_truth
            else time_period
        )
        generator_prompt = (
            f"Do not use asterisks. Write a text of about {len(text.split())} words with a {tone} tone, a {genre} genre, in the {register} register for the target audience of {target_audience} and in the {time_period} time period, covering the following points:\n"
            + "\n".join(f"- {bp}" for bp in bullet_points)
        )
        paraphrased_texts = self._generate_paraphrase_from_bullet_points(
            bullet_points=bullet_points,
            prompt=generator_prompt,
            n_responses=n_responses,
            temperature=temperature,
        )

        return paraphrased_texts


class TaskParaphraser(BulletPointParaphraser):
    """
    A paraphrasing model that first extracts the task upon which the text was generated, tone and genre from the input text using one LLM and then generates a paraphrase based on this information.
    """

    def __init__(self, text_extractor: Paraphraser, text_generator: Paraphraser):
        """
        Initializes the TaskParaphraser model.
        :param text_extractor: A model or function to extract the task that was prompted, tone and genre from the input text.
        :param text_generator: A model or function to generate text based on the extracted task, tone and genre.
        """
        super().__init__(text_extractor=text_extractor, text_generator=text_generator)
        self.extractor_prompt = 'Act as the author of the text above. From that perspective, infer your role or identity, the topic being addressed, and the purpose or instruction behind writing the text. Combine these elements into a concise task prompt that you would give to an LLM to reproduce the text. Respond ONLY with a JSON object in the following format: {"task":"<task>","tone":"<tone>","time_period":<time_period>,"language_register":<register>,"target_audience":"<target_audience>","genre":"<genre>"}.'

    def paraphrase(
        self,
        text: str,
        prompt: Optional[str] = None,
        n_responses: int = 5,
        max_length: int = CONFIG.MAX_LENGTH,
        temperature: float = CONFIG.TEMPERATURE,
        response_schema: Optional[dict[str, Any]] = None,
        ground_truth: Optional[
            dict
        ] = None,  # if any ground truth is available, use it rather than the LLM extracted text
    ) -> List[str]:
        task, tone, genre, time_period, register, target_audience = (
            self._extract_bullet_points(
                text=text,
                prompt=self.extractor_prompt,
                temperature=temperature,
                key="task",
                response_schema=TaskSchema.model_json_schema(),
            )
        )
        genre = ground_truth.get("genre", genre) if ground_truth else genre
        time_period = (
            ground_truth.get("time_period", time_period)
            if ground_truth
            else time_period
        )

        generator_prompt = "Do not use asterisks. Write a text of about {l} words with a {tone} tone, a {genre} genre, in the {register} register for the target audience of {target_audience} and in the {time_period} time period, covering the following task:\n{task}".format(
            l=len(text.split()),
            tone=tone,
            genre=genre,
            task=task,
            time_period=time_period,
            register=register,
            target_audience=target_audience,
        )
        paraphrased_texts = self._generate_paraphrase_from_bullet_points(
            bullet_points=task,
            prompt=generator_prompt,
            n_responses=n_responses,
            temperature=temperature,
        )

        return paraphrased_texts


class TopicParaphraser(BulletPointParaphraser):
    """
    A paraphrasing model that first extracts the topic of the text, tone and genre from the input text using one LLM and then generates a paraphrase based on this information.
    """

    def __init__(self, text_extractor: Paraphraser, text_generator: Paraphraser):
        """
        Initializes the TopicParaphraser model.
        :param text_extractor: A model or function to extract the topic, tone and genre from the input text.
        :param text_generator: A model or function to generate text based on the extracted topic, tone and genre.
        """
        super().__init__(text_extractor=text_extractor, text_generator=text_generator)
        self.extractor_prompt = 'Extract the topic, tone, time period, register, target audience, and genre from the text above. Respond ONLY with a JSON object in the following format: {"topic":"<topic>","tone":"<tone>","time_period":<time_period>,"language_register":<register>,"target_audience":"<target_audience>","genre":"<genre>"}.'

    def paraphrase(
        self,
        text: str,
        prompt: Optional[str] = None,
        n_responses: int = 5,
        max_length: int = CONFIG.MAX_LENGTH,
        temperature: float = CONFIG.TEMPERATURE,
        ground_truth: Optional[
            dict
        ] = None,  # if any ground truth is available, use it rather than the LLM extracted text
    ) -> List[str]:
        logger.info(
            f"\n[DEBUG] Using TopicParaphraser with prompt: {self.extractor_prompt}"
        )
        topic, tone, genre, time_period, register, target_audience = (
            self._extract_bullet_points(
                text=text,
                prompt=self.extractor_prompt,
                temperature=temperature,
                key="topic",
                response_schema=TopicSchema.model_json_schema(),
            )
        )
        logger.info(
            f"\n[DEBUG] Extracted topic, tone, genre, time period, and register."
        )
        genre = ground_truth.get("genre", genre) if ground_truth else genre
        time_period = (
            ground_truth.get("time_period", time_period)
            if ground_truth
            else time_period
        )
        topic = ground_truth.get("topic", topic) if ground_truth else topic

        generator_prompt = "Do not use asterisks. Write a text of about {l} words with a {topic} topic, {tone} tone, a {genre} genre, in the {register} register for the target audience of {target_audience} and in the {time_period} time period.".format(
            l=len(text.split()),
            tone=tone,
            genre=genre,
            topic=topic,
            time_period=time_period,
            register=register,
            target_audience=target_audience,
        )
        paraphrased_texts = self._generate_paraphrase_from_bullet_points(
            bullet_points=topic,
            prompt=generator_prompt,
            n_responses=n_responses,
            temperature=temperature,
        )
        logger.info(f"\n[DEBUG] Generated paraphrased texts: {paraphrased_texts}")

        return paraphrased_texts


class TitleParaphraser(BulletPointParaphraser):
    """
    A paraphrasing model that first extracts the title of the text, tone and genre from the input text using one LLM and then generates a paraphrase based on this information.
    """

    def __init__(self, text_extractor: Paraphraser, text_generator: Paraphraser):
        """
        Initializes the TopicParaphraser model.
        :param text_extractor: A model or function to extract the title, tone and genre from the input text.
        :param text_generator: A model or function to generate text based on the extracted title, tone and genre.
        """
        super().__init__(text_extractor=text_extractor, text_generator=text_generator)
        self.extractor_prompt = 'Find a concise title for the text, extract the tone, time period, register, target audience and genre from the text above. Respond ONLY with a JSON object in the following format: {"title":"<title>","tone":"<tone>","time_period":<time_period>,"language_register":<register>,"target_audience":"<target_audience>","genre":"<genre>"}.'

    def paraphrase(
        self,
        text: str,
        prompt: Optional[str] = None,
        n_responses: int = 5,
        max_length: int = CONFIG.MAX_LENGTH,
        temperature: float = CONFIG.TEMPERATURE,
        response_schema: Optional[dict[str, Any]] = None,
        ground_truth: Optional[
            dict
        ] = None,  # if any ground truth is available, use it rather than the LLM extracted text
    ) -> List[str]:
        title, tone, genre, time_period, register, target_audience = (
            self._extract_bullet_points(
                text=text,
                prompt=self.extractor_prompt,
                temperature=temperature,
                key="title",
                response_schema=TitleSchema.model_json_schema(),
            )
        )
        genre = ground_truth.get("genre", genre) if ground_truth else genre
        time_period = (
            ground_truth.get("time_period", time_period)
            if ground_truth
            else time_period
        )

        generator_prompt = "Do not use asterisks. Write a text of about {l} words with a {title} title, {tone} tone, a {genre} genre, in the {register} register for the target audience of {target_audience} and in the {time_period} time period.".format(
            l=len(text.split()),
            tone=tone,
            genre=genre,
            title=title,
            time_period=time_period,
            register=register,
            target_audience=target_audience,
        )
        paraphrased_texts = self._generate_paraphrase_from_bullet_points(
            bullet_points=title,
            prompt=generator_prompt,
            n_responses=n_responses,
            temperature=temperature,
        )

        return paraphrased_texts


class TranslationParaphraser(NonNaiveParaphraser):
    """
    A paraphrasing model that first extracts the title of the text, tone and genre from the input text using one LLM and then generates a paraphrase based on this information.

    Inspired by the work of:
    C. Zhou, C. Qiu, L. Liang and D. E. Acuna, "Paraphrase Identification With Deep Learning: A Review of Datasets and Methods," in IEEE Access, vol. 13, pp. 65797-65822, 2025, doi: 10.1109/ACCESS.2025.3556899. keywords: {Semantics;Training;Natural language processing;Deep learning;Sports;Reviews;Plagiarism;Electronic mail;Syntactics;Switches;Paraphrase identification;deep learning;review;plagiarism;datasets},


    """

    def __init__(
        self,
        text_extractor: Paraphraser,
        text_generator: Paraphraser,
        language: str = "French",
    ):
        """
        Initializes the TranslationParaphraser model.
        :param text_extractor: A model or function to translate to foreign languages.
        :param text_generator: A model or function to translate from foreign languages.
        """
        super().__init__(text_extractor=text_extractor, text_generator=text_generator)
        self.language = language
        self.extractor_prompt = f"Translate the text above into {self.language}."
        self.generator_prompt = (
            f"Translate the text above from {self.language} into English."
        )

    def paraphrase(
        self,
        text: str,
        prompt: Optional[str] = None,
        n_responses: int = 5,
        max_length: int = CONFIG.MAX_LENGTH,
        temperature: float = CONFIG.TEMPERATURE,
        response_schema: Optional[dict[str, Any]] = None,
        ground_truth: Optional[dict] = None,
    ) -> List[str]:
        translation = self.text_extractor.paraphrase(
            text=text,
            prompt=self.extractor_prompt,
            n_responses=1,
            max_length=max_length,
            temperature=temperature,
            response_schema=response_schema,
        )[0]

        paraphrased_texts = self.text_generator.paraphrase(
            text=translation,
            prompt=self.generator_prompt,
            n_responses=n_responses,
            temperature=temperature,
        )

        return paraphrased_texts


if __name__ == "__main__":
    # models
    ollama_model_id = "mistral:7b"  # "default:latest"
    paraphrasers = {
        # 'T5_ChatGPT': T5ChatGPTParaphraser(),
        # 'T5_Google_PAWS': T5GooglePAWSParaphraser(),
        # 'Blablador': BlabladorParaphraser(model_id="1 - Llama3 405 the best general model and big context size"),
        "Ollama": OllamaParaphraser(model_id=ollama_model_id),
        "TopicParaphraser": TopicParaphraser(
            text_extractor=OllamaParaphraser(model_id=ollama_model_id),
            text_generator=OllamaParaphraser(model_id=ollama_model_id),
        ),
        # "TaskParaphraser": TaskParaphraser(
        #     text_extractor=OllamaParaphraser(model_id=ollama_model_id),
        #     text_generator=OllamaParaphraser(model_id=ollama_model_id),
        # ),
        # "TitleParaphraser": TitleParaphraser(
        #     text_extractor=OllamaParaphraser(model_id=ollama_model_id),
        #     text_generator=OllamaParaphraser(model_id=ollama_model_id),
        # ),
        # "BulletPointParaphraser": BulletPointParaphraser(
        #     text_extractor=OllamaParaphraser(model_id=ollama_model_id),
        #     text_generator=OllamaParaphraser(model_id=ollama_model_id),
        # ),
    }
    # paraphrasers.update({f'Blablador_{name}': BlabladorParaphraser(model_id=name) for name in list(get_args(ModelName))})

    # prompts = [
    #         "Paraphrase the following text and output only the paraphrased version:",
    #         "First, extract bullet points capturing the main ideas, then create a text based on these bullet points. Only output the final text (i.e. do not output the bullet points or any additional chain of thoughts):",
    #         "Paraphrase the sentence by first identifying the main subject, verb, and object. Then find synonyms for each and construct a new sentence. Only output the final paraphrased sentence.",
    #         "Paraphrase the sentence using the same tone as the original with approximately the same number of words:",
    #         "Paraphrase this sentence. Do not change the meaning, but use different words and structure. Output only the paraphrased sentence:"
    #         ]

    # path2datasets = Path(__file__).resolve().parent.parent.parent / "data" / "datasets" / "custom_texts"
    # assert path2datasets.exists(), f"Path to datasets {path2datasets} does not exist."
    # file_name = "cnn_230625"
    # original_text = open(path2datasets / f"{file_name}.txt").read()
