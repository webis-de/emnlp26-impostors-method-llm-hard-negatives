from abc import ABC
import ast
from collections import defaultdict
import datetime
from itertools import product
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, List, Literal, Optional, get_args
import unicodedata
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel
import requests
import sklearn
# FIXME
from word_mover_distance import model   # https://pypi.org/project/word-mover-distance/
# import gensim.similarities
# from gensim.similarities import WmdSimilarity
from sentence_transformers import SentenceTransformer
import gensim.downloader
import torch
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from openai import OpenAI
nltk.download('wordnet')    # necessary for METEOR score
from nltk.translate import bleu_score, meteor_score
import evaluate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import CONFIG
from genai_detection.util import preprocess_text as _preprocess_text
import dirtyjson

# cf. https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/ (15.06.2025)
ModelName = Literal[
    "1 - Llama3 405 the best general model and big context size",
    "1 - Ministral 8b - the fast model",
    "1 - Teuken-7B-instruct-research-v0.4 - The OpenGPT-X model",
    "10 Mistral-Nemo-Instruct-2407 - Our fast-experimental - with a large context size",
    #"2 - QwenLong L1 32B - A long context reasoning model from 28.05.2025", # specially trained for reasoning, not available om 16.06.2025
    "3 - DeepCoder-14B-Preview - the code model from 09.04.2025",
    "5 - GritLM-7B - For Chat AND Text Embeddings",
    "alias-code",   # specially trained for code
    "alias-embeddings", # specially made for embeddings
    "alias-fast",   # high throughput
    "alias-fast-experimental",
    "alias-llama3-huge",    # most accurate, but slowest?
    "alias-opengptx",
    "gpt-3.5-turbo",
    "text-davinci-003",
    "text-embedding-ada-002",
]

TEMPERATURE = 0.7
MAX_LENGTH = 512  # Maximum length of the generated paraphrase

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
    def paraphrase(self, text: str, prompt: str, n_responses:int=5, max_length:int=MAX_LENGTH, response_schema: Optional[dict[str, Any]] = None,) -> List[str]:
        """
        Generate a paraphrase of the input text.

        :param text: The input text to be paraphrased.
        :param prompt: The prompt to be used for paraphrasing.
        :param n_responses: The number of paraphrases to generate.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :return: A paraphrased version of the input text.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def paraphrase_batch(self, texts: list[str], prompt:str=None) -> list[str]:
        """
        Generate paraphrases for a batch of input texts.

        :param texts: A list of input texts to be paraphrased.
        :param prompt: The prompt to be used for paraphrasing. If None, a default prompt will be used.
        :return: A list of paraphrased versions of the input texts.
        """
        return [self.paraphrase(text=text, prompt=prompt) for text in texts]
    
    def _post_process_llm_response(self, response: str, key:str) -> str:
        resp = unicodedata.normalize("NFKC", response)
        print(f"\n[DEBUG] Response for {key} extraction: {resp}\n")
        match = re.search(r'\{.*?\}', resp)
        if match:
            resp = match.group(0)
        try:
            return ast.literal_eval(resp).get(key, '') 
        except Exception as e:
            try:
                return dirtyjson.loads(resp).get(key, '')  
            except Exception as e:
                return resp
    
    def get_tone(self, text: str) -> str:
        """
        Extract the tone of the input text.
        
        :param text: The input text from which to extract the tone.
        :return: The tone of the text as a string.
        """
        resp = self.paraphrase(text=text, n_responses=1, prompt="Extract the tone (i.e. quality in the voice that expresses the speaker's feelings or thoughts) of the text. Respond ONLY with a JSON object (i.e. no chain-of-thought, no explanations) in the following format: {'tone':'<tone>'}. Text to extract tone from:")[0]
        return self._post_process_llm_response(resp, key='tone')
  
    
    def get_genre(self, text: str) -> str:
        """
        Extract the genre of the input text.
        
        :param text: The input text from which to extract the genre.
        :return: The genre of the text as a string.
        """
        resp = self.paraphrase(text=text, n_responses=1, prompt="Extract the genre (i.e. subject or style of literature) of the text. Respond ONLY with a JSON object (i.e. no chain-of-thought, no explanations) in the following format: {'genre':'<genre>'}. Text to extract genre from:")[0]
        return self._post_process_llm_response(resp, key='genre')

    def get_time_period(self, text: str) -> str:
        """
        Extract the time period of the input text.
        
        :param text: The input text from which to extract the time period.
        :return: The time period of the text as a string.
        """
        resp = self.paraphrase(text=text, n_responses=1, prompt="Extract the time period (i.e. when the events in the text take place) of the text. Respond ONLY with a JSON object (i.e. no chain-of-thought, no explanations) in the following format: {'time_period':'<time_period>'}. Text to extract time period from:")[0]
        return self._post_process_llm_response(resp, key='time_period')

    def get_register(self, text: str) -> str:
        """
        Extract the register of the input text.
        
        :param text: The input text from which to extract the register.
        :return: The register of the text as a string.
        """
        resp = self.paraphrase(text=text, n_responses=1, prompt="Extract the register (i.e. level of formality or informality) of the text. Respond ONLY with a JSON object (i.e. no chain-of-thought, no explanations) in the following format: {'language_register':'<register>'}. Text to extract register from:")[0]
        return self._post_process_llm_response(resp, key='register')
    

class NaiveParaphraser(Paraphraser):
    """
    A naive paraphrasing model that cannot extract tone or genre at the same time as paraphrasing, since models inherenting from the class proved unable to consistently return a valid JSON object.
    This class is a placeholder and does not implement actual paraphrasing logic.
    """
    

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
        self.tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(self.device)

 
    def paraphrase(self, text: str, prompt: str, num_beams=5, num_beam_groups=5, n_responses:int=5, repetition_penalty=10.0, diversity_penalty=3.0, no_repeat_ngram_size=2, max_length:int=MAX_LENGTH, temperature:float=TEMPERATURE, response_schema: Optional[dict[str, Any]] = None,) -> List[str]:
        input_ids = self.tokenizer(f'{prompt.strip()} {text}', return_tensors="pt", padding="longest", max_length=max_length, truncation=True).input_ids.to(self.device)
        
        outputs = self.model.generate(
            input_ids, repetition_penalty=repetition_penalty,
            num_return_sequences=n_responses, no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams, num_beam_groups=num_beam_groups,
            max_length=max_length, diversity_penalty=diversity_penalty
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
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(self.device)

    def paraphrase(self, text: str, prompt:str, n_responses:int=5, max_length:int=MAX_LENGTH, temperature:float=TEMPERATURE, response_schema: Optional[dict[str, Any]] = None,) -> List[str]:
        # TODO: no duplication penalty, and thus, there are duplicates in the output
        # print(f"[DEBUG] Using T5GooglePAWSParaphraser with prompt: {prompt}")
        encoding = self.tokenizer.encode_plus(f'{prompt.strip()} {text} </s>', padding="max_length", return_tensors="pt")
        
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=max_length,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            num_return_sequences=n_responses,
        )

        res = []
        for output in outputs:
            line = self.tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            res.append(line)

        return res
    

class OllamaParaphraser(Paraphraser):
    """
    Ollama paraphrasing model hosted by Webis.
    """

    def __init__(self, model_id: str="default:latest"):
        self.client = OpenAI(
            base_url = "https://llm.web.webis.de/api",  
            api_key = CONFIG.OPENAI_KEY,
            )
        # custom (non-OpenAI) endpoint: Use requests library
        response = requests.get(
            "https://llm.web.webis.de/ollama/api/tags",
            headers={"Authorization": f"Bearer {CONFIG.OPENAI_KEY}"}
        )
        models = response.json()['models']
        assert model_id in [model['name'] for model in models]
        self.model_id = model_id 
    
    def paraphrase(
        self,
        text: str,
        prompt: str='Paraphrase the following text. Respond ONLY with a JSON object in the following format: {"genre":"<genre>","tone":"<tone>","paraphrase":"<paraphrased version of the text>"}. Text to paraphrase:',
        max_length: int = MAX_LENGTH,
        temperature: float = TEMPERATURE,
        n_responses: int = 1,
        response_schema: Optional[dict[str, Any]] = None,  # optional pydantic schema to validate the response
    ) -> List[str]:
        """
        Generate paraphrased versions of the input text.

        :param text: The input text to be paraphrased. 
        :param prompt: The prompt to be used for paraphrasing. This model allows for JSON structured ouput, hence, specify here the prompt to be used for paraphrasing.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :param temperature: Controls the randomness of the output. Lower values make the output more deterministic
        :param n_responses: The number of paraphrases to generate.
        :return: A list of paraphrased versions of the input text.
        """
        # if response_schema is None:
        #     print("[WARNING] No response format specified. Using default JSON object format.")
        # print(f"[DEBUG] paraphrase of Ollama: Response format: {format.model_json_schema()}/{type(format)}")
        responses = []
        for i in range(n_responses):    # directly using parameter n does not return n responses, but only one response
            body = {  "model" : "default:latest",
                "messages" : [{"role": "user", "content": f"{prompt.strip()} {text}"}],
                "n" : 1,
                "max_tokens" : max_length,
                "temperature" : temperature,}
            # if response_schema:
            #      # text={"format": {"type": "json_object"}},  # FIXME: keyword unknwon even though: https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#json-mode
            #     body['response_format'] = {"type": "json_schema", "json_schema": response_schema}  # use pydantic schema to validate the response, https://ollama.com/blog/structured-outputs
            response = self.client.chat.completions.create(**body)
            # print(f"[DEBUG] Response from Ollama paraphraser: {response.choices[0].message.content}")
            # print(f"[DEBUG] Response format: {format.model_json_schema()}/{type(format)}")
            resp = response.choices[0].message.content
            resp = re.sub("'", ' ', resp)  # replace single quotes with double quotes
            resp = re.sub(r'\s+', ' ', resp)  # remove excessive whitespaces
            
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

    def __init__(self, model_id: ModelName="1 - Ministral 8b - the fast model"):
        self.base_url = "https://api.helmholtz-blablador.fz-juelich.de/v1"
        self.headers = {
            "Authorization": f"Bearer {CONFIG.BLABLADOR_KEY}",
            "Accept": "application/json"
        }
        assert model_id in self._get_available_models(verbose=False), f"Model {model_id} is not available. Please choose from the available models."
        self.model_id = model_id 

    def _get_available_models(self,verbose: bool = True) -> list[str]:
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
            raise Exception(f"Error fetching models: {response.status_code} - {response.text}")


    def paraphrase(self, text: str, prompt:str, verbose:bool=False, max_length:int=MAX_LENGTH, temperature:float=TEMPERATURE, n_responses:int=5, response_schema: Optional[dict[str, Any]] = None,) -> List[str]:
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
            "presence_penalty": 2.0,    # encourage diversity, new content
            "frequency_penalty": 2.0,   # discourage repetition, word-level redundancy
        }
        # use completions endpoint for paraphrasing, bc we don't need a multi-turn role-based instruction chat
        response = requests.post(f"{self.base_url}/completions", headers={**self.headers, "Content-Type": "application/json"}, data=json.dumps(payload))

        if response.status_code == 200:
            paraphrased_texts = [choice["text"] for choice in response.json()["choices"]] if n_responses > 1 else [response.json()["choices"][0]["text"]]
            if verbose:
                print("Paraphrased text(s):\n", paraphrased_texts)
            return paraphrased_texts
        else:
            print("Error:", response.status_code, response.text)


class BulletPointParaphraser(Paraphraser):
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
        assert isinstance(text_extractor, Paraphraser), f"{text_extractor.__name__} must inherit from Paraphraser"
        assert isinstance(text_generator, Paraphraser), f"{text_generator.__name__} must inherit from Paraphraser"
        assert hasattr(text_extractor, 'paraphrase'), "text_extractor must implement a 'paraphrase' method"
        assert hasattr(text_generator, 'paraphrase'), "text_generator must implement a 'paraphrase' method"
        
        self.text_extractor = text_extractor
        self.text_generator = text_generator

    def add_tailoring_quotes(self, value):
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            return value  # already quoted properly
        value = value.strip("'").strip('"').strip() 
        return f'"{value}"'

    def _extract_bullet_points(self, text: str, prompt: Optional[str], temperature:float=TEMPERATURE, key:str='bullet_points', response_schema: Optional[dict[str, Any]] = None,) -> tuple[List[str], str, str, str, str]:
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
            prompt = 'Summarize the following text in five to six short bullet points. Respond ONLY with a JSON object in the following format: {"bullet_points":"<list of bullet points>","tone":"<tone>","time_period":<time_period>,"language_register":<register>,"genre":"<genre>"}. Do not use direct quotes. Text to summarize:'
            # TODO: temporarily use this prompt, because the text extractor is not able to return a valid JSON object with the default prompt
            # prompt = 'Summarize the following text in five to six short bullet points. Respond ONLY with a JSON object in the following format: {"genre":"<genre>","tone":"<tone>","bullet_points":"<list of bullet points>"}. Do not use direct quotes. Text to summarize:'

        # print(f"[DEBUG] Using text extractor with prompt: {prompt}")
        # print(f"[DEBUG] reponse schema: ", response_schema, type(response_schema))
        res = self.text_extractor.paraphrase(text=text, prompt=prompt, n_responses=1, max_length=MAX_LENGTH, temperature=temperature, response_schema=response_schema)[0]
    
        if isinstance(res, str):  
            res = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
            try:
                res = unicodedata.normalize("NFKC", res)  # normalize unicode characters
                res = re.search(r'\{.*\}', res).group() # keep only the JSON part
            except Exception as e: # NoneType has no attribute 'group'
                if not res.endswith('}'):
                    res += '}'
                if not res.startswith('{'):
                    res = '{' + res
        try:
            resp = ast.literal_eval(res) if isinstance(res, str) else res  # ensure res is a dictionary
        except Exception as e:
            try:
                resp = dirtyjson.loads(res) # fallback to dirtyjson.loads if ast.literal_eval fails
            except Exception as e:
                # resp = {key: [res]}
                # TODO: try again, until valid JSON is returned
                return self._extract_bullet_points(text=text, prompt=prompt, temperature=temperature, key=key, response_schema=response_schema)
                print(f"[ERROR] [{key}] Failed to parse text extractor response with ast.literal_eval or dirtyjson as JSON: {res}/{type(res)}\nWith error: {e}\nFor prompt: {prompt}")
        assert isinstance(resp, dict), f"Expected a Dictionary response, got {type(resp)}"

        print(f"\n[DEBUG] Extracted {key}, tone, genre, time period, and register: {res}\n")
        bullet_points, tone, genre, time_period, register = resp.get(key, res), resp.get('tone', ''), resp.get('genre', ''), resp.get('time_period', ''), resp.get('language_register', '')
        return bullet_points, tone, genre, time_period, register
    
    def _generate_paraphrase_from_bullet_points(self, bullet_points, tone, genre, time_period, register, prompt: Optional[str], n_responses:int=3, temperature:float=TEMPERATURE) -> tuple[List[str], str, str]:
        """
        Generate a paraphrase using the extracted bullet points, tone, and genre.

        :param bullet_points: List of key points.
        :param tone: Tone of the original text.
        :param genre: Genre of the original text.
        :param time_period: Time period of the original text.
        :param register: Register of the original text.
        :param prompt: Optional prompt to guide generation.
        :param n_responses: Number of paraphrased versions to generate.
        :return: List of paraphrased texts.
        """
        assert self.text_extractor is not None, "Text extractor must be provided."
        if prompt is None:
            # PAN24: (fallback)
            text = bullet_points[0] * 5 # TODO: only bc currently no access to text in this function
            prompt = f"Write a text of about {len(text)} words which covers the following items:"  + "\n".join(f"- {bp}" for bp in bullet_points)
        return self.text_generator.paraphrase(text='', prompt=prompt, n_responses=n_responses, max_length=MAX_LENGTH, temperature=temperature)


    def paraphrase(self, text: str, prompt:Optional[str], n_responses:int=5, max_length:int=MAX_LENGTH, temperature:float=TEMPERATURE, response_schema: Optional[dict[str, Any]] = None,) -> List[str]:
        """
        Generate a paraphrase of the input text by first extracting bullet points.
        
        :param text: The input text to be paraphrased.
        :param prompt: The prompt to be used for extracting the bulletpoints, tone, genre, time period, register (default works well).
        :param n_responses: The number of paraphrases to generate.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :return: A paraphrased version of the input text.
        """
        bullet_points, tone, genre, time_period, register = self._extract_bullet_points(text=text, prompt=prompt, temperature=temperature, response_schema=BulletSchema.model_json_schema())
        # time period and register breaks paraphrasing with LLM for some reason, so we do not specify them in the prompt
        # time_period = self.get_time_period(text=text) if time_period=='' else time_period
        # register = self.get_register(text=text) if register=='' else register
        generator_prompt = (
                f"Do not use asterisks. Write a text of about {len(text.split())} words with a {tone} tone, a {genre} genre, in the {register} register and in the {time_period} time period, covering the following points:\n"
                + "\n".join(f"- {bp}" for bp in bullet_points)
            )
        paraphrased_texts = self._generate_paraphrase_from_bullet_points(bullet_points=bullet_points, tone=tone, genre=genre, time_period=time_period, register=register, prompt=generator_prompt, n_responses=n_responses, temperature=temperature)

        return paraphrased_texts
    

class TaskParaphraser(BulletPointParaphraser):
    """
    A paraphrasing model that first extracts the task upon which the text was generated, tone and genre from the input text using one LLM and then generates a paraphrase based on this information.
    """
    def __init__(self, text_extractor:Paraphraser, text_generator:Paraphraser):
        """
        Initializes the TaskParaphraser model.
        :param text_extractor: A model or function to extract the task that was prompted, tone and genre from the input text.
        :param text_generator: A model or function to generate text based on the extracted task, tone and genre.
        """
        super().__init__(text_extractor=text_extractor, text_generator=text_generator)

    def paraphrase(self, text: str, prompt:Optional[str], n_responses:int=5, max_length:int=MAX_LENGTH, temperature:float=TEMPERATURE, response_schema: Optional[dict[str, Any]] = None,) -> List[str]:
        extractor_prompt = 'Act as the author of the text. From that perspective, infer your role or identity, the topic being addressed, and the purpose or instruction behind writing the text. Combine these elements into a concise task prompt that you would give to an LLM to reproduce the text. Respond ONLY with a JSON object in the following format: {"task":"<task>","tone":"<tone>","time_period":<time_period>,"language_register":<register>,"genre":"<genre>"}. Text to extract the intended task (prompt), tone, genre, time period and register from:'
        # FIXME: time_period and register breaks often
        # extractor_prompt = 'Act as the author of the text. From that perspective, infer your role or identity, the topic being addressed, and the purpose or instruction behind writing the text. Combine these elements into a concise task prompt that you would give to an LLM to reproduce the text. Respond ONLY with a JSON object in the following format: {"task":"<task>","tone":"<tone>","genre":"<genre>"}. Text to extract the intended task (prompt), tone, and genre from:'
        task, tone, genre, time_period, register = self._extract_bullet_points(text=text, prompt=extractor_prompt, temperature=temperature, key='task', response_schema=TaskSchema.model_json_schema())

        generator_prompt = "Do not use asterisks. Write a text of about {l} words with a {tone} tone, a {genre} genre, in the {register} register and in the {time_period} time period, covering the following task:\n{task}".format(l=len(text.split()), tone=tone, genre=genre, task=task, time_period=time_period, register=register)
        paraphrased_texts = self._generate_paraphrase_from_bullet_points(bullet_points=task, tone=tone, genre=genre, time_period=time_period, register=register, prompt=generator_prompt, n_responses=n_responses, temperature=temperature)

        return paraphrased_texts
    
class TopicParaphraser(BulletPointParaphraser):
    """
    A paraphrasing model that first extracts the topic of the text, tone and genre from the input text using one LLM and then generates a paraphrase based on this information.
    """
    def __init__(self, text_extractor:Paraphraser, text_generator:Paraphraser):
        """
        Initializes the TopicParaphraser model.
        :param text_extractor: A model or function to extract the topic, tone and genre from the input text.
        :param text_generator: A model or function to generate text based on the extracted topic, tone and genre.
        """
        super().__init__(text_extractor=text_extractor, text_generator=text_generator)

    def paraphrase(self, text: str, prompt:Optional[str], n_responses:int=5, max_length:int=MAX_LENGTH, temperature:float=TEMPERATURE) -> List[str]:
        extractor_prompt = 'Extract the topic, tone, time period, register and genre from the text. Respond ONLY with a JSON object in the following format: {"topic":"<topic>","tone":"<tone>","time_period":<time_period>,"language_register":<register>,"genre":"<genre>"}. Text to extract task, tone, genre, time period and register from:'
        # FIXME: time_period and register breaks often
        # extractor_prompt = 'Extract the topic, tone and genre from the text. Respond ONLY with a JSON object in the following format: {"topic":"<topic>","tone":"<tone>","genre":"<genre>"}. Text to extract task, tone, and genre from:'
        topic, tone, genre, time_period, register = self._extract_bullet_points(text=text, prompt=extractor_prompt, temperature=temperature, key='topic', response_schema=TopicSchema.model_json_schema())

        generator_prompt = "Do not use asterisks. Write a text of about {l} words with a {topic} topic, {tone} tone, a {genre} genre, in the {register} register and in the {time_period} time period.".format(l=len(text.split()), tone=tone, genre=genre, topic=topic, time_period=time_period, register=register)
        paraphrased_texts = self._generate_paraphrase_from_bullet_points(bullet_points=topic, tone=tone, genre=genre, time_period=time_period, register=register, prompt=generator_prompt, n_responses=n_responses, temperature=temperature)

        return paraphrased_texts
    
class TitleParaphraser(BulletPointParaphraser):
    """
    A paraphrasing model that first extracts the title of the text, tone and genre from the input text using one LLM and then generates a paraphrase based on this information.
    """
    def __init__(self, text_extractor:Paraphraser, text_generator:Paraphraser):
        """
        Initializes the TopicParaphraser model.
        :param text_extractor: A model or function to extract the title, tone and genre from the input text.
        :param text_generator: A model or function to generate text based on the extracted title, tone and genre.
        """
        super().__init__(text_extractor=text_extractor, text_generator=text_generator)

    def paraphrase(self, text: str, prompt:Optional[str], n_responses:int=5, max_length:int=MAX_LENGTH, temperature:float=TEMPERATURE, response_schema: Optional[dict[str, Any]] = None,) -> List[str]:
        extractor_prompt = 'Find a concise title for the text, extract the tone, time period, register and genre from the text. Respond ONLY with a JSON object in the following format: {"title":"<title>","tone":"<tone>","time_period":<time_period>,"language_register":<register>,"genre":"<genre>"}. Text to extract title, tone, genre, time period and register from:'
        # FIXME: time_period and register breaks often
        # extractor_prompt = 'Find a concise title for the text, extract the tone and genre from the text. Respond ONLY with a JSON object in the following format: {"title":"<title>","tone":"<tone>","genre":"<genre>"}. Text to extract title, tone, and genre from:'
        title, tone, genre, time_period, register = self._extract_bullet_points(text=text, prompt=extractor_prompt, temperature=temperature, key='title', response_schema=TitleSchema.model_json_schema())

        generator_prompt = "Do not use asterisks. Write a text of about {l} words with a {title} title, {tone} tone, a {genre} genre, in the {register} register and in the {time_period} time period.".format(l=len(text.split()), tone=tone, genre=genre, title=title, time_period=time_period, register=register)
        paraphrased_texts = self._generate_paraphrase_from_bullet_points(bullet_points=title, tone=tone, genre=genre, time_period=time_period, register=register, prompt=generator_prompt, n_responses=n_responses, temperature=temperature)

        return paraphrased_texts
    
    
class WMDReadyKeyedVectors:
    def __init__(self, keyed_vectors):
        self.model = keyed_vectors

    def __getitem__(self, key):
        return self.model[key]

    def __contains__(self, key):
        return key in self.model

    def keys(self):
        return self.model.key_to_index.keys()

class ParaphrasingEvaluator:
    def __init__(self, paraphrasers:dict, prompts:List[str], original_text:str, n_responses:int=3, max_length:int=MAX_LENGTH, temperature:float=TEMPERATURE):
        """
        Initializes the ParaphrasingEvaluator with the given paraphrasers and prompts.
        :param paraphrasers: A dictionary of paraphraser instances with their names as keys.
        :param prompts: A list of prompts to be used for paraphrasing.
        :param original_text: The original text to be paraphrased.
        :param n_responses: The number of paraphrases to generate for each paraphraser.
        :param max_length: The maximum length of the generated paraphrase.
        :param temperature: Controls the randomness of the output. Lower values make the output more deterministic.
        """
        assert isinstance(paraphrasers, dict) and all(isinstance(p, Paraphraser) for p in paraphrasers.values()), "paraphrasers must be a dictionary of Paraphraser instances."
        self.paraphrasers = paraphrasers
        assert isinstance(prompts, list) and all(isinstance(p, str) for p in prompts), "prompts must be a list of strings."
        self.prompts = prompts
        assert isinstance(original_text, str) and original_text.strip(), "original_text must be a non-empty string."
        self.original_text = _preprocess_text(original_text)
        assert isinstance(n_responses, int) and n_responses > 0, "n_responses must be a positive integer."
        self.n_responses = n_responses
        assert isinstance(max_length, int) and max_length > 0, "max_length must be a positive integer."
        self.max_length = max_length
        self.temperature = temperature

        self.rouge_score = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore") 
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # for cosine similarity
        # https://pypi.org/project/word-mover-distance/ Word Mover's Distance (WMD)
        self.pretr_word_model = WMDReadyKeyedVectors(gensim.downloader.load('glove-twitter-25'))
        self.wmd_model = model.WordEmbedding(model=self.pretr_word_model)


    def evaluate(self, save_to_disk:bool=True):
        """
        Evaluate the paraphrasers using BERTScore, BLEU and ROUGE metrics.
        :param save_to_disk: If True, saves the results to a CSV file.
        :return: A pandas DataFrame containing the evaluation results.
        """
        results = []
        references = [self.original_text] * self.n_responses
        original_split = self.original_text.split()

        bp_counts = defaultdict(int)

        for (name, paraphraser), prompt in tqdm(
            product(self.paraphrasers.items(), self.prompts),
            desc="Evaluating Paraphrasers",
            total=len(self.paraphrasers) * len(self.prompts)
        ):
            if isinstance(paraphraser, BulletPointParaphraser) or isinstance(paraphraser, TaskParaphraser) or isinstance(paraphraser, TopicParaphraser) or isinstance(paraphraser, TitleParaphraser): 
                if bp_counts[name] == 0:
                    prompt = None # paraphrasers above have specific prompt, which extracts bullet points/task/topic, tone and genre from the text
                    bp_counts[name] += 1
                else:
                    continue
            try:
                print(f"[DEBUG] Using paraphraser '{name}' with prompt '{prompt}'")
                paraphrases = [_preprocess_text(p) for p in paraphraser.paraphrase(
                    text=self.original_text,
                    n_responses=self.n_responses,
                    prompt=prompt
                )]
                if not paraphrases:
                    raise ValueError("Empty paraphrase list.")

            except Exception as e:
                print(f"[ERROR] Paraphraser '{name}' with prompt '{prompt}' failed: {e}")
                continue

            try:
                # input is list of strings, each string is a paraphrase/ reference
                bert_scores = self.bertscore.compute(predictions=paraphrases, references=references, model_type="distilbert-base-uncased")
                # rouge returns one value for all paraphrases, hence: list comprehension
                rouge_scores = [self.rouge_score.compute(predictions=[p], references=[self.original_text]) for p in paraphrases]
                               
                for i, paraphrase in enumerate(paraphrases):
                    results.append(self._build_result_row(
                        name, prompt, paraphrase, original_split, bert_scores, rouge_scores[i], i
                    ))

            except Exception as e:
                print(f"[ERROR] Scoring failed for '{name}' with prompt '{prompt}': {e}")
                continue

        df = pd.DataFrame(results)
        if save_to_disk:
            save_base_path = Path(__file__).resolve().parent.parent.parent / CONFIG.SAVE_PATH
            assert save_base_path.exists(), f"Savefig base path {save_base_path} does not exist."
            save_base_path = save_base_path / 'paraphrasing' 
            os.makedirs(save_base_path, exist_ok=True)
            save_path = save_base_path / f"paraphrasing_results_comparison_temp{self.temperature}_maxLength{self.max_length}.csv"
            df.to_csv(save_path, index=False, float_format="%.4f")
            print(f"Results saved to {save_path}")

        return df

    def _build_result_row(self, name:str, prompt:str, paraphrase:str, original_split:List[str], bert_scores:dict, rouge_score:dict, idx:int) -> dict:
        """
        Build a result row for the DataFrame.
        :param name: Name of the paraphraser.
        :param prompt: The prompt used for paraphrasing excluding the text to paraphrase and tailoring whitespaces, but including bulletpoints etc.
        :param paraphrase: One of the generated paraphrase.
        :param original_split: The original text split into tokens.
        :param bert_scores: BERTScore results.  
        :param rouge_score: ROUGE scores for the paraphrase.
        :param idx: Index of the paraphrase in the list of BERTScores.
        :return: A dictionary representing the result row.
        """
        # in [-1, 1] range, where 1 is identical, 0 is no similarity, -1 is opposite
        cos_sim = torch.cosine_similarity(self.sbert_model.encode(self.original_text, convert_to_tensor=True),
                                                         self.sbert_model.encode(paraphrase, convert_to_tensor=True), dim=0).item()
        res = {
            "model": name,
            "prompt": f"{prompt} <TEXT>",
            "parameters": {
                "n_responses": self.n_responses,
                "max_tokens": self.max_length,
                "temperature": self.temperature
            },
            "original_text": self.original_text,
            "paraphrased_text": paraphrase,
            # avoid division by zero using smoothing
            # bleu averages scores obtained from splits of paraphrase and (one of the) reference(s); here: only one reference (i.e. original text)
            "bleu_score": bleu_score.sentence_bleu( # in [0, 1]
                references=[original_split],
                hypothesis=paraphrase.split(),
                smoothing_function=bleu_score.SmoothingFunction().method1
            ),  # syntactic similarity metric
            # METEOR requires tokens as input
            "meteor_score": meteor_score.single_meteor_score(original_split, paraphrase.split()), # in [0, 1]
            "rouge1": rouge_score["rouge1"], # syntactic similarity metric # in [0, 1]
            "rouge2": rouge_score["rouge2"], # in [0, 1]
            "rougeL": rouge_score["rougeL"], # syntactic similarity metric # in [0, 1]
            "rougeLsum": rouge_score["rougeLsum"], # in [0, 1]
            # bertscore metrics in range [0, 1] cf. https://docs.kolena.com/metrics/bertscore/ (03.07.2025)
            "bertscore_precision": bert_scores["precision"][idx], # semantic similarity metric
            "bertscore_recall": bert_scores["recall"][idx], # semantic similarity metric
            "bertscore_f1": bert_scores["f1"][idx], # semantic similarity metric
            # normalized (in [0, 1] by using exp) word_mover_similarity
            "sbert_wms": np.exp(-self.wmd_model.wmdistance(list(map(str.lower, original_split)), paraphrase.lower().split())), # semantic similarity metric: exp(-distance) stable version of 1/distance
            # normalize: (cos - (-1)) / (1 - (-1)), so that it is in [0, 1] range
            "sbert_cos": (cos_sim + 1) / 2,   # semantic similarity metric
            # bertscore hashcode for the paraphrase
            "bertscore_hash": bert_scores["hashcode"],
        }
        semantic_sim_average = np.mean([res["bertscore_precision"], res["bertscore_recall"], res["bertscore_f1"], res["sbert_wms"], res["sbert_cos"]])
        res["sem_sim_avg"] = semantic_sim_average
        syntactic_sim_average = np.mean([res["bleu_score"], res["rouge1"], res["rougeL"]])
        res["syn_sim_avg"] = syntactic_sim_average
        res["gohsen_delta"] = semantic_sim_average - syntactic_sim_average
        return res

    
    def get_metric_names(self) -> List[str]:
        """
        Get the names of the metrics used in the evaluation.
        :return: A list of metric names.
        """
        return [
            "bleu_score", "meteor_score", "rouge1", "rouge2", "rougeL", "rougeLsum",
            "bertscore_precision", "bertscore_recall", "bertscore_f1",
            "sbert_wms", "sbert_cos",
            "sem_sim_avg", "syn_sim_avg", "gohsen_delta"
        ]
    
    def plot_models_metrics(self, df: pd.DataFrame, save_path: Optional[Path] = None, data_category:Optional[str]=None, group_by:Optional[str]='model'):
        """
        Plot the performance of models per metric.
        
        :param df: DataFrame containing the evaluation results.
        :param metric_name: The name of the metric to plot.
        :param save_path: Optional path to save the plot (without filename). If None, the plot will not be saved.
        :param data_category: Optional category of the data, used for the plot title.
        :param group_by: The column to group the data by (default is 'model'). Alternatives could be 'prompt'.
        :return: A list of matplotlib figures.
        """
        # Enforce fixed metric order
        all_labels = self.get_metric_names()
        labels = [metric for metric in all_labels if metric in df.columns]
        assert group_by in df.columns, f"Group by column '{group_by}' not found in DataFrame."
        grouped_mean = df.groupby(group_by)[labels].mean()
        grouped_std = df.groupby(group_by)[labels].std()


        # Compute angle of each axis
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        # Complete the loop
        angles += angles[:1]

        # Start plot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        for model_name in grouped_mean.index:
            mean_values = grouped_mean.loc[model_name].tolist()
            std_values = grouped_std.loc[model_name].tolist()

            # Close the loop
            mean_values += mean_values[:1]
            std_values += std_values[:1]

            lower = np.maximum(0, np.array(mean_values) - np.array(std_values))
            upper = np.minimum(1, np.array(mean_values) + np.array(std_values))

            ax.plot(angles, mean_values, label=model_name)
            ax.fill_between(angles, lower, upper, color=ax.get_lines()[-1].get_color(), alpha=0.2)


        # Add labels to axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        # Optional: Set value range
        ax.set_ylim(0, 1)

        # Add legend and title
        ax.legend(loc='lower left', bbox_to_anchor=(1.1, 0.7))
        title = f"Radar Chart: Paraphrasing Metric\non {data_category} text, grouped by {group_by}" if data_category else f"Radar Chart: Paraphrasing Metrics\ngrouped by {group_by}"
        plt.title(title)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path = save_path / f"paraphrasing_metrics_grouped_by_{group_by}_radar_chart_{timestamp}.png"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()
    
    def plot_metric_scatter(self, df: pd.DataFrame, save_path: Optional[Path] = None, data_category: Optional[str] = None, group_by:Optional[str]='model'):
        """
        Scatter plot of semantic similarity vs syntactic similarity per model.

        :param df: DataFrame with at least 'sem_sim_avg', 'syn_sim_avg', and 'model' columns.
        :param save_path: Optional path to save the plot. If None, the plot is not saved.
        :param data_category: Optional category of the data, used for the plot title.
        :param group_by: The column to group the data by (default is 'model'). Alternatives could be 'prompt'.
        :return: matplotlib Figure object.
        """
        required_cols = ['sem_sim_avg', 'syn_sim_avg', group_by]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

        # Calculate min and max with padding
        x_min, x_max = df['sem_sim_avg'].min(), df['sem_sim_avg'].max()
        y_min, y_max = df['syn_sim_avg'].min(), df['syn_sim_avg'].max()

        x_pad = (x_max - x_min) * 0.05 if (x_max - x_min) > 0 else 0.05
        y_pad = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.05

        x_lim = (max(0, x_min - x_pad), min(1, x_max + x_pad))
        y_lim = (max(0, y_min - y_pad), min(1, y_max + y_pad))

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        sns.scatterplot(
            data=df,
            x='sem_sim_avg',
            y='syn_sim_avg',
            hue=group_by,
            palette='tab10',
            alpha=0.7,
            s=100,
            edgecolor='k',
            ax=ax
        )

        ax.set_xlabel('Semantic Similarity (sem_sim_avg)')
        ax.set_ylabel('Syntactic Similarity (syn_sim_avg)')
        title = f"Semantic vs Syntactic Similarity\non {data_category} Texts, grouped by {group_by}" if data_category else f"Semantic vs Syntactic Similarity\ngrouped by {group_by}"
        ax.set_title(title)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(True)

        # Place legend outside the plot on the right
        ax.legend(
            title=group_by,
            loc='upper left',
            bbox_to_anchor=(1.03, 1),
            borderaxespad=0.,
            frameon=True
        )

        # Inset with full range
        inset_size = 0.25
        inset_ax = fig.add_axes([0.9, 0.1, inset_size, inset_size])

        sns.scatterplot(
            data=df,
            x='sem_sim_avg',
            y='syn_sim_avg',
            hue=group_by,
            palette='tab10',
            alpha=0.7,
            s=40,
            edgecolor='k',
            legend=False,  # No legend on inset
            ax=inset_ax
        )

        inset_ax.set_xlim(0, 1)
        inset_ax.set_ylim(0, 1)
        inset_ax.set_title('Full range')
        inset_ax.grid(True)
        inset_ax.set_xticks([0, 0.5, 1])
        inset_ax.set_yticks([0, 0.5, 1])
        inset_ax.tick_params(axis='both', which='major', labelsize=8)

        # Adjust layout to leave 25% room for legend/inset on the right
        #plt.tight_layout(rect=[0, 0, 0.75, 1])

        if save_path:
            save_path = Path(save_path)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            full_path = save_path / f"sem_syn_scatter_grouped_by_{group_by}_{timestamp}.png"
            plt.savefig(full_path, bbox_inches='tight')
            print(f"Plot saved to {full_path}")

        plt.show()
    
    def plot_metric_distributions(self, df: pd.DataFrame, save_path: Optional[Path] = None, data_category: Optional[str] = None, group_by:Optional[str]='model'):
        """
        Plot distribution of each metric per model in subplots.

        :param df: DataFrame containing metric scores and a 'model' column.
        :param save_path: Optional path to save the plot. If None, the plot is not saved.
        :param data_category: Optional string for plot title context.
        :param group_by: The column to group the data by (default is 'model'). Alternatives could be 'prompt'.
        :return: None
        """
        metric_names = [metric for metric in self.get_metric_names() if metric in df.columns]
        assert len(metric_names) > 0, "No valid metrics found in DataFrame."
        assert group_by in df.columns, f"Group by column '{group_by}' not found in DataFrame."

        n_metrics = len(metric_names)
        n_cols = 2
        n_rows = (n_metrics + 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, metric in enumerate(metric_names):
            ax = axes[i]
            if metric not in df.columns:
                continue

            # Find models with only one data point for this metric
            counts = df.groupby(group_by)[metric].count()

            # Models with multiple entries (for KDE)
            models_multi = counts[counts > 1].index
            # Models with single entry (for scatter)
            models_single = counts[counts == 1].index

            # Plot KDE for models with multiple points
            if len(models_multi) > 0:
                sns.kdeplot(
                    data=df[df[group_by].isin(models_multi)],
                    x=metric,
                    hue=group_by,
                    fill=True,
                    common_norm=False,
                    alpha=0.4,
                    ax=ax,
                    palette='tab10'
                )

            # Scatter for models with a single point
            for model in models_single:
                single_val = df[(df[group_by] == model)][metric].values[0]
                color = sns.color_palette('tab10')[list(df[group_by].unique()).index(model) % 10]
                ax.scatter(single_val, 0.1, label=model, color=color, s=50, edgecolor='k', zorder=5)

            ax.set_title(f"Distribution of {metric}")
            ax.set_xlim(0, 1)  # assuming similarity metrics in [0, 1]
            ax.set_xlabel(metric)
            ax.set_ylabel("Density")

        # Remove unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        title = f"Metric Distributions by Model\non {data_category} data, grouped by {group_by}" if data_category else f"Metric Distributions by Model\ngrouped by {group_by}"
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            save_path = Path(save_path)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            full_path = save_path / f"metric_distributions_grouped_by_{group_by}_{timestamp}.png"
            plt.savefig(full_path, bbox_inches='tight')
            print(f"Plot saved to {full_path}")

        plt.show()




if __name__ == "__main__":
    # models
    paraphrasers = {
                    # 'T5_ChatGPT': T5ChatGPTParaphraser(), 
                    # 'T5_Google_PAWS': T5GooglePAWSParaphraser(), 
                    # 'Blablador': BlabladorParaphraser(model_id="1 - Llama3 405 the best general model and big context size"), 
                    'Ollama': OllamaParaphraser(model_id="default:latest")
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
    n_responses = 3  # number of paraphrases to generate
    max_length = MAX_LENGTH  # Maximum length of the generated paraphrase
    temperature = TEMPERATURE  # Controls the randomness of the output. Lower values make the output more deterministic.


    # paraphrase_evaluator = ParaphrasingEvaluator(paraphrasers=paraphrasers, prompts=prompts, original_text=original_text, n_responses=n_responses, max_length=max_length, temperature=temperature)
    # paraphrase_evaluator.evaluate()
