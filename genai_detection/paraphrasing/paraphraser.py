from abc import ABC
from collections import defaultdict
from itertools import product
import json
import os
from pathlib import Path
import re
import sys
from typing import List, Literal, Optional, get_args
import pandas as pd
import requests
from simpletransformers.t5 import T5Model
import sklearn
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from openai import OpenAI
nltk.download('wordnet')    # necessary for METEOR score
from nltk.translate import bleu_score, meteor_score
import evaluate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import CONFIG

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

class Paraphraser(ABC):
    """
    Abstract base class for paraphrasing models.
    """

    def paraphrase(self, text: str, prompt: str, n_responses:int=5, max_length:int=MAX_LENGTH) -> List[str]:
        """
        Generate a paraphrase of the input text.

        :param text: The input text to be paraphrased.
        :param prompt: The prompt to be used for paraphrasing.
        :param n_responses: The number of paraphrases to generate.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :return: A paraphrased version of the input text.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def paraphrase_batch(self, texts: list[str]) -> list[str]:
        """
        Generate paraphrases for a batch of input texts.

        :param texts: A list of input texts to be paraphrased.
        :return: A list of paraphrased versions of the input texts.
        """
        return [self.paraphrase(text) for text in texts]
    

class T5ChatGPTParaphraser(Paraphraser):
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

    def paraphrase(self, text: str, prompt: str, num_beams=5, num_beam_groups=5, n_responses:int=5, repetition_penalty=10.0, diversity_penalty=3.0, no_repeat_ngram_size=2, max_length:int=MAX_LENGTH) -> List[str]:
        input_ids = self.tokenizer(f'{prompt.strip()} {text}', return_tensors="pt", padding="longest", max_length=max_length, truncation=True).input_ids.to(self.device)
        
        outputs = self.model.generate(
            input_ids, repetition_penalty=repetition_penalty,
            num_return_sequences=n_responses, no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams, num_beam_groups=num_beam_groups,
            max_length=max_length, diversity_penalty=diversity_penalty
        )

        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res
    

class T5GooglePAWSParaphraser(Paraphraser):
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

    def paraphrase(self, text: str, prompt:str, n_responses:int=5, max_length:int=MAX_LENGTH) -> List[str]:
        # TODO: no duplication penalty, and thus, there are duplicates in the output
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
        prompt: str,
        max_length: int = MAX_LENGTH,
        temperature: float = TEMPERATURE,
        n_responses: int = 1,
    ) -> List[str]:
        """
        Generate paraphrased versions of the input text.
        """
        # print('prompt:', prompt)
        responses = []
        for i in range(n_responses):    # directly using parameter n does not return n responses, but only one response
            response = self.client.chat.completions.create(
                model = "default:latest",
                messages = [{"role": "user", "content": f"{prompt.strip()} {text}"}],
                n = 1,
                max_tokens = max_length,
                temperature = temperature,
            )
            resp = response.choices[0].message.content
            resp = re.sub("'", ' ', resp)  # replace single quotes with double quotes
            resp = re.sub(r'\s+', ' ', resp)  # remove excessive whitespaces
            
            try: 
                data = json.loads(resp)
                responses.append(data) 
            except json.JSONDecodeError:
                responses.append(response.choices[0].message.content)

        for i, resp in enumerate(responses):
            print(f"[DEBUG] Response {i+1}: {resp.keys()}")
    
        return responses



class BlabladorParaphraser(Paraphraser):
    """
    Blablador paraphrasing model hosted by Jülich/ Helmholtz AI.
    """

    def __init__(self, model_id: ModelName="1 - Llama3 405 the best general model and big context size"):
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
                print("Available models:")
                for model in models:
                    print(model)
            return models
        else:
            raise Exception(f"Error fetching models: {response.status_code} - {response.text}")


    def paraphrase(self, text: str, prompt:str, verbose:bool=False, max_length:int=MAX_LENGTH, temperature:float=TEMPERATURE, n_responses:int=5) -> List[str]:
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

# FIXME:
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

    def _extract_dict(self, text: str) -> dict:
        # Match a JSON object: starts with `{` and ends with `}`
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found.")

        json_str = match.group(0)
        components_tmp = json_str.split(':')
        components = {}
        for i in range(1,len(components_tmp)):
            key = components_tmp[i-1].split(',')[-1].strip().strip('{').strip('}').strip('\n').replace("’", '"').replace("\"", '"') # omit escaping backslashes
            key = self.add_tailoring_quotes(key)
            if key == '"':
                continue  # skip empty keys
            print(f"[DEBUG] Key: {key}")
            value = ','.join(components_tmp[i].split(',')[:-1]).strip().strip('{').strip('}').strip('\n').strip("'").replace("’", '"').replace('""', '"')#.replace('\n', ' ')
            value = re.sub(r'\s+', ' ', value)
            if value.startswith('['):
                values = value.strip('[]').strip().split('\n')
                for j in range(len(values)):
                    values[j] = self.add_tailoring_quotes(values[j])
            else:
                values = self.add_tailoring_quotes(value)
            
            print(f"[DEBUG] Value: {values}")
            components[key] = values
        return components

    def add_tailoring_quotes(self, value):
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            return value  # already quoted properly
        return f'"{value.strip("'").strip('"').strip()}"'

    def _extract_bullet_points(self, text: str, prompt: Optional[str]) -> tuple[List[str], str, str]:
        """
        Extract bullet points, tone, and genre from the input text.

        :param text: The input text.
        :param prompt: Optional custom prompt.
        :return: Tuple of bullet points, tone, and genre.
        """
        assert self.text_extractor is not None, "Text extractor must be provided."
        if prompt is None:
            prompt = "Summarize the following text in five to six short bullet points and give an overall description of the genre and tone of the text. Return only one JSON object with keys 'bullet_points' (values is list of bullet points where each bullet point is an element of the list), 'tone' (value is text tone), and 'genre' (value is text genre). Return only one valid JSON. Do not escape all quotes inside strings. Use double quotes only. Do not return any chain of thought or instructions.\n\nText:\n"
            # prompt = "Summarize the following text in five to six short bullet points and give an overall description of the genre and tone of the text." + text
    
        res = self.text_extractor.paraphrase(text=text, prompt=prompt, n_responses=1, max_length=MAX_LENGTH)
        
        parsed_dict = self._extract_dict(res[0])    
        print(f"[DEBUG] Extracted bullet points, tone and genre: {parsed_dict}")
        bp, tone, genre = parsed_dict.get('bullet_points', []), parsed_dict.get('tone', ''), parsed_dict.get('genre', '')
        return bp, tone, genre
    
    def _generate_paraphrase_from_bullet_points(self, bullet_points, tone, genre, prompt: Optional[str]) -> tuple[List[str], str, str]:
        """
        Generate a paraphrase using the extracted bullet points, tone, and genre.

        :param bullet_points: List of key points.
        :param tone: Tone of the original text.
        :param genre: Genre of the original text.
        :param prompt: Optional prompt to guide generation.
        :param n_responses: Number of paraphrased versions to generate.
        :return: List of paraphrased texts.
        """
        assert self.text_extractor is not None, "Text extractor must be provided."
        if prompt is None:
            prompt = (
                f"Write a text with a {tone} tone and {genre} genre, covering the following points:\n"
                + "\n".join(f"- {bp}" for bp in bullet_points)
            )
            # PAN24:
            #f"Write a text of about {len(text)} words which covers the following items:"
        return self.text_generator.paraphrase(text='', prompt=prompt, n_responses=1, max_length=MAX_LENGTH)


    def paraphrase(self, text: str, prompt:Optional[str], n_responses:int=5, max_length:int=MAX_LENGTH) -> List[str]:
        """
        Generate a paraphrase of the input text by first extracting bullet points.
        
        :param text: The input text to be paraphrased.
        :param prompt: The prompt to be used for paraphrasing.
        :param n_responses: The number of paraphrases to generate.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :return: A paraphrased version of the input text.
        """
        bullet_points, tone, genre = self._extract_bullet_points(text, prompt)
        paraphrased_texts = self._generate_paraphrase_from_bullet_points(bullet_points, tone, genre, prompt, n_responses)

        # Placeholder implementation
        return paraphrased_texts
    
    

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
        self.original_text = original_text
        assert isinstance(n_responses, int) and n_responses > 0, "n_responses must be a positive integer."
        self.n_responses = n_responses
        assert isinstance(max_length, int) and max_length > 0, "max_length must be a positive integer."
        self.max_length = max_length
        self.temperature = temperature

        self.rouge_score = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore") 


    def evaluate(self, save_to_disk:bool=True):
        """
        Evaluate the paraphrasers using BERTScore, BLEU and ROUGE metrics.
        :param save_to_disk: If True, saves the results to a CSV file.
        :return: A pandas DataFrame containing the evaluation results.
        """
        results = []
        references = [self.original_text] * self.n_responses
        original_split = self.original_text.split()

        for (name, paraphraser), prompt in tqdm(
            product(self.paraphrasers.items(), self.prompts),
            desc="Evaluating Paraphrasers",
            total=len(self.paraphrasers) * len(self.prompts)
        ):
            try:
                paraphrases = paraphraser.paraphrase(
                    text=self.original_text,
                    n_responses=self.n_responses,
                    prompt=prompt
                )
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
        return {
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
            "bleu_score": bleu_score.sentence_bleu(
                references=[original_split],
                hypothesis=paraphrase.split(),
                smoothing_function=bleu_score.SmoothingFunction().method1
            ),
            # METEOR requires tokens as input
            "meteor_score": meteor_score.single_meteor_score(original_split, paraphrase.split()),
            "rouge1": rouge_score["rouge1"],
            "rouge2": rouge_score["rouge2"],
            "rougeL": rouge_score["rougeL"],
            "rougeLsum": rouge_score["rougeLsum"],
            "bertscore_precision": bert_scores["precision"][idx],
            "bertscore_recall": bert_scores["recall"][idx],
            "bertscore_f1": bert_scores["f1"][idx],
            "bertscore_hash": bert_scores["hashcode"],
        }


if __name__ == "__main__":
    # models
    # paraphrasers = {'T5_ChatGPT': T5ChatGPTParaphraser(), 'T5_Google_PAWS': T5GooglePAWSParaphraser()}
    # paraphrasers.update({f'Blablador_{name}': BlabladorParaphraser(model_id=name) for name in list(get_args(ModelName))})

    # prompts = [
    #         "Paraphrase the following text and output only the paraphrased version:", 
    #         "First, extract bullet points capturing the main ideas, then create a text based on these bullet points. Only output the final text (i.e. do not output the bullet points or any additional chain of thoughts):",
    #         "Paraphrase the sentence by first identifying the main subject, verb, and object. Then find synonyms for each and construct a new sentence. Only output the final paraphrased sentence.",
    #         "Paraphrase the sentence using the same tone as the original with approximately the same number of words:",
    #         "Paraphrase this sentence. Do not change the meaning, but use different words and structure. Output only the paraphrased sentence:"
    #         ]
    
    original_text = "The quick brown fox jumps over the lazy dog."
    ollama_paraphraser = OllamaParaphraser(model_id="default:latest")
    prompt = 'Paraphrase the following text. Respond ONLY with a JSON object in the following format: {"genre":"<genre>","tone":"<tone>","paraphrase":"<paraphrased version of the text>"}. Text to paraphrase:'
    print("Ollama Paraphrasing Example:", ollama_paraphraser.paraphrase(text=original_text, prompt=prompt, n_responses=3, max_length=MAX_LENGTH))

    # FIXME
    # original_text = 'Vice President JD Vance, in his first public comments since President Donald Trump authorized US strikes on Iranian nuclear sites, emphasized that the US is “not at war” with Iran as he laid out the president’s decision-making process. “We’re not at war with Iran. We’re at war with Iran’s nuclear program,” Vance said in an interview with NBC’s “Meet the Press with Kristen Welker,” calling the strikes a “testament to the power of the American military.”Asked what intelligence led to the decision, Vance described a “narrow window of opportunity.” “We had a narrow window of opportunity. We might not have been able to carry out this attack six months down the road. … We have a narrow window in which we can set that program back a very long time. It would have been irresponsible, I think, for the president not to take the action that he did,” he said. Vance added, “Of course we trust our intelligence community, but we also trust our instincts. … The Iranians stopped negotiating in good faith – that was the real catalyst.” The vice president also suggested that Trump arrived at the conclusion to authorize the strikes after issuing what he described as “private ultimatums” to Iran. He declined to detail those ultimatums. CNN has previously reported that Trump offered Iran a 60-day window in April to negotiate, asking Israeli Prime Minister Benjamin Netanyahu to hold off on striking the country to allow those talks to progress.'
    # n_reponses = 3  # number of paraphrases to generate
    # max_length = MAX_LENGTH  # Maximum length of the generated paraphrase
    # temperature = TEMPERATURE  # Controls the randomness of the output. Lower values make the output more deterministic.

    # bullet_point_paraphraser = BulletPointParaphraser(text_extractor=BlabladorParaphraser(model_id="1 - Llama3 405 the best general model and big context size",), text_generator=BlabladorParaphraser(model_id="1 - Ministral 8b - the fast model"))
    # print("Bullet Point Paraphrasing Example:", bullet_point_paraphraser.paraphrase(
    #     text=original_text, prompt=None, n_responses=1, max_length=max_length))

    # paraphrase_evaluator = ParaphrasingEvaluator(paraphrasers=paraphrasers, prompts=prompts, original_text=original_text, n_responses=n_reponses, max_length=max_length, temperature=temperature)
    # paraphrase_evaluator.evaluate()

    # paraphraser = T5ChatGPTParaphraser()
    # paraphraser = T5GooglePAWSParaphraser()
    # paraphraser = BlabladorParaphraser()
    # paraphraser._get_available_models()
    # text = "The quick brown fox jumps over the lazy dog."
    # paraphrased_text = paraphraser.paraphrase(text, n_responses=1)
    # print(f"Original: {text}")
    # print(f"Paraphrased: {paraphrased_text}")
    
    # # Batch paraphrasing
    # texts = ["The sky is blue.", "I love programming.", "Artificial intelligence is fascinating."]
    # paraphrased_batch = paraphraser.paraphrase_batch(texts)
    # for original, paraphrased in zip(texts, paraphrased_batch):
    #     print(f"Original: {original} | Paraphrased: {paraphrased}")