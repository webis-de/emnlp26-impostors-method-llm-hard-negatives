from abc import ABC
from itertools import product
import json
import os
from pathlib import Path
import sys
from typing import Literal, get_args
import pandas as pd
import requests
from simpletransformers.t5 import T5Model
import sklearn
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import CONFIG

# cf. https://sdlaml.pages.jsc.fz-juelich.de/ai/guides/blablador_api_access/ (15.06.2025)
ModelName = Literal[
    "1 - Llama3 405 the best general model and big context size",
    "1 - Ministral 8b - the fast model",
    "1 - Teuken-7B-instruct-research-v0.4 - The OpenGPT-X model",
    "10 Mistral-Nemo-Instruct-2407 - Our fast-experimental - with a large context size",
    "2 - QwenLong L1 32B - A long context reasoning model from 28.05.2025", # specially trained for reasoning
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
MAX_LENGTH = 256  # Maximum length of the generated paraphrase

class Paraphraser(ABC):
    """
    Abstract base class for paraphrasing models.
    """

    def paraphrase(self, text: str, prompt: str, n_responses:int=5, max_length:int=MAX_LENGTH) -> str:
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

    def paraphrase(self, text: str, prompt: str, num_beams=5, num_beam_groups=5, n_responses:int=5, repetition_penalty=10.0, diversity_penalty=3.0, no_repeat_ngram_size=2, temperature:float=TEMPERATURE, max_length:int=MAX_LENGTH) -> str:
        input_ids = self.tokenizer(f'{prompt.strip()} {text}', return_tensors="pt", padding="longest", max_length=max_length, truncation=True).input_ids.to(self.device)
        
        outputs = self.model.generate(
            input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
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

    def paraphrase(self, text: str, prompt:str, n_responses:int=5, max_length:int=MAX_LENGTH) -> str:
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


    def paraphrase(self, text: str, prompt:str, verbose:bool=False, max_length:int=MAX_LENGTH, temperature:float=TEMPERATURE, n_responses:int=5) -> str:
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
            paraphrased_texts = [choice["text"] for choice in response.json()["choices"]] if n_responses > 1 else response.json()["choices"][0]["text"]
            if verbose:
                print("Paraphrased text(s):\n", paraphrased_texts)
            return paraphrased_texts
        else:
            print("Error:", response.status_code, response.text)
    
    
    


if __name__ == "__main__":
    df = pd.DataFrame(columns=["model", "prompt", "parameters", "original_text", "paraphrased_text"])
    paraphrasers = {'T5_ChatGPT': T5ChatGPTParaphraser(), 'T5_Google_PAWS': T5GooglePAWSParaphraser()}
    paraphrasers.update({f'Blablador_{name}': BlabladorParaphraser(model_id=name) for name in list(get_args(ModelName))})

    text = "The quick brown fox jumps over the lazy dog."
    n_responses = 1  # Number of paraphrases to generate
    prompts = ["Paraphrase the following text and output only the paraphrased version:", 
            #   "First, extract bullet points capturing the main ideas, then create a text based on these bullet points. Only output the final text (i.e. do not output the bullet points or any additional chain of thoughts):",
            #   "Paraphrase the sentence by first identifying the main subject, verb, and object. Then find synonyms for each and construct a new sentence. Only output the final paraphrased sentence.",
            #   "Paraphrase the sentence using the same tone as the original with approximately the same number of words:",
              "Paraphrase this sentence. Do not change the meaning, but use different words and structure. Output only the paraphrased sentence:"]
    for (name, paraphraser), prompt_text in tqdm(product(paraphrasers.items(), prompts), desc="Paraphrasing with all model-prompt combinations", total=len(paraphrasers) * len(prompts)):
        paraphrased_text = paraphraser.paraphrase(text=text, n_responses=n_responses, prompt=prompt_text)
       
        df = pd.concat([df, pd.DataFrame([{
            "model": name,
            "prompt": f"{prompt_text} <TEXT>",
            "parameters": {
                "n_responses": n_responses,
                "max_tokens": MAX_LENGTH,
                "temperature": TEMPERATURE
            },
            "original_text": text,
            "paraphrased_text": paraphrased_text
        }])], ignore_index=True)
    
    # Save the results to a CSV file
    save_base_path = Path(__file__).resolve().parent.parent.parent / CONFIG.SAVE_PATH
    assert save_base_path.exists(), f"Savefig base path {save_base_path} does not exist."
    save_base_path = save_base_path / 'paraphrasing' 
    os.makedirs(save_base_path, exist_ok=True)
    save_path = save_base_path / "paraphrasing_results_comparison.csv"
    df.to_csv(save_path, index=False)





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