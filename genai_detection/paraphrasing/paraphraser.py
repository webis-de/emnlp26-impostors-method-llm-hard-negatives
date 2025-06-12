from abc import ABC
from simpletransformers.t5 import T5Model
import sklearn
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Paraphraser(ABC):
    """
    Abstract base class for paraphrasing models.
    """

    def paraphrase(self, text: str) -> str:
        """
        Generate a paraphrase of the input text.

        :param text: The input text to be paraphrased.
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

    def paraphrase(self, text: str, num_beams=5, num_beam_groups=5, num_return_sequences=5, repetition_penalty=10.0, diversity_penalty=3.0, no_repeat_ngram_size=2, temperature=0.7, max_length=128) -> str:
        input_ids = self.tokenizer(f'paraphrase: {text}', return_tensors="pt", padding="longest", max_length=max_length, truncation=True).input_ids.to(self.device)
        
        outputs = self.model.generate(
            input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
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

    def paraphrase(self, text: str) -> str:
        # TODO: no duplication penalty, and thus, there are duplicates in the output
        encoding = self.tokenizer.encode_plus(f'paraphrase: {text} </s>', padding="max_length", return_tensors="pt")
        
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            num_return_sequences=5,
        )

        res = []
        for output in outputs:
            line = self.tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            res.append(line)

        return res
    
    


if __name__ == "__main__":
    # paraphraser = T5ChatGPTParaphraser()
    paraphraser = T5GooglePAWSParaphraser()
    text = "The quick brown fox jumps over the lazy dog."
    paraphrased_text = paraphraser.paraphrase(text)
    print(f"Original: {text}")
    print(f"Paraphrased: {paraphrased_text}")
    
    # Batch paraphrasing
    texts = ["The sky is blue.", "I love programming.", "Artificial intelligence is fascinating."]
    paraphrased_batch = paraphraser.paraphrase_batch(texts)
    for original, paraphrased in zip(texts, paraphrased_batch):
        print(f"Original: {original} | Paraphrased: {paraphrased}")