from typing import DefaultDict, Iterable, List
import typing as t
import numpy as np
import torch
from genai_detection.detectors.detector_base import DetectorBase
from evaluate import load

class PerplexityDetector(DetectorBase):
    def __init__(self):
        super().__init__()
        self.perplexity = load("perplexity", module_type="metric")


    def _get_score_impl(self, text: Iterable[str]) -> t.Union[torch.Tensor, np.ndarray, t.Iterable[float]]:
        """
        Calculate the perplexity score for the given text.
        Perplexity is calculated on token level (since the denominator is the number of tokens):
        perplexity = e**(sum(losses) / num_tokenized_tokens)
        where losses are the negative log-likelihood losses, which are the negative log probabilities of the tokens
        given the previous tokens in the text in the original formula.

        :param text: input text to score (no batch!), if list we assume it is tokenized.
        :return: Average (over token) perplexity score for the input text, in the form of a dictionary with model names as keys

        Reference: https://huggingface.co/spaces/evaluate-metric/perplexity (09.07.2025)
        """
        models = ["gpt2", "EleutherAI/gpt-neo-125M"] # very big: "ctrl", "llama", "mistral", "deepseek_v3"
        perplexity_scores = DefaultDict(dict)
        if type(text) is str:
            text = [text]
        for model in models:
            # in [0, inf), lower is better, cf. https://huggingface.co/spaces/evaluate-metric/perplexity (09.07.2025)
            perplexity_scores[model] = [round(self.perplexity.compute(predictions=t, model_id=model)["mean_perplexity"], 2) for t in text]

        return perplexity_scores

    def get_score(
        self, text: t.Union[str, t.Iterable[str]], normalize: bool = False
    ) -> t.Union[np.float32, np.ndarray, np.nan]:
        """
        Return scores indicating the probability of the input text(s) being machine-generated.

        Scores can be normalized to represent valid probability values with higher values meaning higher confidence
        in the texts being machine-generated. However, scores between different detector implementations are not
        necessarily comparable. If ``normalize`` is ``False`` (the default), unnormalized raw scores are returned
        instead. Interpretation of these raw scores is entirely implementation-dependent.

        :param text: input text or batch of input texts (as string or list of strings)
        :param normalize: normalize scores to represent probabilities in the range [0, 1]
        :return: score indicating whether the input text being machine-generated
        """
        text = [text] if isinstance(text, str) else text
        scores = [self._get_score_impl(t) for t in text]    # list of dicts
        
        if normalize and scores is not NotImplemented:
            scores_matrix = np.array([list(s.values()) for s in scores])
            scores_matrix = [self._normalize_scores(scores_matrix[i] for i in range(len(scores_matrix)))]

            scores = [{model_name: scores_matrix[j][i] for i, model_name in enumerate(list(scores[0].keys()))} for j in range(len(scores_matrix))]
        return scores
    
    def get_prediction(self, text: Iterable[str], threshold=0.5) -> List[bool]:
        """
        Predict if the input text(s) were written by any of the LLMs.

        :param text: input text or batch of input texts
        :return: boolean classifications of whether inputs are likely machine-generated
        """
        # perplexity on token level
        scores = self.get_score(text)  # [Dict[str, List[float]]]; upper elements are texts, containing dicts with their scores for different models
        model_names = list(scores[0].keys())
        model_names.append("unclear")  # Always include 'unclear' at the end

        # Build score matrix (num_texts x num_models), excluding 'unclear'
        score_matrix = np.array([list(s.values()) for s in scores])

        predictions = []
        for row in score_matrix:
            min_score = np.min(row)
            print(f"Max score: {min_score}, Threshold: {threshold}")
            one_hot = [0] * len(model_names)
            # TODO: find threshold: https://thegradient.pub/understanding-evaluation-metrics-for-language-models/
            if min_score < threshold:
                one_hot[-1] = 1  # Set 'unclear'
            else:
                best_index = int(np.argmax(row))
                one_hot[best_index] = 1
            predictions.append(one_hot)
        # first text access via predictions[0]
        return predictions, model_names
    

if __name__ == "__main__":
    detector = PerplexityDetector()
    text = "This is a test sentence."
    scores = detector.get_score(text)
    print("Scores:", scores)
    predictions, model_names = detector.get_prediction(text)
    print("Predictions:", predictions)
    print("Model Names:", model_names)