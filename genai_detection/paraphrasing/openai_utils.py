import tiktoken

INPUT_PRICE_PER_ONE_M_TOKENS_USD = 0.250  # gpt-5 mini pricing as of October 2025
OUTPUT_PRICE_PER_ONE_M_TOKENS_USD = 2.000  # gpt-5 mini pricing as of October 2025

class OpenaiCostEstimator:

    def __init__(self, openai_model_name:str="gpt-4o"):
        self.openai_model_name = openai_model_name

    def num_tokens_from_string(self, string: str, openai_model_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(openai_model_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def compute_cost(self, prompt:str, text:str):


        n_tokens_essay = self.num_tokens_from_string(string=text, openai_model_name=self.openai_model_name)
        n_tokens_prompt = self.num_tokens_from_string(string=prompt, openai_model_name=self.openai_model_name)
        total_price_usd = INPUT_PRICE_PER_ONE_M_TOKENS_USD * (
                (n_tokens_prompt + n_tokens_essay) / 1_000_000) + OUTPUT_PRICE_PER_ONE_M_TOKENS_USD * (
                                  n_tokens_essay * 3 / 1_000_000)

        costs = {
            "n_words_text": len(text.split()),
            "n_words_prompt": len(prompt.split()),
            "n_tokens_text": n_tokens_essay,
            "n_tokens_prompt": n_tokens_prompt,
            "total_price_usd": total_price_usd,
            "total_price_euro": total_price_usd * 0.86,
        }
        return costs
