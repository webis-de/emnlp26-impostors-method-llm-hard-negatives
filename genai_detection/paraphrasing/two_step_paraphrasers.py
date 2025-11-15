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

from typing import Optional, List, Any

import dspy

from genai_detection.config import CONFIG
from genai_detection.paraphrasing.paraphraser import Paraphraser

# Field descriptions dictionary
FIELD_DESCRIPTIONS = {
    "heading": "A brief header or heading suitable for the text, capturing its main subject or theme in a clear and concise way.",
    "genre": "The literary or content genre of the text (e.g., essay, poem, news article, academic paper, novel, speech, letter, religious text, etc.).",
    "tone": "The emotional or stylistic tone (e.g., formal, humorous, persuasive, neutral, melancholic, satirical, didactic).",
    "register_style": "The social level of language used in the text (e.g., casual, colloquial, academic, bureaucratic, poetic, technical).",
    "century": "The century in which the text was likely written (e.g., '18th century', '20th century', '21st century'). Use language, style, and context clues to infer.",
    "audience": "The intended audience or readership (e.g., general public, scholars, children, political leaders, students, religious followers).",
    "author": "A concise author profile including likely traits such as gender (if implied), profession, education level, nationality, or perspective (e.g., 'a 19th-century British poet', 'a modern journalist', 'an academic researcher'), using idiosyncrasies in the text (e.g., spelling or grammar errors) to guide the inference.",
    "bulletpoints": "A list of concise bullet points summarizing the main ideas or arguments of the text, written in plain language, focusing on meaning rather than wording.",
    "length": "The approximate target length of the generated text, in number of words.",
}


class ExtractInfo(dspy.Signature):
    """Extract structured literary and stylistic information from text."""

    text: str = dspy.InputField(desc="The full input text to analyze.")
    heading: str = dspy.OutputField(desc=FIELD_DESCRIPTIONS["heading"])
    genre: str = dspy.OutputField(desc=FIELD_DESCRIPTIONS["genre"])
    tone: str = dspy.OutputField(desc=FIELD_DESCRIPTIONS["tone"])
    register_style: str = dspy.OutputField(desc=FIELD_DESCRIPTIONS["register_style"])
    century: str = dspy.OutputField(desc=FIELD_DESCRIPTIONS["century"])
    audience: str = dspy.OutputField(desc=FIELD_DESCRIPTIONS["audience"])
    author: str = dspy.OutputField(desc=FIELD_DESCRIPTIONS["author"])
    bulletpoints: list[str] = dspy.OutputField(desc=FIELD_DESCRIPTIONS["bulletpoints"])

# We found that instructing the LLM to generate more than one paraphrase at a time reduces the length of the paraphrase drastically and therefore, omit n_paraphrases
class GenerateText(dspy.Signature):
    """Generate one complete alternative version of the text, unifying all bullet points into a coherent composition and reflecting the defined style and structure."""

    heading: str = dspy.InputField(desc=FIELD_DESCRIPTIONS["heading"])
    genre: str = dspy.InputField(desc=FIELD_DESCRIPTIONS["genre"])
    tone: str = dspy.InputField(desc=FIELD_DESCRIPTIONS["tone"])
    register_style: str = dspy.InputField(desc=FIELD_DESCRIPTIONS["register_style"])
    century: str = dspy.InputField(desc=FIELD_DESCRIPTIONS["century"])
    audience: str = dspy.InputField(desc=FIELD_DESCRIPTIONS["audience"])
    author: str = dspy.InputField(desc=FIELD_DESCRIPTIONS["author"])
    bulletpoints: list[str] = dspy.InputField(desc=FIELD_DESCRIPTIONS["bulletpoints"])
    length: int = dspy.InputField(desc=FIELD_DESCRIPTIONS["length"])
    text: str = dspy.OutputField(
        desc=(
            "A single, complete paraphrased text written in the style defined by the provided parameters. "
            "The text must integrate **all** bullet points into one coherent composition, follow the specified "
            "**heading**, **genre**, **tone**, **register**, **century**, **audience**, and **author profile**, and "
            "approximately match the requested word length."
        )
    )

class TwoStepParaphraser(Paraphraser):
    """
    This paraphrasing model first extracts information from the input text using one LLM and then generates a paraphrase based on this information using a second LLM.
    LLM communication via DSPy (i.e., wrapper for OpenAI's GPT models).
    DSPy (https://dspy.ai) is a high-level framework for building LLM applications.
    You need to specify a signature (i.e., input and output fields), docstring with short, declarative LM instruction and a module for DSPy to work.
    A module is the way the LLM is invoked.
    Find more information at https://arxiv.org/abs/2310.03714 (31.10.2025).
    """

    def __init__(
        self,
        model_id: str = CONFIG.OPENAI_MODEL,
        temperature: float = CONFIG.TEMPERATURE,
        n_paraphrases: int = 50,
    ):
        """
        Initializes the OpenAI paraphraser model using DSPy.
        :param model_id: Name of the OpenAI model to use.
        :param temperature: Temperature for the model. Needs to be 1.0 for reasoning models like GPT-5.
        :param n_paraphrases: Number of paraphrases to generate.
        """
        super().__init__(n_paraphrases, model_id)
        is_reasoning_model = any(name in model_id.lower() for name in ["gpt-5"])

        self.lm = dspy.LM(
            model_id,
            api_base=CONFIG.OPENAI_URL,
            api_key=CONFIG.OPENAI_API_KEY,
            model_type="chat",  # better for structured output such as extracted information
            cache=False, # to avoid reusing the same response
            temperature=1.0 if is_reasoning_model else temperature, # TODO: change to non-reasoning for less deterministic results?
            max_tokens=16000 if is_reasoning_model else CONFIG.MAX_LENGTH,
        )
        self.model_id = model_id
        dspy.configure(lm=self.lm)
        self.n_paraphrases = n_paraphrases

    def set_n_paraphrases(self, n_paraphrases: int):
        """
        Sets the number of paraphrases to generate.
        :param n_paraphrases: Number of paraphrases to generate.
        :return: -
        """
        self.n_paraphrases = n_paraphrases

    def generate_multiple_paraphrase_based_on_extracted_information(
        self, extracted_info: dict[str, Any], verbose: bool = True
    ) -> tuple[List[str], float]:
        paraphrases, total_cost = [], []
        for i in range(self.n_paraphrases):
            try:
                new_impostor, cost = (
                    self.generate_one_paraphrase_based_on_extracted_information(
                        verbose=verbose, extracted_info=extracted_info
                    )
                )
                paraphrases.append(new_impostor)
                total_cost += cost
            except Exception as e:
                print(e)
        return paraphrases, total_cost

    def generate_one_paraphrase_based_on_extracted_information(
        self, extracted_info: dict[str, Any], verbose: bool = True
    ) -> tuple[str, float]:
        """
        Generate a paraphrase based on the extracted information.
        This method will generate one paraphrase.
        :param verbose: Whether to print debug information.
        :param extracted_info: Dictionary containing the extracted information.
        :return: A tuple containing the generated paraphrase (as one string) and the costs after generation.
        """
        # Module: Predict, signature: GenerateText
        generator = dspy.Predict(GenerateText)

        if verbose:
            cfg = generator.get_config()
            print("Config:", cfg)
            sig = generator.signature
            print("Signature:", sig)
            print("generating paraphrase with extracted info", extracted_info)
        try:
            extracted_keys = set(extracted_info.keys())
            field_keys = set(FIELD_DESCRIPTIONS.keys())

            # Remove 'text' from both sets
            extracted_keys.discard("text")
            field_keys.discard("text")

            # Assert equality with a detailed message
            assert extracted_keys == field_keys, (
                f"Keys differ besides 'text'. "
                f"Missing in extracted_info: {field_keys - extracted_keys}, "
                f"Extra in extracted_info: {extracted_keys - field_keys}"
            )

            paraphrase = generator(**extracted_info)
            if verbose:
                print(f"generated paraphrase:", paraphrase)
            cost_after_generator = sum(
                x["cost"] for x in self.lm.history if x["cost"] is not None
            )
            return paraphrase.text, cost_after_generator
        except Exception as e:
            raise e

    def paraphrase(
        self,
        text: str,
        max_length: int = CONFIG.MAX_LENGTH,
        prompt: Optional[str] = None,
    ) -> tuple[dict[str, Any], List[str], float]:
        """
        Generate paraphrased versions of the input text.

        :param text: The input text to be paraphrased.
        :param prompt: Is not used, as DSPy handles prompts internally.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :return: A tuple containing the extracted information, the generated paraphrase(s) (as a list) and the total cost (for generating all paraphrases).
        """
        extractor = dspy.Predict(ExtractInfo)
        extracted_info = extractor(text=text).toDict()
        extracted_info["length"] = len(text.split())
        cost_after_extractor = sum(
            [x["cost"] for x in self.lm.history if x["cost"] is not None]
        )  # in USD, as calculated by LiteLLM for certain providers
        # https://dspy.ai/tutorials/rag/#keeping-an-eye-on-cost
        print(
            "ATTENTION. Cost of the extraction requests: ${:.6f}".format(
                cost_after_extractor
            )
        )
        # Paraphrase: One string
        paraphrases, costs = [], []
        try:
            summing = True
            for i in range(self.n_paraphrases):
                paraphrase, cost_after_generator = self.generate_one_paraphrase_based_on_extracted_information(
                    extracted_info=extracted_info, verbose=False
                )
                paraphrases.append(paraphrase)
                costs.append(cost_after_generator)
                # TODO: loop to generate multiple paraphrases, look for temperature

                # Detect cost accumulation (monotonically increasing across iterations)
                if (len(costs) > 1) and not (costs[-1] > costs[-2]):
                    summing = False
            cost_after_generator = sum(costs) if not summing else costs[-1]
            print("The costs are summed up: ", summing)

            # Total cost for both requests
            print(f"ATTENTION. Total cost for extractor + generator: ${cost_after_generator:.6f}")
        except Exception as e:
            print(e)
        return extracted_info, paraphrases, cost_after_extractor
