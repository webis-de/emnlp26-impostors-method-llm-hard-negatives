# Copyright 2024 Klara M. Gutekunst, Webis
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

import deepl
import dspy

from genai_detection.config import CONFIG
from genai_detection.paraphrasing.one_step_paraphrasers import OneStepParaphraser
from genai_detection.paraphrasing.paraphraser import Paraphraser

# Field descriptions dictionary
FIELD_DESCRIPTIONS = {
    "title": "A short, descriptive title summarizing the text in one phrase or sentence.",
    "genre": "The literary or content genre of the text (e.g., essay, poem, news article, academic paper, novel, speech, letter, religious text, etc.).",
    "tone": "The emotional or stylistic tone (e.g., formal, humorous, persuasive, neutral, melancholic, satirical, didactic).",
    "century": "The century in which the text was likely written (e.g., '18th century', '20th century', '21st century'). Use language, style, and context clues to infer.",
    "audience": "The intended audience or readership (e.g., general public, scholars, children, political leaders, students, religious followers).",
    "author": "A concise author profile including likely traits such as gender (if implied), profession, education level, nationality, or perspective (e.g., 'a 19th-century British poet', 'a modern journalist', 'an academic researcher').",
    "bulletpoints": "A list of concise bullet points summarizing the main ideas or arguments of the text, written in plain language.",
    "length": "The approximate target length of the generated text, in number of words.",
}


class ExtractInfo(dspy.Signature):
    """Extract structured literary and stylistic information from text."""

    text: str = dspy.InputField(desc="The full input text to analyze.")
    title: str = dspy.OutputField(desc=FIELD_DESCRIPTIONS["title"])
    genre: str = dspy.OutputField(desc=FIELD_DESCRIPTIONS["genre"])
    tone: str = dspy.OutputField(desc=FIELD_DESCRIPTIONS["tone"])
    century: str = dspy.OutputField(desc=FIELD_DESCRIPTIONS["century"])
    audience: str = dspy.OutputField(desc=FIELD_DESCRIPTIONS["audience"])
    author: str = dspy.OutputField(desc=FIELD_DESCRIPTIONS["author"])
    bulletpoints: list[str] = dspy.OutputField(desc=FIELD_DESCRIPTIONS["bulletpoints"])


class GenerateText(dspy.Signature):
    """Generate a text based on structured literary and stylistic information."""

    title: str = dspy.InputField(desc=FIELD_DESCRIPTIONS["title"])
    genre: str = dspy.InputField(desc=FIELD_DESCRIPTIONS["genre"])
    tone: str = dspy.InputField(desc=FIELD_DESCRIPTIONS["tone"])
    century: str = dspy.InputField(desc=FIELD_DESCRIPTIONS["century"])
    audience: str = dspy.InputField(desc=FIELD_DESCRIPTIONS["audience"])
    author: str = dspy.InputField(desc=FIELD_DESCRIPTIONS["author"])
    bulletpoints: list[str] = dspy.InputField(desc=FIELD_DESCRIPTIONS["bulletpoints"])
    length: int = dspy.InputField(desc=FIELD_DESCRIPTIONS["length"])
    text: str = dspy.OutputField(
        desc=(
            "A newly generated text that sounds as if written by the described author. "
            "It must reflect the specified **title**, **genre**, **tone**, **century**, "
            "**audience**, and **author profile**. "
            "The content should elaborate naturally on the given **bulletpoints** and "
            "have a length approximately matching the provided word count."
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
    ):
        """
        Initializes the OpenAI paraphraser model using DSPy.
        :param model_id: Name of the OpenAI model to use.
        :param temperature: Temperature for the model. Needs to be 1.0 for reasoning models like GPT-5.
        """
        is_reasoning_model = any(name in model_id.lower() for name in ["gpt-5"])

        self.lm = dspy.LM(
            model_id,
            api_base=CONFIG.OPENAI_URL,
            api_key=CONFIG.OPENAI_API_KEY,
            model_type="chat",
            temperature=1.0 if is_reasoning_model else temperature,
            max_tokens=16000 if is_reasoning_model else CONFIG.MAX_LENGTH,
        )
        self.model_id = model_id
        dspy.configure(lm=self.lm)

    def generate_paraphrase(
        self, extracted_info: dict[str, Any], verbose: bool = True
    ) -> tuple[str, float]:
        """
        Generate a paraphrase based on the extracted information.
        :param verbose: Whether to print debug information.
        :param extracted_info: Dictionary containing the extracted information.
        :return: A tuple containing the generated paraphrase (as string) and the cost after generation.
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
                print("generated paraphrase", paraphrase)
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
    ) -> tuple[dict[str, Any], str, float]:
        """
        Generate paraphrased versions of the input text.

        :param text: The input text to be paraphrased.
        :param prompt: Is not used, as DSPy handles prompts internally.
        :param max_length: The maximum number of tokens to generate in the paraphrase.
        :return: A tuple containing the extracted information, the generated paraphrase and the total cost.
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

        try:
            paraphrase, cost_after_generator = self.generate_paraphrase(
                extracted_info=extracted_info
            )

            # Total cost for both requests
            total_cost = cost_after_extractor + (
                cost_after_generator - cost_after_extractor
            )
            print(f"ATTENTION. Total cost for extractor + generator: ${total_cost:.6f}")
            return extracted_info, paraphrase, total_cost
        except Exception as e:
            print(e)
            return extracted_info, "", cost_after_extractor


class TranslationParaphraser(Paraphraser):
    """
    A paraphrasing model that first extracts the title of the text, tone and genre from the input text using one LLM and then generates a paraphrase based on this information.

    Inspired by the work of:
    C. Zhou, C. Qiu, L. Liang and D. E. Acuna, "Paraphrase Identification With Deep Learning: A Review of Datasets and Methods," in IEEE Access, vol. 13, pp. 65797-65822, 2025, doi: 10.1109/ACCESS.2025.3556899.

    For implementation details, see:
    https://github.com/deeplcom/deepl-python (accessed 25.07.2025)
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
        self.model_id = "translation"
        assert isinstance(text_extractor, OneStepParaphraser) and isinstance(
            text_generator, OneStepParaphraser
        ), "Both text_extractor and text_generator must be instances of NaiveParaphraser or its subclasses."
        super().__init__(text_extractor=text_extractor, text_generator=text_generator)
        self.language = language
        self.extractor_prompt = f"Translate the text above into {self.language}. Do not use direct quotes or newlines. Output only the translated text, without any additional commentary or formatting."
        self.generator_prompt = f"Translate the text above from {self.language} into English. Do not use direct quotes or newlines. Output only the translated text, without any additional commentary or formatting."
        self.deepl_client = deepl.DeepLClient(CONFIG.DEEPL_API_KEY)
        self.text_extractor = text_extractor
        self.text_generator = text_generator

    def paraphrase(
        self, text: str, prompt: Optional[str], max_length: int = CONFIG.MAX_LENGTH
    ) -> List[str]:
        print(
            f"[DEBUG] Using TranslationParaphraser with prompt: {self.extractor_prompt}"
        )
        paraphrased_texts = []
        try:
            translation = self.deepl_client.translate_text(text, target_lang="FR")
            language = (
                "EN-US"
                if translation.detected_source_lang == "EN"
                else translation.detected_source_lang
            )
            res = self.deepl_client.translate_text(
                translation.text, target_lang=language
            )
            paraphrased_texts.append(res.text)
        except Exception as e:
            print(f"[ERROR] Failed to translate text using DeepL: {e}")

        translation = self.text_extractor.paraphrase(
            text=text, prompt=self.extractor_prompt, max_length=max_length
        )
        while not translation:
            print(
                f"[WARNING] No translation returned. Retrying with the same text and prompt: {self.extractor_prompt}"
            )
            translation = self.text_extractor.paraphrase(
                text=text, prompt=self.extractor_prompt, max_length=max_length
            )
        paraphrased_texts.extend(
            self.text_generator.paraphrase(
                text=translation[0], prompt=self.generator_prompt
            )
        )

        return paraphrased_texts
