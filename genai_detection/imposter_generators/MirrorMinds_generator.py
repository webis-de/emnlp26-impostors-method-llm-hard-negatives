import os
from pathlib import Path
from llm_question_generator.question_generator import QuestionGenerator
from llm_question_generator.parser import apply_extraction
from llm_question_generator.question_config import QuestionGeneratorConfig
from llm_response_generator.response_generator import ResponseGenerator
from llm_response_generator.response_config import ResponseGeneratorConfig
import pandas as pd
import torch
import time
from huggingface_hub import snapshot_download
from genai_detection.imposter_generators.ImposterGenerator import BaseImposterGenerator
from genai_detection.config import CONFIG


class MirrorMindsGenerator(BaseImposterGenerator):
    """
    MirrorMindsGenerator is a generator that creates imposter texts by mirroring the structure of the original text.
    It uses the MirrorMinds model to generate paraphrases that maintain the original meaning while altering the wording.
    """

    def __init__(self, model_id: str = "MirrorMinds/mirror_minds"):
        super().__init__(n_impostors=1)
        torch.cuda.empty_cache()
        self.model_path = snapshot_download(repo_id="google/flan-t5-small")

    def generate_imposters(self, text: str) -> str:
        """
        Generate an imposter text by mirroring the structure of the input text.

        :param text: The original text to be mirrored.
        :return: The generated imposter text.
        """
        start_time = time.time()

        text_df = pd.DataFrame({"Essay": [text]})
        text_df.to_pickle(CONFIG.DATA_BASE_PATH + "/sample_essays.pkl")
        imposter_texts = []
        for i in range(self.n_impostors):
            question_config = QuestionGeneratorConfig(
                model_path=self.model_path,
                input_path=CONFIG.DATA_BASE_PATH + "/sample_essays.pkl",
                output_path=CONFIG.DATA_BASE_PATH + "/generated_questions.pkl",
            )
            response_config = ResponseGeneratorConfig(
                model_path=self.model_path,
                input_path=question_config.output_path,
                output_path=CONFIG.DATA_BASE_PATH + "/generated_responses.pkl",
            )

            df = pd.read_pickle(question_config.input_path)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            generator = QuestionGenerator(
                model_path=question_config.model_path, device=device
            )
            df = generator.generate_questions(
                df, question_config.essay_column, question_config.max_new_tokens
            )
            df = apply_extraction(df)
            df.to_pickle(question_config.output_path)

            response_generator = ResponseGenerator(
                model_path=response_config.model_path, device=device
            )
            df = response_generator.generate_responses(
                response_config.input_path, response_config.output_path
            )
            if not df.empty:
                imposter_texts.append(df["generated_text"].iloc[0])

        print(f"Total script runtime: {time.time() - start_time:.2f} seconds")
        print("Generated questions and responses saved successfully.")
        print(df)
        return imposter_texts


if __name__ == "__main__":
    generator = MirrorMindsGenerator()
    path2datasets = Path(os.getcwd()).resolve() / "data" / "datasets" / "custom_texts"
    file_name = "cnn_040725"  # Dalai Lama
    original_text = open(path2datasets / f"{file_name}.txt").read()
    imposter_text = generator.generate_imposters(original_text)
    print("Imposter Text:", imposter_text)
