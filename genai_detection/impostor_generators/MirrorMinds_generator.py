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
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from bson import ObjectId
from huggingface_hub import snapshot_download
from llm_question_generator.parser import apply_extraction
from llm_question_generator.question_config import QuestionGeneratorConfig
from llm_question_generator.question_generator import QuestionGenerator
from llm_response_generator.response_config import ResponseGeneratorConfig
from llm_response_generator.response_generator import ResponseGenerator

from genai_detection.impostor_generators.ImpostorGenerator import MongoDBSavedGenerator

logger = logging.getLogger(__name__)

class MirrorMindsGenerator(MongoDBSavedGenerator):
    """
    MirrorMindsGenerator is a generator that creates impostor texts by mirroring the structure of the original text.
    It uses the MirrorMinds model to generate paraphrases that maintain the original meaning while altering the wording.
    """

    def __init__(
        self, model_id: str = "MirrorMinds/mirror_minds", n_impostors: int = 1, top_n_freq_words:int=100000
    ):
        super().__init__(n_impostors=n_impostors, top_n_freq_words=top_n_freq_words)
        torch.cuda.empty_cache()
        self.model_id = model_id
        self.model_path = snapshot_download(repo_id="google/flan-t5-small")

    def generate_impostors(
        self, text: Optional[str], text_id: Optional[ObjectId]
    ) -> List[str]:
        """
        Generate an impostor text by mirroring the structure of the input text.

        :param text: The original text to be mirrored.
        :return: The generated impostor text.
        """
        start_time = time.time()
        assert isinstance(text, str), "Input text must be a string."

        text_df = pd.DataFrame({"Essay": [text]})
        # make tmp directory if it does not exist
        tmp_path = Path(os.getcwd()) / "tmp-MirrorMinds"
        os.makedirs(tmp_path, exist_ok=True)
        text_df.to_pickle(tmp_path / "sample_essays.pkl")
        impostor_texts = []
        for i in range(self.n_impostors):
            question_config = QuestionGeneratorConfig(
                model_path=self.model_path,
                input_path=tmp_path / "sample_essays.pkl",
                output_path=tmp_path / "generated_questions.pkl",
            )
            response_config = ResponseGeneratorConfig(
                model_path=self.model_path,
                input_path=question_config.output_path,
                output_path=tmp_path / "generated_responses.pkl",
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
                val = df["generated_text"].iloc[0]
                if pd.notna(val) and not (isinstance(val, float)):
                    impostor_texts.append(val)
            else:
                logging.warning(f"ERROR: No valid response generated. Original was: {text[:200]}")
                impostor_texts.append("ERROR: No valid response generated.")

        logging.info(f"Total script runtime: {time.time() - start_time:.2f} seconds")
        logging.info("Generated questions and responses saved successfully.")
        os.remove(tmp_path / "sample_essays.pkl")
        os.remove(tmp_path / "generated_questions.pkl")
        os.remove(tmp_path / "generated_responses.pkl")
        os.rmdir(tmp_path)
        return impostor_texts


if __name__ == "__main__":
    generator = MirrorMindsGenerator()
    path2datasets = Path(os.getcwd()).resolve() / "data" / "datasets" / "custom_texts"
    file_name = "cnn_040725"  # Dalai Lama
    original_text = open(path2datasets / f"{file_name}.txt").read()
    impostor_text = generator.generate_impostors(original_text)
    logging.info("Impostor Text: %s", impostor_text)
