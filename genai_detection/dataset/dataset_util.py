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

import argparse
import random
from pathlib import Path

from genai_detection.dataset.blog_corpus_dataset_loader import BlogCorpusDatasetLoader
from genai_detection.dataset.gutenberg_dataset_loader import GutenbergDatasetLoader
from genai_detection.dataset.pan_dataset_loader import (
    Pan23DatasetLoader,
    Pan20DatasetLoader,
    Pan25DatasetLoader,
)
from genai_detection.dataset.student_essays_dataset_loader import StudentEssayDatasetLoader
from genai_detection.paraphrasing.two_step_paraphrasers import *

logger = logging.getLogger(__name__)

random.seed(42)
# Koppel et al. (2004): 500 words
# Bevendorff et al. (2019): 700 words (https://www.degruyterbrill.com/document/doi/10.1515/itit-2019-0046/html?casa_token=pbCaF7FgUXoAAAAA:8Vw71FUWE5spAbSsEuGGTdIjjm_o1_eb_inHwU3BR6eSrdVMOYy3--iqvDJwCV7EQ1HWtQBh610)
# Bevendorff et al. (2025): 3000 characters (https://aclanthology.org/2025.findings-acl.194.pdf)
MIN_NUM_WORDS = 700  # minimum number of words in a text to be considered valid


# === SYSTEM SPECIFIC USAGE ===

def run_student_essay():
    base_dir = (
        Path(__file__).resolve().parents[2]
        / CONFIG.DATA_BASE_PATH
        / "student_essays/Intro2006"
    )
    assert (
        base_dir.exists()
    ), f"Path {base_dir} to student essays dataset does not exist."
    output_dir = os.path.join(base_dir, "student-essays-dataset-converted")

    loader = StudentEssayDatasetLoader(path=base_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)


def run_blog_corpus():
    base_dir = (
        Path(__file__).resolve().parents[2] / CONFIG.DATA_BASE_PATH / "Blog_corpus/"
    )
    assert (
        base_dir.exists()
    ), f"Base directory {base_dir} does not exist. Current path: {os.getcwd()}"
    output_dir = base_dir / "blog-dataset-converted"

    loader = BlogCorpusDatasetLoader(path=base_dir / "blogtext.csv")
    dataset = loader.load()
    dataset.save_to_disk(output_dir)

def delete_blog_from_mongoDB():
    loader = BlogCorpusDatasetLoader(path="blogtext.csv")
    loader.delete_dataset_from_mongoDB()

def merge_train_test_mongodb_collections():
    loader = BlogCorpusDatasetLoader(path="blogtext.csv")
    loader.merge_train_test_mongodb_collections()

def run_gutenberg_corpus():
    base_dir = (
        Path(__file__).resolve().parent.parent / CONFIG.DATA_BASE_PATH / "gutenberg/"
    )
    assert (
        base_dir.exists()
    ), f"Base directory {base_dir} does not exist. Current path: {os.getcwd()}"
    output_dir = base_dir / "gutenberg-dataset-converted"

    loader = GutenbergDatasetLoader(path=base_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # run_blog_corpus()
    # delete_blog_from_mongoDB()
    # run_gutenberg_corpus()
    # run_student_essay()
    merge_train_test_mongodb_collections()
