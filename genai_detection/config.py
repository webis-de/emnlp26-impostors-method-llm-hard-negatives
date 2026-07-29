# Copyright 2026 Klara M. Gutekunst, Webis
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


# added to git since no secrets
import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

print(load_dotenv())          # True = Datei gefunden
print(os.getcwd())            # aktuelles Arbeitsverzeichnis
print(os.getenv("MONGO_INITDB_ROOT_USERNAME"))


class BaseConfig:
    SERVER = False
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    SAIA_KEY = os.getenv("SAIA_KEY")
    SAIA_KEY_K = os.getenv("SAIA_KEY_K")
    SAIA_KEYS = [SAIA_KEY, SAIA_KEY_K]#os.getenv("SAIA_KEY_"), os.getenv("SAIA_KEY_A"), os.getenv("SAIA_KEY_B"),
    # os.getenv("SAIA_KEY_P"), os.getenv("SAIA_KEY_G")]
    DEEPL_API_KEY = os.getenv("DEEPL_KEY")
    OLLAMA_KEY = os.getenv("OLLAMA_KEY")
    OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")
    CHATNOIR_KEY = os.getenv("CHATNOIR_KEY")
    DATA_BASE_PATH = "data/datasets"
    PATH2PAN25 = f"{DATA_BASE_PATH}/pan25-genai-identification/pan25-dataset-converted"
    PATH2PAN23 = (
        f"{DATA_BASE_PATH}/pan23-authorship-verification/pan23-dataset-converted"
    )
    PATH2PAN20 = (
        f"{DATA_BASE_PATH}/pan20-authorship-verification/pan20-dataset-converted"
    )
    PATH2KOPPEL_WEBIS = f"{DATA_BASE_PATH}/corpus-webis-authorship/koppel/koppel-webis-dataset-converted"
    PATH2GUTENBERG = f"{DATA_BASE_PATH}/gutenberg/gutenberg-dataset-converted"
    PATH2BLOG = f"{DATA_BASE_PATH}/Blog_corpus/blog-dataset-converted"
    PATH2STUDENT_ESSAYS = (
        f"{DATA_BASE_PATH}/student_essays/Intro2006/student-essays-dataset-converted"
    )
    PATH2ARTIFICIAL_STUDENT_ESSAYS = f"{DATA_BASE_PATH}/artificial_student_essays/artificial-student-essays-dataset-converted"
    PATH2GENERIC_ON_FLY_IMP = f"{DATA_BASE_PATH}/google-on-the-fly-impostors"
    PATH2CROSS_GENRE = f"{DATA_BASE_PATH}/cross_genre/cross-genre-dataset"
    SAVE_PATH = "results/"
    PAN23 = "pan23"
    PAN25 = "pan25"
    PAN20 = "pan20"
    KOPPEL = "koppel"
    BLOG = "blog"
    GUTENBERG = "gutenberg"
    STUDENT_ESSAYS = "student_essays"
    ARTIFICIAL_STUDENT_ESSAYS = "artificial_student_essays"
    IMPOSTOR = "impostor"
    UNMASKING = "unmasking"
    TEMPERATURE = 1.0
    MAX_LENGTH = 2100  # 512  # Maximum length of the generated paraphrase
    OLLAMA_MODEL = "zephyr:7b"  # "mistral:7b"  # "default:latest"
    OLLAMA_URL = "https://llm.web.webis.de/api"
    SAIA_URL = "https://chat-ai.academiccloud.de/v1"
    SAIA_MODEL = "openai-gpt-oss-120b"
    OPENAI_URL = "https://api.openai.com/v1"
    OPENAI_MODEL = "openai/gpt-5-nano-2025-08-07"  # specify snapshot for consistency: https://platform.openai.com/docs/models/gpt-5-nano (30.10.2025)
    # too expensive without batching:
    BATCH_OPENAI_MODEL =  "gpt-5.4-nano-2026-03-17"     # https://developers.openai.com/api/docs/models/gpt-5.4-nano
    HUGGINGFACE_FINETUNED_MODEL = "tuner007/pegasus_paraphrase"
    PROMPT = "Paraphrase the text above without changing its meaning. Use different words and vary the sentence structure while maintaining a consistent tone. Your paraphrase should be three times as long than the original. Output only the paraphrased sentence, with NO explanations or extra text."

    MONGO_DATABASE = "impostors"  # initial db (!= collection in db)
    MONGO_ORIGINAL_TEXT_COLLECTION = "original_texts"  # collection (!= db)
    MONGO_PARAPHRASE_COLLECTION = "non_naive_paraphrases"  # collection (!= db); for (formerly) two-step paraphrases
    MONGO_NAIVE_PARAPHRASE_COLLECTION = "naive_paraphrases"  # collection (!= db); for (formerly) one-step paraphrases
    MONGO_ON_THE_FLY_COLLECTION = "on_the_fly_paraphrases"  # collection (!= db)
    MONGO_TRANSLATION_COLLECTION = "translation_paraphrases"  # collection (!= db)
    MONGO_PARAPHRASE_SCORE_COLLECTION = "paraphrase_scores"  # collection (!= db)
    MONGO_IMPOSTOR_OUTPUT_COLLECTION = "impostors_outputs" # collection (!= db)
    MONGO_IMPOSTOR_ABLATION_OUTPUT_COLLECTION = "impostors_outputs_ablations" # collection (!= db)
    MONGO_BASELINE_OUTPUT_COLLECTION = "baseline_outputs" # collection (!= db)
    MONGO_PAN_METRICS_COLLECTION = "pan_metrics"
    MONGO_TEST_PAIRS_COLLECTION = "test_pairs" # collection (!= db); for IDs of texts and their ground truth (
    # reproducibility of evaluation)
    MONGO_TRAIN_PAIRS_COLLECTION = (
        "train_pairs"  # collection (!= db); for IDs of texts and their ground truth
    )
    MONGO_ALL_PAIRS_COLLECTION = "all_pairs"
    MONGO_SUP_BASELINE_CONFIG_PREDS = "supervised_baseline_diff_config_preds"
    MONGO_SUP_BASELINE_CONFIG_SCORES = "supervised_baseline_diff_config_scores"
    MONGO_JOB_COLLECTION_NAME = "openai_batch_paraphrase_jobs"
    MONGO_USER = os.getenv("MONGO_INITDB_ROOT_USERNAME")
    MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
    MONGO_HOST = "localhost"  # Local, but forwarded to server
    MONGO_PORT = 27018

    THRESHOLDS = np.arange(0.0, 1.001, 0.001)

    RETRIEVAL_INDEX_TRANSLATIONS = {
        "on_the_fly_chatnoir": "msmarco-v2.1",
        "on_the_fly_serpapi": "Google",
        "on_the_fly_startpage": "Google-Startpage",
    }

    LABEL_TRANSLATIONS = {
        "in_domain": "In-Domain",
        "on_the_fly": "Retrieval-Based",
        "on_the_fly_chatnoir": "Retrieval-Based (ChatNoir)",
        "on_the_fly_serpapi": "Retrieval-Based (SerpApi)",
        "on_the_fly_startpage": "Retrieval-Based (Startpage)",
        "one_step_llm": "One-Step Paraphraser (LLM)",
        "two_step_llm": "Two-Step Paraphraser (LLM)",
        "random_words": "Random English Words",
        "unsupervised_baseline_min-max": "Unsup. Min-Max (B)",
        "unsupervised_baseline_cosine": "Unsup. Cosine (B)",
        "supervised_baseline": "Sup. SVM (B)",
        "unmasking": "Unmasking",
        "ppmd": "PPMd",
        "translation": "Translation",
        "potha2017": "Potha et al., 2017",
        "asgalf": "Khonji & Iraqi, 2014",
        "std_impostor": "Kestemont et al., 2016",
        "bdi": "Nagy, 2024",
        "homotopy": "Gutierrez et al., 2015",
    }

    ONE_STEP_LLM_LABEL_TRANSLATIONS = {
        "gpt-5.4-nano-2026-03-17": "GPT 5.4 Nano",
        "tuner007/pegasus_paraphrase": "Pegasus",
    }

    # Fixed colors per label (matplotlib-compatible)
    LABEL_COLORS = {
        "in_domain": "#1f77b4",  # blue
        "on_the_fly": "#9467bd",  # purple
        "on_the_fly_chatnoir": "#5e3c99",  # dark violet
        "on_the_fly_serpapi": "#b103fc",  # violet
        "on_the_fly_startpage": "#7bccc4",  # light teal
        "one_step_llm": "#8c564b",  # brown
        "two_step_llm": "#e377c2",  # pink
        "random_words": "#A80000",  # red
        "unsupervised_baseline_min-max": "#ff7f0e",  # orange
        "unsupervised_baseline_cosine": "#2ca02c",  # green
        "supervised_baseline": "#d62728",  # red
        "unmasking": "#7f7f7f",  # gray
        "ppmd": "#bcbd22",  # olive
        "translation": "#17becf",  # cyan
        "potha2017": "#aec7e8",  # light blue
        "asgalf": "#ffbb78",  # light orange
        "std_impostor": "#98df8a",  # light green
        "bdi": "#c5b0d5",  # light purple
        "homotopy": "#ff9896",  # light red
    }

    ONE_STEP_LLM_LABEL_COLORS = {
        "gpt-5.4-nano-2026-03-17": "#005f73", # darker blue
        "tuner007/pegasus_paraphrase": "#005f78", # darker purple
    }

    SCORE_TRANSLATIONS = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "accuracy": "Accuracy",
        "c_at_1": "c@1",
        "auroc": "AUROC",
        "auroc_c_at_1": "AUROC×c@1",
    }

    DATASET_TRANSLATIONS = {BLOG:"Blog Posts", STUDENT_ESSAYS: "Student Essays"}

CONFIG = BaseConfig()
