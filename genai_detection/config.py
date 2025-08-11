# added to git since no secrets
import os
from dotenv import load_dotenv

load_dotenv()


class BaseConfig:
    SERVER = False
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    BLABLADOR_KEY = os.getenv("BLABLADOR_KEY")
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    SAIA_KEY = os.getenv("SAIA_KEY")
    DEEPL_API_KEY = os.getenv("DEEPL_KEY")
    IONOS_KEY = os.getenv("IONOS_KEY")
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
    IMPOSTOR = "impostor"
    UNMASKING = "unmasking"
    TEMPERATURE = 0.7
    MAX_LENGTH = 512  # Maximum length of the generated paraphrase
    OLLAMA_MODEL = "zephyr:7b"  # "mistral:7b"  # "default:latest"
    SAIA_URL = "https://chat-ai.academiccloud.de/v1"
    SAIA_MODEL = "openai-gpt-oss-120b"
    IONOS_URL = "https://openai.inference.de-txl.ionos.com/v1"
    IONOS_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


class ServerConfig(BaseConfig):
    SERVER = True
    # TODO: Add server-specific configurations here
    PATH2PAN20 = "TODO"
    PATH2PAN23 = "/mnt/ceph/storage/data-in-progress/data-research/authorship/pan23-authorship-verification/"
    PATH2PAN25 = "/mnt/ceph/storage/data-in-progress/data-research/authorship/pan24-genai-authorship-verification/dataset-extended-2025/"


# Choose config
CONFIG = ServerConfig() if os.path.exists("mnt/ceph") else BaseConfig()
