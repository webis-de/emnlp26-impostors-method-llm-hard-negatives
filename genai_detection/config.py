# added to git since no secrets
import os

from dotenv import load_dotenv

load_dotenv()


class BaseConfig:
    SERVER = False
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    OPENAI_KEY = os.getenv("OPENAI_KEY")
    SAIA_KEY = os.getenv("SAIA_KEY")
    DEEPL_API_KEY = os.getenv("DEEPL_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_KEY")
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

    PROMPT = "Paraphrase the text above without changing its meaning. Use different words and vary the sentence structure while maintaining a consistent tone. Your paraphrase should be three times as long than the original. Output only the paraphrased sentence, with NO explanations or extra text."

    MONGO_DATABASE = "impostors"  # initial db (!= collection in db)
    MONGO_ORIGINAL_TEXT_COLLECTION = "original_text"  # collection (!= db)
    MONGO_PARAPHRASE_COLLECTION = "paraphrase"  # collection (!= db)
    MONGO_PARAPHRASE_SCORE_COLLECTION = "paraphrase_score"  # collection (!= db)
    MONGO_IMPOSTOR_OUTPUT_COLLECTION = "impostors_output" # collection (!= db)
    MONGO_USER = os.getenv("MONGO_INITDB_ROOT_USERNAME")
    MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
    MONGO_HOST = "localhost:27018"  # Local, but forwarded to server
    # MONGO_HOST = "artificial-authorship-verification-mongodb.webisservices.svc.cluster.local:27017" # Kubernetes


CONFIG = BaseConfig()
