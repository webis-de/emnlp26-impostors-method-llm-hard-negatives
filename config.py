# added to git since no secrets
import os
from dotenv import load_dotenv

load_dotenv()

class BaseConfig:
    SERVER = False
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    BLABLADOR_KEY = os.getenv("BLABLADOR_KEY")
    PATH2PAN25 = "data/datasets/pan25-genai-identification/pan25-dataset-converted"
    PATH2PAN23 = "data/datasets/pan23-authorship-verification/pan23-dataset-converted"
    PATH2PAN20 = "data/datasets/pan20-authorship-verification/pan20-dataset-converted"
    PATH2KOPPEL_WEBIS = "data/datasets/corpus-webis-authorship/koppel/koppel-webis-dataset-converted"
    PATH2GUTENBERG = "data/datasets/gutenberg/gutenberg-dataset-converted"
    PATH2BLOG = "data/datasets/Blog_corpus/blog-dataset-converted"
    PATH2GENERIC_ON_FLY_IMP = "data/datasets/google-on-the-fly-imposters"
    SAVE_PATH = "results/"
    PAN23 = "pan23"
    PAN25 = "pan25"
    PAN20 = "pan20"
    KOPPEL = "koppel"
    BLOG = "blog"
    GUTENBERG = "gutenberg"
    IMPOSTER = "imposter"
    UNMASKING = "unmasking"

class ServerConfig(BaseConfig):
    SERVER = True
    # TODO: Add server-specific configurations here
    PATH2PAN20 = "TODO"
    PATH2PAN23 = "/mnt/ceph/storage/data-in-progress/data-research/authorship/pan23-authorship-verification/"
    PATH2PAN25 = "/mnt/ceph/storage/data-in-progress/data-research/authorship/pan24-genai-authorship-verification/dataset-extended-2025/"

# Choose config
CONFIG = ServerConfig() if os.path.exists('mnt/ceph') else BaseConfig()
