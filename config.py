# added to git since no secrets
import os

class BaseConfig:
    SERVER = False
    PATH2PAN25 = "data/datasets/pan25-genai-identification/pan25-dataset-converted"
    PATH2PAN23 = "data/datasets/pan23-authorship-verification/pan23-dataset-converted"
    PATH2PAN20 = "data/datasets/pan20-authorship-verification/pan20-dataset-converted"
    SAVE_PATH = "results/"
    PAN23 = "pan23"
    PAN25 = "pan25"
    PAN20 = "pan20"

class ServerConfig(BaseConfig):
    SERVER = True
    # TODO: Add server-specific configurations here
    PATH2PAN25 = "data-in-progress/data-research/authorship/pan24-genai-authorship-verification/dataset-extended-2025/"

# Choose config
ENV = os.getenv("ENV", "local")  # default to "local" if not set
CONFIG = BaseConfig() if ENV == "local" else ServerConfig()
