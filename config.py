# added to git since no secrets
import os

class BaseConfig:
    SERVER = False
    PATH2PAN25 = "data/datasets/dataset-extended-2025-part-converted"

class ServerConfig(BaseConfig):
    SERVER = True
    # TODO: Add server-specific configurations here
    PATH2PAN25 = "data-in-progress/data-research/authorship/pan24-genai-authorship-verification/dataset-extended-2025/"

# Choose config
ENV = os.getenv("ENV", "local")  # default to "local" if not set
CONFIG = BaseConfig() if ENV == "local" else ServerConfig()
