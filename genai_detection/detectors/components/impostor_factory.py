# components/impostor_factory.py
import inspect

from genai_detection.impostor_generators.MirrorMinds_generator import MirrorMindsGenerator
from genai_detection.impostor_generators.chatnoir_impostor_generator import ChatNoirSearchImpostorGenerator
from genai_detection.impostor_generators.content_impostor_generator import ContentImpostorGenerator
from genai_detection.impostor_generators.google_search_impostor_generator import GoogleSearchImpostorGenerator
from genai_detection.impostor_generators.in_domain_impostor_generator import (
    InDomainImpostorGenerator,
    BlogImpostorGenerator,
)
from genai_detection.impostor_generators.naive_impostor_generator import NaiveImpostorGenerator
from genai_detection.impostor_generators.startpage_impostor_generator import StartPageSearchImpostorGenerator
from genai_detection.impostor_generators.text_length_impostor_generator import TextLenImpostorGenerator
from genai_detection.impostor_generators.translation_impostor_generator import TranslationImpostorGenerator
from genai_detection.impostor_generators.two_step_impostor_generator import TwoStepImpostorGenerator

IMPOSTOR_GENERATORS = {
    "two_step_llm": TwoStepImpostorGenerator,
    "naive_llm": NaiveImpostorGenerator,
    "translation": TranslationImpostorGenerator,
    "fixed": InDomainImpostorGenerator,
    "blogs": BlogImpostorGenerator,
    "on_the_fly":StartPageSearchImpostorGenerator,
    "on_the_fly_chatnoir": ChatNoirSearchImpostorGenerator,
    "on_the_fly_serpapi": GoogleSearchImpostorGenerator,
    "on_the_fly_startpage": StartPageSearchImpostorGenerator,
    "content": ContentImpostorGenerator,
    "mirror_minds": MirrorMindsGenerator,
    "text_len": TextLenImpostorGenerator,
}

def create_impostor_generator(impostor_technique, **kwargs):
    if impostor_technique not in IMPOSTOR_GENERATORS:
        raise ValueError(f"Unknown impostor technique: {impostor_technique}")
    cls = IMPOSTOR_GENERATORS[impostor_technique]

    # inspect constructor parameters
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}

    # filter kwargs to only those accepted by this class
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return cls(**filtered_kwargs)
