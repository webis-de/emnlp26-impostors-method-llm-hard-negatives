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
import inspect

from genai_detection.impostor_generators.chatnoir_impostor_generator import ChatNoirSearchImpostorGenerator
from genai_detection.impostor_generators.google_search_impostor_generator import GoogleSearchImpostorGenerator
from genai_detection.impostor_generators.in_domain_impostor_generator import InDomainImpostorGenerator
from genai_detection.impostor_generators.naive_impostor_generator import NaiveImpostorGenerator
from genai_detection.impostor_generators.startpage_impostor_generator import StartPageSearchImpostorGenerator
from genai_detection.impostor_generators.translation_impostor_generator import TranslationImpostorGenerator
from genai_detection.impostor_generators.two_step_impostor_generator import TwoStepImpostorGenerator

IMPOSTOR_GENERATORS = {
    "two_step_llm": TwoStepImpostorGenerator,
    "one_step_llm": NaiveImpostorGenerator,
    "translation": TranslationImpostorGenerator,
    "in_domain": InDomainImpostorGenerator,
    "on_the_fly": StartPageSearchImpostorGenerator,
    "on_the_fly_chatnoir": ChatNoirSearchImpostorGenerator,
    "on_the_fly_serpapi": GoogleSearchImpostorGenerator,
    "on_the_fly_startpage": StartPageSearchImpostorGenerator,
    # "mirror_minds": MirrorMindsGenerator
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
