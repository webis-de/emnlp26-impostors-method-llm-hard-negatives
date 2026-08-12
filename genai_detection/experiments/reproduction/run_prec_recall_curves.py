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
import argparse

from genai_detection.experiments.reproduction.prec_recall_curves import *

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    our_figure_impostor_options = [
        "one_step_llm",
        # "random_words",
        # "two_step_llm",
        # "on_the_fly_chatnoir",
        # # "on_the_fly_serpapi",
        # "on_the_fly_startpage",
        # "in_domain",
        # "translation",
        # "mirror_minds",   # raises error
    ]

    run_prec_recall_curves(
        dataset_name=CONFIG.STUDENT_ESSAYS,
        imp_gen_techniques=our_figure_impostor_options,
            one_step_llm_label_translations=CONFIG.ONE_STEP_LLM_LABEL_TRANSLATIONS,
    )

    run_prec_recall_curves(
        dataset_name=CONFIG.BLOG, imp_gen_techniques=our_figure_impostor_options
    )
