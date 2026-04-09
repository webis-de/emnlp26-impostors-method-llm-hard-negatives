import argparse

import ray

from genai_detection.experiments.reproduction.prec_recall_curves import *

# ray.init()

# ray job submit --address https://ray.srv.webis.de --working-dir . --runtime-env env.yml -- python genai_detection/experiments/reproduction/run_prec_recall_curves_from_mongodb.py

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    our_figure_impostor_options = [
        "on_the_fly_chatnoir",
        # "on_the_fly_serpapi",
        "on_the_fly_startpage",
        "in_domain",
        # "one_step_llm",   # SAIA URL on betaweb error: httpx.InvalidURL: Invalid port: 'academiccloud:de'
        "two_step_llm",
        # "translation",
        # "mirror_minds",   # raises error
    ]

    run_prec_recall_curves(
        dataset_name=CONFIG.STUDENT_ESSAYS,
        imp_gen_techniques=our_figure_impostor_options,
        compute_score_fn=compute_prec_recall_f1_acc_dict_on_existing_impostor_scores,
    )

    run_prec_recall_curves(
        dataset_name=CONFIG.BLOG, imp_gen_techniques=our_figure_impostor_options, compute_score_fn=compute_prec_recall_f1_acc_dict_on_existing_impostor_scores
    )
