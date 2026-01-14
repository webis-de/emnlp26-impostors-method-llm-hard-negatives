import argparse

import ray

from genai_detection.experiments.reproduction.prec_recall_curves import *

# ray.init()

# ray job submit --address https://ray.srv.webis.de --working-dir . --runtime-env env.yml -- python genai_detection/experiments/reproduction/run_prec_recall_curves.py

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    print(logger.level)

    parser = argparse.ArgumentParser(description="Compare detectors.")
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Number of rounds (default: %(default)s)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=100000,
        help="Number of top space-free ngrams to consider (default: %(default)s)",
    )

    parser.add_argument(
        "--n_impostors",
        type=int,
        default=50,
        help="Number of impostors to generate per candidate (default: %(default)s)",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=[CONFIG.STUDENT_ESSAYS, CONFIG.BLOG],
        default=CONFIG.BLOG,
        help="Dataset to use for visualization (default: %(default)s)",
    )

    parser.add_argument(
        "--impostor_technique",
        type=str,
        choices=[
            "translation",
            "on-the-fly",   # Startpage by default
            "on_the_fly_chatnoir",
            "on_the_fly_serpapi",
            "on_the_fly_startpage",
            "in_domain",
            "one_step_llm",
            "two_step_llm",
            "mirror_minds",
        ],
        default="in_domain",
        help="impostor technique to use (default: %(default)s)",
    )

    parser.add_argument(
        "--upsample",
        type=bool,
        default=True,
        help="Whether texts below 500 words should be skipped or upsampled (default: %(default)s)",
    )

    args = parser.parse_args()
    logging.info(f"Arguments for impostor detector: {args}")

    our_figure_impostor_options = [
        "on_the_fly",
        "in_domain",
        # "one_step_llm",   # SAIA URL on betaweb error: httpx.InvalidURL: Invalid port: 'academiccloud:de'
        # "two_step_llm",
        # "translation",
        # "mirror_minds",   # raises error
    ]

    run_prec_recall_curves(
        dataset_name=CONFIG.BLOG, imp_gen_techniques=our_figure_impostor_options
    )

    run_prec_recall_curves(
        dataset_name=CONFIG.STUDENT_ESSAYS, imp_gen_techniques=our_figure_impostor_options
    )
