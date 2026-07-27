import argparse
import logging

import ray

from genai_detection.config import CONFIG
from genai_detection.detectors.impostor import ImpostorDetector


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    ray.init()

    parser = argparse.ArgumentParser(
        description="Run the Impostor Detector on document pairs."
    )

    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--top_n", type=int, default=100000)
    parser.add_argument("--n_impostors", type=int, default=50)

    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=[CONFIG.STUDENT_ESSAYS, CONFIG.BLOG],
        default=CONFIG.STUDENT_ESSAYS,
    )

    parser.add_argument(
        "--impostor_technique",
        type=str,
        choices=[
            "translation",
            "on-the-fly",
            "on_the_fly_chatnoir",
            "on_the_fly_serpapi",
            "on_the_fly_startpage",
            "in_domain",
            "one_step_llm",
            "two_step_llm",
            "random_words",
            "mirror_minds",
        ],
        default="in_domain",
    )

    parser.add_argument(
        "--upsample",
        action="store_true",
        help="Upsample texts below min token threshold",
    )

    parser.add_argument(
        "--input_document_ids",
        nargs="+",
        required=True,
        help="Document IDs to score (must be pairs)",
    )

    args = parser.parse_args()
    logging.info("Arguments: %s", args)

    detector = ImpostorDetector(
        impostor_technique=args.impostor_technique,
        n_impostors=args.n_impostors,
        upsample=args.upsample,
        rounds=args.rounds,
        top_n=args.top_n,
        dataset_name=args.dataset_name,
    )

    scores = detector.get_score(text=args.input_document_ids)
    logging.info("Scores: %s", scores)

# ray job submit --address https://ray.srv.webis.de --working-dir . --runtime-env env.yml -- python genai_detection/cli/run_impostor.py --input_document_ids 68f50029edacdf3d5c0279dd 68f50029edacdf3d5c0279d9

# generating 1 x 50 impostors takes around 40 minutes using openai.
# ..dd: Ass4 and author TDH426, ..4f: Ass3 and author TDH426, ...e8: Ass4 and author ASR497
# doc_pairs = ["68f50029edacdf3d5c0279dd", #"68f50029edacdf3d5c027d4f"]#, "68f50029edacdf3d5c0279e8"]#,
# "68f50029edacdf3d5c0279eb"]
# "68f50029edacdf3d5c0279d9"]
# "68f50029edacdf3d5c0279e0"]

if __name__ == "__main__":
    main()
