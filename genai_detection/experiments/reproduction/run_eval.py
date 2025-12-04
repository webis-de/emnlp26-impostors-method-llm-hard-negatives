import argparse

from genai_detection.experiments.reproduction.prec_recall_curves import *

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )


if __name__ == "__main__":
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
        default=CONFIG.STUDENT_ESSAYS,
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
        "one_step_llm",
        "two_step_llm",
        "translation",
        # "mirror_minds",   # raises error
    ]

    results_dict = compute_prec_recall_f1_acc_dict(dataset_name=args.dataset_name, imp_gen_techniques=our_figure_impostor_options)
    logging.info(f"results_dict: {results_dict}")

    # results_dict: {approach_name: DataFrame}
    dfs = []
    for approach, df in results_dict.items():
        temp_df = df.copy()
        temp_df["approach"] = approach
        dfs.append(temp_df)

    # Combine all approaches
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save to CSV
    combined_df.to_csv(LOCAL_SAVE_PATH / "effectiveness_scores.csv", index=False)
    logging.info(f"Saved effectiveness scores as csv to {LOCAL_SAVE_PATH}/effectiveness_scores.csv.")

    plot_precision_recall_curve(results=results_dict, dataset_name=args.dataset_name)
