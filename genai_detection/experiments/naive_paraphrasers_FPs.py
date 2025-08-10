"""
Experiment: Naive Paraphrasers and False Positives
The goal of this experiment is to find out whether Naive Paraphrasers risk False Positives because they (hypothesis) know too much about the original text.
Hence, we hypothesize if the impostor model uses only Naive Paraphrasers (i.e. T5, Ollama, etc.) to generate impostors, it will likely lead to a high number of False Positives.
We therefore created (Non-)Naive LLM-based impostor generators in the `LLMImpostorGenerator` class and use only the (Non-)Naive Paraphrasers.
"""

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from typing import DefaultDict
from genai_detection.config import CONFIG
from datasets import load_from_disk
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from genai_detection.detectors.impostor import ImpostorDetector

# from genai_detection.detectors.unmasking import UnmaskingDetector
# from genai_detection.detectors.ppmd import PPMdDetector
import pandas as pd

NAIVE_PARAPHRASER_NAMES = [
    "T5_ChatGPT",
    "T5_Google_PAWS",
    "Ollama",
]
NON_NAIVE_PARAPHRASER_NAMES = [
    "BulletPoint",
    "Task",
    "Topic",
    "Title",
    "Translation",
]
SAVE_PATH = (
    Path(__file__).resolve().parents[2]
    / CONFIG.SAVE_PATH
    / "paraphrasing"
    / "experiments"
    / "naive_paraphrasers_FPs"
)
TIMESTAMP = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_dataset(path2dataset: str) -> pd.DataFrame:
    """
    Load the cross-genre dataset from the specified path.
    The dataset is expected to be in a format compatible with the `load_from_disk` function.

    The construction of this cross-genre dataset is in file `genai_detection/dataset_util.py`.
    """
    assert os.path.exists(path2dataset), f"Dataset path {path2dataset} does not exist."
    dataset = load_from_disk(path2dataset)["train"].to_pandas()
    print(f"Loaded Cross-genre dataset with {len(dataset)} examples.")
    dataset["paraphraser"] = dataset["authors"].apply(lambda x: x[1])
    return dataset


def split_dataset_by_paraphraser_naivety(
    dataset: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Split the dataset into two subsets based on the paraphraser type.
    Naive paraphrasers are those that are created in one step and non-naive paraphrasers are those that are created in a two-step approach.
    """
    # versions without naive/ non-naive paraphrasers in authors
    naive_p_dataset = dataset[
        ~dataset["paraphraser"].isin(NON_NAIVE_PARAPHRASER_NAMES)
    ]  # contains only naive paraphrasers
    non_naive_p_dataset = dataset[
        ~dataset["paraphraser"].isin(NAIVE_PARAPHRASER_NAMES)
    ]  # contains only non-naive paraphrasers
    print(
        f"len(non_naive_p_dataset) examples with non-naive paraphrasers: {len(non_naive_p_dataset)}"
    )
    print(
        f"len(naive_p_dataset) examples with naive paraphrasers: {len(naive_p_dataset)}"
    )
    dataset_dict = {
        "naive": naive_p_dataset,
        "non_naive": non_naive_p_dataset,
    }
    return dataset_dict


def load_detectors() -> dict[str, ImpostorDetector]:
    """
    Load the detectors for the experiment.
    The detectors are expected to be in the `genai_detection.detectors` module.
    """
    # TODO: only commented to fix error with non-naive paraphrasers
    # naive_impostor_detector = ImpostorDetector(
    #     path2imp=Path(os.getcwd()).resolve().parent / CONFIG.PATH2BLOG,
    #     impostor_technique="naive_llm",  # Use only naive paraphrasers for impostor generation
    # )
    non_naive_impostor_detector = ImpostorDetector(
        path2imp=Path(os.getcwd()).resolve().parent / CONFIG.PATH2BLOG,
        impostor_technique="non_naive_llm",  # Use only non-naive paraphrasers for impostor generation
    )
    generalized_impostor_detector = ImpostorDetector(
        path2imp=Path(os.getcwd()).resolve().parent / CONFIG.PATH2BLOG,
        impostor_technique="llm",  # Use all paraphrasers for impostor generation
    )
    # unmasking_detector = UnmaskingDetector()
    # ppmd_detector = PPMdDetector()
    detector_dict = {
        # "naive": naive_impostor_detector,
        "non_naive": non_naive_impostor_detector,
        "generalized": generalized_impostor_detector,
    }
    return detector_dict


# for parallelization
def _compute_score(detector, row: collections.OrderedDict) -> tuple[int, float]:
    original_text = row["disputed_text"]
    paraphrased_text = row["candidate_text"]
    score = detector.get_score([original_text, paraphrased_text], normalize=False)
    try:
        id = row["Index"]
    except Exception as e:
        print(f"Error getting index from row: {e}")
        print(f"Row keys: {row.keys()}")
    try:
        return id, np.round(score, 2)  # return index + score
    except TypeError as e:
        raise TypeError(
            f"Error rounding score: {score}. Ensure the score is a number. Row: {row}. Error: {str(e)}"
        ) from e


def get_detector_scores(
    detector_dict: dict[str, ImpostorDetector],
    dataset_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Calculate the scores for each detector on the given datasets.
    The scores are added to the datasets as new columns.
    """
    all_dfs = []
    detector_scores_save_path = SAVE_PATH / "detector_scores"
    detector_scores_save_path.mkdir(parents=True, exist_ok=True)

    for detector_name, detector in detector_dict.items():
        for dataset_name, dataset in dataset_dict.items():
            if detector_name == dataset_name:
                continue  # Skip the detector if it is the same as the dataset name, bc candidate text has same author as some impostors

            # Skip if already computed
            output_file = (
                detector_scores_save_path
                / f"scores_{detector_name}_on_{dataset_name}.csv"
            )
            if output_file.exists():
                print(f"Loading existing file: {output_file}")
                dataset_scored = pd.read_csv(output_file)

            else:
                score_col = f"{detector_name}_score"
                dataset_scored = dataset.copy()
                dataset_scored[score_col] = np.nan
                rows = list(dataset_scored.itertuples())  # Faster + safer for indexing
                # FIXME: Paraphrasers multiple times given (borrowed): Maybe multiple inits?
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(_compute_score, detector, row._asdict())
                        for row in rows
                    ]
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Scoring {detector_name} on {dataset_name}",
                    ):
                        idx, score = future.result()
                        dataset.at[idx, score_col] = score

                # FIXME: try sequential processing for simplicity
                # for row in rows:
                #     idx, score = _compute_score(detector, row._asdict())
                #     try:
                #         dataset.at[idx, score_col] = score
                #     except Exception as e:
                #         print(
                #             f"Error setting score for index {idx} in {detector_name} on {dataset_name}: {e}"
                #         )
                #         print(f"Row: {row._asdict()}")
                #         continue

                dataset_scored.to_csv(output_file, index=False)
        all_dfs.append(dataset_scored)
    final_dataset = pd.concat(all_dfs, ignore_index=True)

    final_output_path = (
        SAVE_PATH / f"exp_imp_gen_av_scores_cross_genre_dataset_{TIMESTAMP}.csv"
    )
    final_dataset.to_csv(
        final_output_path,
        index=False,
    )
    print(f"Saved final dataset to: {final_output_path}")
    return final_dataset


def _get_rsme_per_paraphraser(
    dataset: pd.DataFrame,
    detector_dict: dict[str, ImpostorDetector],
) -> pd.DataFrame:
    # calculate rmse per paraphraser and AV model
    print(
        "Calculating RMSE per paraphraser and detector...",
        dataset.columns,
        detector_dict.keys(),
    )
    rmse_per_paraphraser = DefaultDict(dict)
    paraphraser_names = NAIVE_PARAPHRASER_NAMES + NON_NAIVE_PARAPHRASER_NAMES
    if not any(dataset["paraphraser"].isin(paraphraser_names)):
        return None
    for paraphraser in dataset["paraphraser"].unique():
        if paraphraser not in paraphraser_names:
            print(
                f"Skipping paraphraser {paraphraser} as it is not in the expected list."
            )
            continue  # Skip if the paraphrasers that are actually human authors
        print(f"Calculating RMSE for paraphraser: {paraphraser}")
        for detector_name, detector in detector_dict.items():
            fp_for_paraphraser = dataset[dataset["paraphraser"] == paraphraser]
            print(fp_for_paraphraser)
            assert (
                not fp_for_paraphraser.empty
            ), f"No data found for paraphraser {paraphraser} with detector {detector_name}."
            scores = fp_for_paraphraser[f"{detector_name}_score"]
            rmse = np.sqrt(np.mean((scores - fp_for_paraphraser["same"]) ** 2))
            rmse_per_paraphraser[paraphraser][detector_name] = rmse
    # to dataframe
    rmse_df = pd.DataFrame(rmse_per_paraphraser).T
    rmse_df.reset_index(inplace=True)
    rmse_df.rename(columns={"index": "paraphraser"}, inplace=True)
    print("RMSE per paraphraser and detector:")
    print(rmse_df)
    return rmse_df


def run_FPs_experiment(
    dataset: pd.DataFrame,
    detector_dict: dict[str, ImpostorDetector],
) -> pd.DataFrame:
    """
    Run the experiment to find False Positives (FPs) and False Negatives (FNs) for each detector.
    The results are saved in a CSV file.
    """
    # check for FPs
    fp_dataset = dataset[~dataset["same"]]

    rmse_df = _get_rsme_per_paraphraser(fp_dataset, detector_dict)
    if rmse_df is None or rmse_df.empty:
        print("No RMSE data found for FPs. Check the dataset and detectors.")
        return pd.DataFrame()

    rmse_df.to_csv(
        SAVE_PATH / f"exp_naive_paraphrasers_rmse_per_paraphraser_FPs_{TIMESTAMP}.csv",
        index=False,
    )
    print(
        "Saved FPs RMSE to CSV: ",
        SAVE_PATH / f"exp_naive_paraphrasers_rmse_per_paraphraser_FPs_{TIMESTAMP}.csv",
    )
    print("RMSE per candidate paraphraser among FPs:", rmse_df)
    return rmse_df


def run_FNs_experiment(
    dataset: pd.DataFrame,
    detector_dict: dict[str, ImpostorDetector],
) -> pd.DataFrame:
    """
    Run the experiment to find False Negatives (FNs) for each detector.
    The results are saved in a CSV file.
    """
    # Check for FNs
    fn_dataset = dataset[dataset["same"]]

    rmse_df = _get_rsme_per_paraphraser(fn_dataset, detector_dict)
    if rmse_df is None or rmse_df.empty:
        print("No RMSE data found for FPs. Check the dataset and detectors.")
        return pd.DataFrame()

    rmse_df.to_csv(
        SAVE_PATH / f"exp_naive_paraphrasers_rmse_per_paraphraser_FNs_{TIMESTAMP}.csv",
        index=False,
    )
    print(
        "Saved FNs RMSE to CSV: ",
        SAVE_PATH / f"exp_naive_paraphrasers_rmse_per_paraphraser_FNs_{TIMESTAMP}.csv",
    )
    print("RMSE per candidate paraphraser among FNs:", rmse_df)
    return rmse_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assess effect of (Non-) Naive impostor generation."
    )
    parser.add_argument(
        "--path2dataset",
        type=str,
        default=Path(__file__).resolve().parents[2] / CONFIG.CROSS_GENRE,
        help="Path to cross-genre dataset (default: %(default)s).",
    )
    args = parser.parse_args()

    # Run the experiment
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset(path2dataset=args.path2dataset)
    dataset_dict = split_dataset_by_paraphraser_naivety(dataset=dataset)
    print("Dataset acquistion completed successfully.")

    detector_dict = load_detectors()
    print("Detectors loaded successfully.")

    scores = get_detector_scores(detector_dict=detector_dict, dataset_dict=dataset_dict)
    print("Scores calculated successfully (parallel).")

    def run_fp():
        return "FP", run_FPs_experiment(dataset=scores, detector_dict=detector_dict)

    def run_fn():
        return "FN", run_FNs_experiment(dataset=scores, detector_dict=detector_dict)

    results = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_fp), executor.submit(run_fn)]
        for future in as_completed(futures):
            label, result = future.result()
            results[label] = result
            print(f"{label} experiment completed successfully.")

    fp_results = results["FP"]
    print("FP results:", fp_results)
    fn_results = results["FN"]
    print("FN results:", fn_results)
