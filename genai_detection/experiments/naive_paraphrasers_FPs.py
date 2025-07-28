"""
Experiment: Naive Paraphrasers and False Positives
The goal of this experiment is to find out whether Naive Paraphrasers risk False Positives because they (hypothesis) know too much about the original text.
Hence, we hypothesize if the imposter model uses only Naive Paraphrasers (i.e. T5, Ollama, etc.) to generate imposters, it will likely lead to a high number of False Positives.
We therefore created (Non-)Naive LLM-based imposter generators in the `LLMImposterGenerator` class and use only the (Non-)Naive Paraphrasers.
"""

import argparse
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
    naive_imposter_detector = ImpostorDetector(
        path2imp=Path(os.getcwd()).resolve().parent / CONFIG.PATH2BLOG,
        imposter_technique="naive_llm",  # Use only naive paraphrasers for imposter generation
    )
    non_naive_imposter_detector = ImpostorDetector(
        path2imp=Path(os.getcwd()).resolve().parent / CONFIG.PATH2BLOG,
        imposter_technique="non_naive_llm",  # Use only non-naive paraphrasers for imposter generation
    )
    generalized_imposter_detector = ImpostorDetector(
        path2imp=Path(os.getcwd()).resolve().parent / CONFIG.PATH2BLOG,
        imposter_technique="llm",  # Use all paraphrasers for imposter generation
    )
    # unmasking_detector = UnmaskingDetector()
    # ppmd_detector = PPMdDetector()
    detector_dict = {
        "naive": naive_imposter_detector,
        "non_naive": non_naive_imposter_detector,
        "generalized": generalized_imposter_detector,
    }
    return detector_dict


def get_detector_scores(
    detector_dict: dict[str, ImpostorDetector],
    dataset_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Calculate the scores for each detector on the given datasets.
    The scores are added to the datasets as new columns.
    """

    # for detector in [imposter_detector, unmasking_detector, ppmd_detector]:
    for detector_name, detector in detector_dict.items():
        for dataset_name, dataset in dataset_dict.items():
            dataset[f"{detector_name}_score"] = np.nan
            if detector_name == dataset_name:
                continue  # Skip the detector if it is the same as the dataset name, bc candidate text has same author as some imposters
            for i in tqdm(
                dataset.index,
                desc=f"Processing {detector_name} scores",
            ):
                original_text = dataset.loc[i, "disputed_text"]
                paraphrased_text = dataset.loc[i, "candidate_text"]
                score = detector.get_score(
                    [original_text, paraphrased_text], normalize=False
                )
                dataset.loc[i, f"{detector_name}_score"] = np.round(score, 2)

    dataset = pd.concat([d for d in dataset_dict.values()], ignore_index=True)

    dataset.to_csv(
        SAVE_PATH / f"exp_imp_gen_av_scores_cross_genre_dataset_{TIMESTAMP}.csv",
        index=False,
    )
    return dataset


def _get_rsme_per_paraphraser(
    dataset: pd.DataFrame,
    detector_dict: dict[str, ImpostorDetector],
) -> pd.DataFrame:
    # calculate rmse per paraphraser and AV model
    rmse_per_paraphraser = DefaultDict(dict)
    paraphraser_names = NAIVE_PARAPHRASER_NAMES + NON_NAIVE_PARAPHRASER_NAMES
    for paraphraser in dataset["paraphraser"].unique():
        if paraphraser not in paraphraser_names:
            continue  # Skip if the paraphrasers that are actually human authors
        for detector_name, detector in detector_dict.items():
            fp_for_paraphraser = dataset[
                dataset["candidate_paraphraser"] == paraphraser
            ]
            scores = fp_for_paraphraser[f"{detector_name}_score"]
            rmse = np.sqrt(np.mean((scores - fp_for_paraphraser["same"]) ** 2))
            rmse_per_paraphraser[paraphraser][detector_name] = rmse
    # to dataframe
    rmse_df = pd.DataFrame(rmse_per_paraphraser).T
    rmse_df.reset_index(inplace=True)
    rmse_df.rename(columns={"index": "candidate_paraphraser"}, inplace=True)
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

    rmse_df.to_csv(
        SAVE_PATH / f"exp_naive_paraphrasers_rmse_per_paraphraser_FPs_{TIMESTAMP}.csv",
        index=False,
    )
    print("RMSE per candidate_paraphraser among FPs:")
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

    rmse_df.to_csv(
        SAVE_PATH / f"exp_naive_paraphrasers_rmse_per_paraphraser_FNs_{TIMESTAMP}.csv",
        index=False,
    )
    print("RMSE per candidate_paraphraser among FNs:")
    return rmse_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assess effect of (Non-) Naive imposter generation."
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
    print("Scores calculated successfully.")

    fp_results = run_FPs_experiment(
        dataset=scores,
        detector_dict=detector_dict,
    )
    print("False Positives experiment completed successfully.")

    fn_results = run_FNs_experiment(
        dataset=scores,
        detector_dict=detector_dict,
    )
    print("False Negatives experiment completed successfully.")
