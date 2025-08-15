"""
Experiment: Naive Paraphrasers and False Positives
The goal of this experiment is to find out whether Naive Paraphrasers risk False Positives because they (hypothesis) know too much about the original text.
Hence, we hypothesize if the impostor model uses only Naive Paraphrasers (i.e. T5, Ollama, etc.) to generate impostors, it will likely lead to a high number of False Positives.
We therefore created (Non-)Naive LLM-based impostor generators in the `LLMImpostorGenerator` class and use only the (Non-)Naive Paraphrasers.
"""

import argparse
from asyncio import sleep
import asyncio
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
    # "T5_ChatGPT",
    # "T5_Google_PAWS",
    # "Ollama",
    "qwen3-32b",
    "mistral-large-instruct",
    "openai-gpt-oss-120b",
    "meta-llama-3.1-8b-instruct",
    "meta-llama/Llama-3.3-70B-Instruct" "mistralai/Mixtral-8x7B-Instruct-v0.1",
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


def load_detectors(detector_name: str = "all") -> dict[str, ImpostorDetector]:
    """
    Load the detectors for the experiment.
    The detectors are expected to be in the `genai_detection.detectors` module.
    """
    naive_impostor_detector = ImpostorDetector(
        path2imp=Path(os.getcwd()).resolve().parent / CONFIG.PATH2BLOG,
        impostor_technique="naive_llm",  # Use only naive paraphrasers for impostor generation
    )
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
        "naive_llm": naive_impostor_detector,
        "non_naive_llm": non_naive_impostor_detector,
        "llm": generalized_impostor_detector,
    }
    if detector_name != "all":
        assert (
            detector_name in detector_dict.keys()
        ), f"Detector {detector_name} not found in {list(detector_dict.keys())}."
        return {detector_name: detector_dict[detector_name]}
    return detector_dict


def _get_rsme_per_img_gen(
    dataset: pd.DataFrame,
    detector_dict: dict[str, ImpostorDetector],
) -> pd.DataFrame:
    """
    Calculate the RMSE (Root Mean Square Error) for each impostor generator (naive llm, non-naive llm or all lmm) in the dataset.
    The dataset is expected to have a column 'impostor_generator' that indicates the type of generator used
    and a column 'same' that indicates whether the text is the same (1) or not (0).
    The RMSE is calculated as the square root of the mean of the squared differences between the scores and the 'same' column.
    """
    print(
        "Calculating RMSE per paraphraser and detector...",
        dataset.columns,
        detector_dict.keys(),
    )
    rmse_per_imp_gen = DefaultDict(dict)
    for imp_gen in detector_dict.keys():
        fp_for_imp_gen = dataset[dataset["impostor_generator"] == imp_gen]
        if fp_for_imp_gen.empty:
            print(f"No data found for impostor generation with {imp_gen}. Skipping.")
            continue
        print("Juhee", fp_for_imp_gen.columns, fp_for_imp_gen)

        scores = fp_for_imp_gen[f"impostor_score_{imp_gen}"]
        print(f"scores calculated with {imp_gen} detector", scores)
        rmse = np.sqrt(np.mean((scores - fp_for_imp_gen["same"]) ** 2))
        rmse_per_imp_gen[imp_gen] = rmse
    # to dataframe
    rmse_df = pd.DataFrame(rmse_per_imp_gen).T
    rmse_df.reset_index(inplace=True)
    rmse_df.rename(columns={"index": "impostor_generator"}, inplace=True)
    print("RMSE per impostor_generator:")
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

    rmse_df = _get_rsme_per_img_gen(fp_dataset, detector_dict)
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

    rmse_df = _get_rsme_per_img_gen(fn_dataset, detector_dict)
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


def read_scores_from_csv(path2csv: Path) -> pd.DataFrame:
    """
    Read the scores from the CSV file.
    The CSV file is expected to contain the scores for each detector on the dataset.
    """
    if not path2csv.exists():
        raise FileNotFoundError(f"CSV file {path2csv} does not exist.")
    df = pd.DataFrame()
    for filename in os.listdir(path2csv):
        if not filename.endswith(".csv"):
            continue
        path2csvfile = path2csv / filename
        csv_df = pd.read_csv(path2csvfile)
        if not csv_df.empty:
            csv_df["impostor_generator"] = filename.split("_")[
                1
            ]  # Extract generator name
            print(f"Reading {len(csv_df)} rows from {path2csvfile}")
            df = pd.concat([df, csv_df], ignore_index=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assess effect of (Non-) Naive impostor generation."
    )
    parser.add_argument(
        "--path2dataset",
        type=str,
        default=Path(__file__).resolve().parents[2] / CONFIG.PATH2CROSS_GENRE,
        help="Path to cross-genre dataset (default: %(default)s).",
    )
    args = parser.parse_args()

    # only read from existing impostor scores for different impostor generators
    # do not compute scores again, hence read-only!
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

    detector_dict = load_detectors()
    print("Detectors loaded successfully.")

    path2csv = (
        Path(__file__).resolve().parents[2]
        / CONFIG.SAVE_PATH
        / "impostor_scores"
        / "experiments"
        / "impostor_generator_comparison"
    )
    scores = read_scores_from_csv(path2csv=path2csv)
    print(
        f"Scores read from disk {path2csv} successfully. Number of rows: {len(scores)}"
    )

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
