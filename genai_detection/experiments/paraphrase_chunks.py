"""
Experiment paraphrasing: Chunks
The goal of this experiment is to find out whether paraphrasing paragraphs or chunks (because the layout information used to identify paragraphs was stripped from the arrow datasets) is more effective paraphrasing whole texts in terms of state-of-the-art paraphrasing metrics.
We will paraphrase chunks and compute scores on chunk-paragraph level.
The scores will be averaged to get a score for the whole text, which will be compared to the score of the whole text paraphrased.
"""

import argparse
from asyncio import sleep
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import json
from pathlib import Path
import os
import textwrap
import sys
import re
from typing import DefaultDict, Dict, List
import nltk
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from datasets import load_from_disk
from genai_detection.util import preprocess_text as _preprocess_text

nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize
from genai_detection.config import CONFIG
from genai_detection.paraphrasing.paraphraser import (
    NaiveParaphraser,
    NonNaiveParaphraser,
    Paraphraser,
    get_paraphraser_dict,
)
from genai_detection.paraphrasing.paraphraser_evaluation import ParaphrasingEvaluator

CATEGORIES = [
    "Blog",
    # "News",
    "Gutenberg",
    "Student Essay",
]
PROMPTS = [
    "For the text above: Paraphrase the sentence by first identifying the main subject, verb, and object. Then find synonyms for each and construct a new sentence. Only output the final paraphrased sentence.",
    "For the text above: Paraphrase this sentence. Do not change the meaning, but use different words and structure. Output only the paraphrased sentence.",
]
SAVE_PATH = (
    Path(__file__).resolve().parents[2]
    / CONFIG.SAVE_PATH
    / "paraphrasing"
    / "experiments"
    / "chunks"
)


def get_dataset(path2dataset: str) -> pd.DataFrame:
    """
    Load the cross-genre dataset from the specified path.
    The dataset is expected to be in a format compatible with the `load_from_disk` function.

    The construction of this cross-genre dataset is in file `genai_detection/dataset_util.py`.
    """
    assert os.path.exists(path2dataset), f"Dataset path {path2dataset} does not exist."
    dataset = load_from_disk(path2dataset)["train"].to_pandas()
    dataset.drop(
        columns=["candidate_text", "same", "pair", "artificial_generation"],
        inplace=True,
    )
    print(f"Loaded Cross-genre dataset with {len(dataset)} examples.")
    return dataset


def split_text_into_chunks(text: str, n: int = 1) -> List[str]:
    """
    Splits the input text into approximately n chunks, ensuring that each chunk has a similar number
    of words. The function first tokenizes the text into sentences, then groups these sentences into
    chunks such that the total word count in each chunk is roughly even across n parts.
    :param text: The input text to be split into chunks.
    :param n: The number of chunks to split the text into. If n is greater than the number of sentences, the function will return fewer chunks.
    :return: A list of text chunks, each containing approximately the same number of words.
    """
    assert text and isinstance(text, str), "Input text must be a non-empty string."
    assert n > 0, "Number of chunks must be a positive integer."
    # Step 1: Tokenize into sentences
    sentences = sent_tokenize(text)

    # Step 2: Group sentences until total word count is ~ even across n parts
    total_words = sum(len(word_tokenize(sent)) for sent in sentences)
    target_words_per_chunk = total_words / n

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sent in sentences:
        sent_words = word_tokenize(sent)
        current_chunk.append(sent)
        current_word_count += len(sent_words)

        if current_word_count >= target_words_per_chunk and len(chunks) < n - 1:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

    # Append remaining sentences to the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


from concurrent.futures import ThreadPoolExecutor, as_completed


def paraphrase_with_config(
    paraphraser_name: str,
    chunk: str,
    n_responses: int,
    config: Dict[str, str] = {},
) -> List[Dict[str, str]]:
    paraphraser = get_paraphraser_dict()[paraphraser_name]
    rows = []
    if isinstance(paraphraser, NonNaiveParaphraser):
        prompt_options = [None]
        temperature_options = np.linspace(0.0, 1.0, num=len(PROMPTS)).tolist()
    elif isinstance(paraphraser, NaiveParaphraser):
        prompt_options = PROMPTS
        temperature_options = [None]
    else:
        raise ValueError(f"Unknown paraphraser type: {type(paraphraser)}")

    for prompt in prompt_options:
        for temperature in temperature_options:
            p_config = {
                "text": chunk,
                "prompt": prompt,
                "n_responses": n_responses,
            }
            if isinstance(paraphraser, NonNaiveParaphraser) and temperature is not None:
                p_config["temperature"] = temperature

            try:
                paraphrased_chunk = paraphraser.paraphrase(**p_config)
                updated_config = config.copy()
                updated_config.update(
                    {
                        "paraphraser": paraphraser_name,
                        "prompt": prompt,
                        "chunk": chunk,
                        "temperature": temperature,
                        "paraphrased_chunk": (
                            paraphrased_chunk[0] if paraphrased_chunk else ""
                        ),
                    }
                )
                rows.append(updated_config)
                break
            except Exception as e:
                raise e
    return rows


def create_and_save_paraphrasers(path2dataset: str, save_path: Path):
    """
    Create paraphrasers and save them to the specified path.

    :param path2dataset: Path to the cross-genre dataset.
    :param save_path: Path to save the results.
    """
    assert os.path.exists(path2dataset), f"Dataset path {path2dataset} does not exist."
    assert save_path.exists(), f"Save path {save_path} does not exist."
    file2existing_paraphrases = save_path / "existing_chunk_paraphrases.json"
    dataset = get_dataset(path2dataset)
    # keep only the first 5 examples per category
    dataset = dataset.groupby("category").head(1)  # TODO: change to 5

    n_responses = 1

    # Initialize paraphrasers and prompts
    paraphrasers = get_paraphraser_dict()

    # work on each text individually
    for i, (original_text, category) in enumerate(
        zip(dataset["disputed_text"], dataset["category"])
    ):

        print(f"Processing text {i+1}/{len(dataset)}: {category}")
        rows = []
        text_key = f"text_{i}"
        if file2existing_paraphrases.exists():
            with open(file2existing_paraphrases, "r") as f:
                data_loaded = json.load(f)
        else:
            data_loaded = {}
            data_loaded.setdefault(text_key, {})

        for num_chunks in tqdm(
            range(1, 6), desc="Evaluating with different chunk sizes"
        ):
            try:
                chunks = split_text_into_chunks(original_text, n=num_chunks)
            except Exception as e:
                print(
                    f"ERROR (paraphrase_chunks, create and save) splitting text into chunks: {e}"
                )
                continue
            print(f"Number of chunks: {num_chunks}")

            missing_configs = []

            n_total_chunks = f"n_chunks_{num_chunks}"
            data_loaded[text_key].setdefault(n_total_chunks, {})
            chunk_data_dict = data_loaded[text_key][n_total_chunks]

            for chunk_id in range(num_chunks):
                chunk_identifier = f"chunk_{chunk_id}"
                chunk_data = chunk_data_dict.setdefault(chunk_identifier, {})

                for paraphraser_name in paraphrasers.keys():
                    if paraphraser_name not in chunk_data.keys():
                        chunk_data[paraphraser_name] = {}
                        missing_configs.append(
                            [
                                text_key,
                                n_total_chunks,
                                chunk_identifier,
                                paraphraser_name,
                            ]
                        )

            if len(missing_configs) == 0:
                print(
                    f"All paraphrases for text {i} with {num_chunks} chunks already exist, skipping."
                )
                continue
            with ThreadPoolExecutor(max_workers=len(missing_configs)) as executor:
                futures = []
                for config in missing_configs:
                    paraphraser_name = config[3]
                    num_chunks = int(config[1].split("_")[-1])
                    chunk_id = int(config[2].split("_")[-1])
                    static_update_dict = {
                        "original_text": original_text,
                        "num_chunks": num_chunks,
                        "paraphraser": paraphraser_name,
                        "chunk_id": chunk_id,
                        "category": category,
                    }
                    if paraphraser_name not in [
                        "T5_ChatGPT",
                        "T5_Google_PAWS",
                        # "Ollama",
                    ]:
                        futures.append(
                            executor.submit(
                                paraphrase_with_config,
                                paraphraser_name,
                                chunks[chunk_id],
                                n_responses,
                                static_update_dict,
                            )
                        )
                for future in as_completed(futures):
                    res = future.result()
                    for paraphrase_dif_temp_prompt in res:
                        assert isinstance(
                            paraphrase_dif_temp_prompt, dict
                        ), "Result should be a dict."
                        chunk_id = paraphrase_dif_temp_prompt.pop("chunk_id")
                        paraphraser_name = paraphrase_dif_temp_prompt.pop("paraphraser")
                        data_loaded[text_key][n_total_chunks][f"chunk_{chunk_id}"][
                            paraphraser_name
                        ] = paraphrase_dif_temp_prompt

            # Sequentially iterate over paraphrasers to avoid API rate limits
            # for paraphraser_name in paraphrasers.keys():
            #     if paraphraser_name not in [
            #         "T5_ChatGPT",
            #         "T5_Google_PAWS",
            #         # "Ollama",
            #     ]:
            #         res = paraphrase_with_config(
            #             paraphraser_name,
            #             chunks,
            #             original_text,
            #             num_chunks,
            #             n_responses,
            #             category,
            #         )
            #         if res:
            #             all_rows.extend(res)

        # save results for this text to json file
        with open(file2existing_paraphrases, "w") as f:
            json.dump(data_loaded, f, indent=4)
        print(f"Saved paraphrases for text {i} to {file2existing_paraphrases}")
        # text_paraphrases_df = pd.DataFrame(rows)
        # # Save the results for this text
        # text_paraphrases_df.to_csv(save_path / f"text_{i}_paraphrases.csv", index=False)


def evaluate_paraphrases(path2dataset: str, save_path: Path) -> List[pd.DataFrame]:
    assert os.path.exists(path2dataset), f"Dataset path {path2dataset} does not exist."
    file2existing_paraphrases = save_path / "existing_chunk_paraphrases.json"
    if not file2existing_paraphrases.exists():
        raise FileNotFoundError(
            f"File with existing paraphrases {file2existing_paraphrases} does not exist."
        )
    loaded_paraphrases = json.load(open(file2existing_paraphrases, "r"))

    # iterate over all csv file containing paraphrases
    rows = []  # of dicts
    for text_id, text_data in tqdm(
        loaded_paraphrases.items(),
        desc="Evaluating paraphrases",
        total=len(loaded_paraphrases),
    ):
        # [n_total_chunks][f"chunk_{chunk_id}"][
        #                     paraphraser_name
        #                 ]
        for n_total_chunks, chunk_data_dict in text_data.items():
            for chunk_id, chunk_data in chunk_data_dict.items():
                for paraphraser_name, paraphrase_data in chunk_data.items():
                    original_text = _preprocess_text(paraphrase_data["original_text"])
                    paraphrase_evaluator = ParaphrasingEvaluator(
                        paraphrasers=get_paraphraser_dict(),
                        prompts=PROMPTS,
                        original_text=original_text,
                        n_responses=1,
                    )
                    paraphrased_chunk = _preprocess_text(
                        paraphrase_data["paraphrased_chunk"]
                    )
                    if not paraphrased_chunk or pd.isna(paraphrased_chunk):
                        print(
                            f"[WARNING] Paraphrased chunk is empty for text ID {text_id}, skipping evaluation."
                        )
                        continue

                    try:
                        # input is list of strings, each string is a paraphrase/ reference
                        bert_scores = paraphrase_evaluator.bertscore.compute(
                            predictions=[paraphrased_chunk],
                            references=[original_text],
                            model_type="distilbert-base-uncased",
                        )
                    except Exception as e:
                        print("[ERROR] BERTScore computation failed:", e)
                        bert_scores = {
                            "precision": [0.0],
                            "recall": [0.0],
                            "f1": [0.0],
                            "hashcode": "",
                        }
                    # placeholder/ fallback
                    rouge_scores = {
                        "rouge1": 0.0,
                        "rouge2": 0.0,
                        "rougeL": 0.0,
                        "rougeLsum": 0.0,
                    }
                    try:
                        rouge_scores = paraphrase_evaluator.rouge_score.compute(
                            predictions=[paraphrased_chunk], references=[original_text]
                        )

                    except Exception as e:
                        print("[ERROR] ROUGE computation failed:", e)

                    res = paraphrase_evaluator._build_result_row(
                        paraphraser_name=paraphraser_name,
                        prompt=paraphrase_data["prompt"],
                        paraphrase=paraphrased_chunk,
                        original_split=original_text.split(),
                        bert_scores=bert_scores,
                        rouge_scores=rouge_scores,
                        idx=0,
                    )
                    assert res is not None, "Result row should not be None."
                    paraphrase_data.update(res)
                    rows.append(paraphrase_data)
    df = pd.DataFrame(rows)
    # Save the results to a CSV file
    save_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path / f"text_paraphrases_evaluation_results.csv", index=False)
    return rows


def get_slim_dfs_for_one_text(n_paragraphs_df: list) -> list:  # of dataframes
    slim_n_paragraphs_dfs = []
    for i in range(len(n_paragraphs_df)):
        # each row is average score over scores of all chunks
        slim_n_paragraphs_df = n_paragraphs_df[i].drop(
            columns=[
                # "prompt",
                "original_text",
                "parameters",
                "paraphrased_text",
                "bertscore_hash",
            ]
        )
        slim_n_paragraphs_dfs.append(slim_n_paragraphs_df)
    return slim_n_paragraphs_dfs


def plot_model_metrics(
    n_paragraphs_df: pd.DataFrame,
    save_dir: str = "model_plots",
    show: bool = True,
    save: bool = True,
    figsize=(10, 6),
    data_category: str = "News",
):
    """
    Create and save line plots of metric scores per model over n_chunks.

    Parameters:
    - n_paragraphs_df: list of pandas DataFrames with columns ['model', 'prompt', 'n_chunks', <metric columns>]
    - save_dir: directory to save plots (default "model_plots")
    - show: whether to display plots using plt.show()
    - save: whether to save plots to disk
    - figsize: figure size for each plot
    """

    # Combine all DataFrames
    combined_df = pd.concat(n_paragraphs_df, ignore_index=True)

    # Identify models and metrics
    models = combined_df["model"].unique()
    metric_cols = [
        col for col in combined_df.columns if col not in ["model", "prompt", "n_chunks"]
    ]

    # Ensure save directory exists
    if save:
        os.makedirs(save_dir, exist_ok=True)

    # Get color map with enough unique colors
    colors = sns.color_palette("colorblind", len(metric_cols))
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "<", ">", "H", "|", "_"]

    # Loop over each model
    for model in models:
        df_model = combined_df[combined_df["model"] == model]

        # Group by n_chunks
        grouped = df_model.groupby("n_chunks")
        mean_df = grouped[metric_cols].mean()
        std_df = grouped[metric_cols].std()
        x = mean_df.index

        # Plot
        plt.figure(figsize=figsize)
        for i, metric in enumerate(metric_cols):
            color = colors[i % len(colors)]
            linestyle = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            plt.plot(
                x,
                mean_df[metric],
                label=metric,
                color=color,
                linestyle=linestyle,
                marker=marker,
            )
            plt.fill_between(
                x,
                mean_df[metric] - std_df[metric],
                mean_df[metric] + std_df[metric],
                color=color,
                alpha=0.3,
            )

        plt.title(f"Metrics for paraphraser: {model}")
        plt.xlabel("n_chunks")
        plt.ylabel("Score")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(
            title="Metric",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        if save:
            filename = os.path.join(
                save_dir, f"{model}_metrics_plot_category_{data_category}.svg"
            )
            plt.savefig(filename, format="svg", bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


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

    parser.add_argument(
        "--task",
        type=str,
        default="evaluate",
        help="Whether to create and save or to load an evaluate paraphrases (default: %(default)s).",
    )

    args = parser.parse_args()

    # Run the experiment
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

    paraphrase_save_path = SAVE_PATH / "cross_genre" / "paraphrases_per_text"
    paraphrase_save_path.mkdir(parents=True, exist_ok=True)

    if args.task == "create":
        # only create paraphrasers and save them
        print(f"Creating and saving paraphrasers to {paraphrase_save_path}.")

        create_and_save_paraphrasers(
            path2dataset=args.path2dataset,
            save_path=paraphrase_save_path,
        )

    elif args.task == "evaluate":
        print(
            f"Paraphrasers created and saved to {paraphrase_save_path}. Next, run the evaluation."
        )

        results = evaluate_paraphrases(
            path2dataset=paraphrase_save_path,
            save_path=SAVE_PATH / "cross_genre",
        )
        print(
            f"Evaluation results saved to {SAVE_PATH / 'cross_genre' / 'text_paraphrases_evaluation_results.csv'}."
        )
        assert type(results) is list, "Results should be a list."
        slim_df = get_slim_dfs_for_one_text(results)
        assert type(slim_df) is list, "Slim DataFrame should be a list of DataFrames."
        plot_model_metrics(
            n_paragraphs_df=slim_df,
            save_dir=SAVE_PATH / "cross_genre" / "plots",
            show=False,
            save=True,
            data_category="Cross-Genre",
        )

    # FIXME: cannot parallelize matplotlib plots
    # print("Next, plot model metrics per text (parallel).")
    # with ProcessPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(_process_df, item) for item in list(scores_per_text.items())
    #     ]
    #     for _ in tqdm(
    #         as_completed(futures),
    #         total=len(futures),
    #         desc="Plotting model metrics per text",
    #     ):
    #         pass
