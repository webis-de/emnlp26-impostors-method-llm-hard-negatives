"""
Experiment paraphrasing: Chunks
The goal of this experiment is to find out whether paraphrasing paragraphs or chunks (because the layout information used to identify paragraphs was stripped from the arrow datasets) is more effective paraphrasing whole texts in terms of state-of-the-art paraphrasing metrics.
We will paraphrase chunks and compute scores on chunk-paragraph level.
The scores will be averaged to get a score for the whole text, which will be compared to the score of the whole text paraphrased.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import os
import textwrap
import sys
import re
from typing import DefaultDict, Dict, List
import nltk
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
    T5ChatGPTParaphraser,
    T5GooglePAWSParaphraser,
    BlabladorParaphraser,
    BulletPointParaphraser,
    OllamaParaphraser,
    TaskParaphraser,
    TopicParaphraser,
    TitleParaphraser,
    TranslationParaphraser,
    Paraphraser,
)
from genai_detection.paraphrasing.paraphraser_evaluation import ParaphrasingEvaluator

CATEGORIES = [
    "Blog",
    "News",
    "Gutenberg",
    "Student Essay",
]
PROMPTS = [
    "Paraphrase the following text and output only the paraphrased version:",
    "First, extract bullet points capturing the main ideas, then create a text based on these bullet points. Only output the final text (i.e. do not output the bullet points or any additional chain of thoughts):",
    "Paraphrase the sentence by first identifying the main subject, verb, and object. Then find synonyms for each and construct a new sentence. Only output the final paraphrased sentence.",
    "Paraphrase the sentence using the same tone as the original with approximately the same number of words:",
    "Paraphrase this sentence. Do not change the meaning, but use different words and structure. Output only the paraphrased sentence:",
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


def get_paraphraser_dict() -> Dict[str, Paraphraser]:
    """
    Returns a dictionary of paraphrasers with their names as keys and instances as values.
    """
    paraphrasers = {
        "T5_ChatGPT": T5ChatGPTParaphraser(),
        "T5_Google_PAWS": T5GooglePAWSParaphraser(),
        "Ollama": OllamaParaphraser(model_id=CONFIG.OLLAMA_VERSION),
    }
    bullet_point_paraphraser = BulletPointParaphraser(
        text_extractor=paraphrasers["Ollama"],
        text_generator=paraphrasers["Ollama"],
    )
    task_paraphraser = TaskParaphraser(
        text_extractor=paraphrasers["Ollama"],
        text_generator=paraphrasers["Ollama"],
    )
    topic_paraphraser = TopicParaphraser(
        text_extractor=paraphrasers["Ollama"],
        text_generator=paraphrasers["Ollama"],
    )
    title_paraphraser = TitleParaphraser(
        text_extractor=paraphrasers["Ollama"],
        text_generator=paraphrasers["Ollama"],
    )
    translation_paraphraser = TranslationParaphraser(
        text_extractor=paraphrasers["Ollama"],
        text_generator=paraphrasers["Ollama"],
    )
    paraphrasers.update(
        {
            "BulletPoint": bullet_point_paraphraser,
            "Task": task_paraphraser,
            "Topic": topic_paraphraser,
            "Title": title_paraphraser,
            "Translation": translation_paraphraser,
        }
    )
    return paraphrasers


def init_paraphrase_evaluator(
    original_text: str,
    n_responses: int = 1,
    max_length: int = CONFIG.MAX_LENGTH,
    temperature: float = CONFIG.TEMPERATURE,
    paraphrasers: Dict[str, Paraphraser] = None,
    prompts: List[str] = PROMPTS,
):
    assert paraphrasers is not None, "Paraphrasers must be provided."
    assert prompts is not None, "Prompts must be provided."
    paraphrase_evaluator = ParaphrasingEvaluator(
        paraphrasers=paraphrasers,
        prompts=prompts,
        original_text=original_text,
        n_responses=n_responses,
        max_length=max_length,
        temperature=temperature,
    )
    return paraphrase_evaluator


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


# to parallelize
def _evaluate_chunk(
    chunk, paraphrasers, prompts, n_responses, max_len, temperature, num_chunks
):
    paraphrase_evaluator = ParaphrasingEvaluator(
        paraphrasers=paraphrasers,
        prompts=prompts,
        original_text=chunk,
        n_responses=n_responses,
        max_length=max_len,
        temperature=temperature,
    )
    df, _ = paraphrase_evaluator.evaluate(
        save_extremest_paraphr_per_score=False, save_to_disk=False
    )
    df["n_chunks"] = num_chunks
    return df


def create_and_save_paraphrasers(path2dataset: str, save_path: Path):
    """
    Create paraphrasers and save them to the specified path.

    :param path2dataset: Path to the cross-genre dataset.
    :param save_path: Path to save the results.
    """
    assert os.path.exists(path2dataset), f"Dataset path {path2dataset} does not exist."
    assert save_path.exists(), f"Save path {save_path} does not exist."
    dataset = get_dataset(path2dataset)
    n_responses = 1

    # Initialize paraphrasers and prompts
    paraphrasers = get_paraphraser_dict()

    # work on each text individually
    for i, (original_text, category) in enumerate(
        zip(dataset["disputed_text"], dataset["category"])
    ):
        # TODO: First only process Student Essays
        filter_cat = "Gutenberg"  # "Student Essays"
        if category != filter_cat:
            print(f"Skipping text {i+1}/{len(dataset)}: {category} (not {filter_cat})")
            continue
        print(f"Processing text {i+1}/{len(dataset)}: {category}")
        rows = []
        if (save_path / f"text_{i}_paraphrases.csv").exists():
            print(f"Paraphrases for text {i} already exist, skipping.")
            continue

        for num_chunks in tqdm(
            range(1, 6), desc="Evaluating with different chunk sizes"
        ):
            chunks = split_text_into_chunks(original_text, n=num_chunks)
            print(f"Number of chunks: {num_chunks}")
            for paraphraser_name, paraphraser in paraphrasers.items():
                if isinstance(paraphraser, NonNaiveParaphraser):
                    prompt_options = [None]
                    temperature_options = [0, 0.5, 1.0]
                elif isinstance(paraphraser, NaiveParaphraser):
                    prompt_options = PROMPTS
                    temperature_options = [None]
                else:
                    raise ValueError(f"Unknown paraphraser type: {type(paraphraser)}")
                for prompt in prompt_options:
                    for temperature in temperature_options:
                        print(
                            f"Using paraphraser: {paraphraser_name} with temperature={temperature}, prompt={prompt}"
                        )

                        for chunk_id, chunk in enumerate(chunks):
                            print(f"Paraphrasing chunk {chunk_id+1}/{len(chunks)}")
                            p_config = {
                                "text": chunk,
                                "prompt": prompt,
                                "n_responses": n_responses,
                            }
                            if (
                                isinstance(paraphraser, NonNaiveParaphraser)
                                and temperature is not None
                            ):
                                p_config["temperature"] = temperature
                            try:
                                paraphrased_chunk = paraphraser.paraphrase(**p_config)
                                rows.append(
                                    {
                                        "original_text": original_text,
                                        "num_chunks": num_chunks,
                                        "paraphraser": paraphraser_name,
                                        "prompt": prompt,
                                        "chunk_id": chunk_id,
                                        "chunk": chunk,
                                        "temperature": temperature,
                                        "paraphrased_chunk": (
                                            paraphrased_chunk[0]
                                            if paraphrased_chunk
                                            else ""
                                        ),
                                        "category": category,
                                    }
                                )
                            except Exception as e:
                                print(
                                    f"Error paraphrasing chunk {chunk_id+1}/{len(chunks)} with {paraphraser_name}: {e}"
                                )

        text_paraphrases_df = pd.DataFrame(rows)
        # Save the results for this text
        text_paraphrases_df.to_csv(save_path / f"text_{i}_paraphrases.csv", index=False)


def evaluate_paraphrases(
    path2dataset: str, save_path: Path
) -> Dict[str, List[pd.DataFrame]]:
    assert os.path.exists(path2dataset), f"Dataset path {path2dataset} does not exist."

    # iterate over all csv file containing paraphrases
    rows = []  # of dicts
    for paraphrases_file in tqdm(
        path2dataset.glob("*.csv"), desc="Evaluating paraphrases"
    ):
        # paraphases of one text
        df = pd.read_csv(paraphrases_file)
        text_id = paraphrases_file.stem.split("_")[1]  # e.g. text_0_paraphrases.csv
        original_text = _preprocess_text(df["original_text"].iloc[0])
        paraphrase_evaluator = ParaphrasingEvaluator(
            paraphrasers=get_paraphraser_dict(),
            prompts=PROMPTS,
            original_text=original_text,
            n_responses=1,
        )
        for i in range(len(df)):
            original_row = df.iloc[i].to_dict()
            paraphrased_chunk = _preprocess_text(original_row["paraphrased_chunk"])
            if not paraphrased_chunk or pd.isna(paraphrased_chunk):
                print(
                    f"[WARNING] Paraphrased chunk is empty for text ID {text_id}, skipping evaluation."
                )
                continue

            print(
                "Paraphrased chunk:", paraphrased_chunk, "..."
            )  # print first 100 characters
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
                # TODO: shouldnt be nan... obsolete?
                if paraphrased_chunk and original_text:
                    rouge_scores = paraphrase_evaluator.rouge_score.compute(
                        predictions=[paraphrased_chunk], references=[original_text]
                    )
                else:
                    print(
                        f"[WARNING] Paraphrased chunk or original text is empty for text ID {text_id}, skipping ROUGE evaluation."
                    )
            except Exception as e:
                print("[ERROR] ROUGE computation failed:", e)

            res = paraphrase_evaluator._build_result_row(
                name=df["paraphraser"].iloc[0],
                prompt=df["prompt"].iloc[0],
                paraphrase=paraphrased_chunk,
                original_split=original_text.split(),
                bert_scores=bert_scores,
                rouge_scores=rouge_scores,
                idx=0,
            )
            assert res is not None, "Result row should not be None."
            assert isinstance(original_row, dict), "Original row should be a dict."
            original_row.update(res)
            original_row["text_id"] = text_id
            rows.append(original_row)
    df = pd.DataFrame(rows)
    # Save the results to a CSV file
    save_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path / f"text_paraphrases_evaluation_results.csv", index=False)
    return df


def run_experiment(path2dataset: str) -> pd.DataFrame:
    """
    Run the paraphrasing experiment on the cross-genre dataset.
    The results are saved in a CSV file.

    :param path2dataset: Path to the cross-genre dataset.
    :return: A dictionary with scores for each text.
    """
    assert os.path.exists(path2dataset), f"Dataset path {path2dataset} does not exist."
    dataset = get_dataset(path2dataset)
    n_responses = 1

    # Initialize paraphrasers and prompts
    paraphrasers = get_paraphraser_dict()

    scores_per_text = DefaultDict(list)
    for original_text, category in tqdm(
        zip(dataset["disputed_text"], dataset["category"]), desc="Processing texts"
    ):
        n_paragraphs_df = []
        for num_chunks in tqdm(
            range(1, 6), desc="Evaluating with different chunk sizes"
        ):
            chunks = split_text_into_chunks(original_text, n=num_chunks)
            print(f"Number of chunks: {num_chunks}")

            res_for_chunks = []
            evaluate_fn = partial(
                _evaluate_chunk,
                paraphrasers=paraphrasers,
                prompts=PROMPTS,
                n_responses=n_responses,
                max_len=CONFIG.MAX_LENGTH,
                temperature=CONFIG.TEMPERATURE,
                num_chunks=num_chunks,
            )

            with ThreadPoolExecutor() as executor:  # not cpu bound but IO bound: Use ThreadPoolExecutor rather than ProcessPoolExecutor
                futures = [executor.submit(evaluate_fn, chunk) for chunk in chunks]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Evaluating paraphrases",
                ):
                    df = future.result()
                    res_for_chunks.append(df)

            # Combine all dataframes
            df_concat = pd.concat(res_for_chunks)

            # Separate numeric and non-numeric columns
            numeric_df = df_concat.select_dtypes(include="number")
            non_numeric_df = df_concat.select_dtypes(exclude="number")

            # Take only the first non-numeric row per group to merge back later
            non_numeric_first = non_numeric_df.groupby(level=0).first()

            # Compute mean of numeric data
            numeric_mean = numeric_df.groupby(level=0).mean()

            # Combine them back together
            averaged_df = pd.concat([non_numeric_first, numeric_mean], axis=1)

            # Add to results list
            n_paragraphs_df.append(averaged_df)
        scores_per_text[original_text] = [n_paragraphs_df, category]
    return scores_per_text


def _save_modelwise_chunk_scores(
    n_paragraphs_df: list, output_dir: str | Path, data_category: str = "News"
):
    assert isinstance(
        n_paragraphs_df, list
    ), "n_paragraphs_df must be a list of DataFrames."
    output_dir = Path(output_dir)
    assert output_dir.exists(), f"Output directory {output_dir} does not exist."
    output_dir.mkdir(parents=True, exist_ok=True)
    full_df = pd.concat(n_paragraphs_df)
    for model_name, model_df in full_df.groupby("model"):
        # Group by prompt, sort by n_chunks
        model_df_sorted = model_df.sort_values(by=["prompt", "n_chunks"])

        # Save to CSV
        file_name = f"{model_name}_chunk_scores_category_{data_category}.csv"
        model_df_sorted.to_csv(output_dir / file_name, index=False)


# to parallelize
def _save_text_chunk_score(i, text, n_paragraphs_df, data_category, output_dir):
    path2results = output_dir / f"text_{i}"
    os.makedirs(path2results, exist_ok=True)

    with open(path2results / "text.txt", "w") as text_file:
        text_file.write(text)

    _save_modelwise_chunk_scores(
        n_paragraphs_df, output_dir=path2results, data_category=data_category
    )


def save_textwise_chunk_scores(scores_per_text: dict, output_dir: str | Path):
    """
    Save the chunk scores for each text in a separate CSV file.
    """
    assert isinstance(scores_per_text, dict), "scores_per_text must be a dictionary."
    output_dir = Path(output_dir)
    assert output_dir.exists(), f"Output directory {output_dir} does not exist."
    output_dir.mkdir(parents=True, exist_ok=True)
    items = [
        (i, k, v[0], v[1], output_dir)
        for i, (k, v) in enumerate(scores_per_text.items())
    ]
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_save_text_chunk_score, *item) for item in items]
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="Saving modelwise scores"
        ):
            pass


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


#### only for parallel processing
def _process_df(args):
    text, (n_paragraphs_df, data_category) = args
    slim_n_paragraphs_dfs = get_slim_dfs_for_one_text(n_paragraphs_df)
    plot_model_metrics(
        slim_n_paragraphs_dfs,
        save_dir=SAVE_PATH,
        data_category=data_category,
        show=False,
    )


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
        print(results.head())

    # scores_per_text = run_experiment(
    #     path2dataset=args.path2dataset,
    # )

    # print(
    #     f"Obtained scores for {len(scores_per_text)} texts. Next, save them in parallel fashion."
    # )
    # save_textwise_chunk_scores(scores_per_text=scores_per_text, output_dir=SAVE_PATH)

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
