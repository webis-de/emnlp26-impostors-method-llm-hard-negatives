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

nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize
from genai_detection.config import CONFIG
from genai_detection.paraphrasing.paraphraser import (
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


def run_experiment(path2dataset: str) -> pd.DataFrame:
    """
    Run the paraphrasing experiment on the cross-genre dataset.
    The results are saved in a CSV file.
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
    args = parser.parse_args()

    # Run the experiment
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
    scores_per_text = run_experiment(
        path2dataset=args.path2dataset,
    )

    print(
        f"Obtained scores for {len(scores_per_text)} texts. Next, save them in parallel fashion."
    )
    save_textwise_chunk_scores(scores_per_text=scores_per_text, output_dir=SAVE_PATH)

    print("Next, plot model metrics per text (parallel).")
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_process_df, item) for item in list(scores_per_text.items())
        ]
        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Plotting model metrics per text",
        ):
            pass
