from pathlib import Path
import os
import textwrap
import sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the root of the project to Python path
# sys.path.append(os.path.abspath(".."))
print("Current working directory:", os.getcwd())
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
)
from genai_detection.paraphrasing.paraphraser_evaluation import ParaphrasingEvaluator

OLLAMA_VERSION = "zephyr:7b"  # "mistral:7b"  # "default:latest"
N_RESPONSES = 1
MAX_LENGTH = 512
TEMPERATURE = 0.7
PROMPTS = [
    "Paraphrase the text above and output only the paraphrased version.",
    "For the text above: First, extract bullet points capturing the main ideas, then create a text based on these bullet points. Only output the final text (i.e. do not output the bullet points or any additional chain of thoughts).",
    "For the text above: Paraphrase the sentence by first identifying the main subject, verb, and object. Then find synonyms for each and construct a new sentence. Only output the final paraphrased sentence.",
    "For the text above: Paraphrase the sentence using the same tone as the original with approximately the same number of words.",
    "For the text above: Paraphrase this sentence. Do not change the meaning, but use different words and structure. Output only the paraphrased sentence.",
]

CATEGORY2DIRECTORY = {
    "News": [
        "custom_texts",
        "cnn_040725",
    ],  # Dalai Lama; "cnn_230625"    # USA attacks Iran
    "Blog": ["Blog_corpus", 27],
    "Gutenberg": ["gutenberg", "Othello_the_Moor_of_Venice_William_Shakespeare"],
    "Student Essays": [
        "student_essays/Intro2006",
        "Ass1/2006_AB4847",
    ],  # stream of consciousness essay: 18 years old girl who writes about her bipolar ex-boyfriend of 8 months and her new boyfriend who loves music like her but is not catholic; she wants to live in the present
}


def get_base_paths():
    data_root = Path(os.getcwd()).resolve() / "data" / "datasets"
    save_path = Path(os.getcwd()).resolve() / CONFIG.SAVE_PATH / "paraphrasing"
    save_path.mkdir(parents=True, exist_ok=True)
    assert data_root.exists(), f"Data root path {data_root} does not exist."
    return data_root, save_path


def load_text(data_category, path, file_name):
    if data_category == "Blog":
        df = pd.read_csv(path / "blogtext.csv")
        return df.loc[file_name, "text"]
    return open(path / f"{file_name}.txt", encoding="utf-8").read()


def load_metadata(path, file_name):
    try:
        df = pd.read_excel(path / "file_metadata.xlsx", index_col="filename")
        return df.loc[file_name].dropna().to_dict()
    except Exception:
        return {}


def create_paraphrasers(model):
    return {
        "T5_ChatGPT": T5ChatGPTParaphraser(),
        "T5_Google_PAWS": T5GooglePAWSParaphraser(),
        "Ollama": model,
        "BulletPoint": BulletPointParaphraser(model, model),
        "Task": TaskParaphraser(model, model),
        "Topic": TopicParaphraser(model, model),
        "Title": TitleParaphraser(model, model),
        "Translation": TranslationParaphraser(model, model),
    }


def evaluate_category(data_category, data_root, save_path):
    print(f"Evaluating category: {data_category}")
    try:
        dir_name, file_name = CATEGORY2DIRECTORY[data_category]
        path2datasets = data_root / dir_name
        if not path2datasets.exists():
            raise FileNotFoundError(f"Missing dataset path: {path2datasets}")

        use_ground_truth = not (data_category in ["Blog", "Student Essays"])
        metadata = load_metadata(path2datasets, file_name) if use_ground_truth else {}
        text = load_text(data_category, path2datasets, file_name)
        model = OllamaParaphraser(model_id=OLLAMA_VERSION)
        print(f"Loaded ollama model: {model}")
        paraphrasers = create_paraphrasers(model)
        print(f"Loaded paraphrasers for {data_category}")

        evaluator = ParaphrasingEvaluator(
            paraphrasers=paraphrasers,
            prompts=PROMPTS,
            original_text=text,
            n_responses=N_RESPONSES,
            max_length=MAX_LENGTH,
            temperature=TEMPERATURE,
            ground_truth=metadata,
            data_category=data_category,
            original_file_name=file_name,
        )
        print(f"Starting evaluation for {data_category}")

        df, extremest = evaluator.evaluate(
            save_extremest_paraphr_per_score=True, save_to_disk=True
        )

        print("Starting plotting for", data_category)
        for group in ["model", "prompt"]:
            evaluator.plot_models_metrics(
                df, save_path, data_category, group_by=group, display_plot=False
            )
            print(f"Plot metrics for {data_category} by {group}: metrics plotted")
            evaluator.plot_metric_scatter(
                df, save_path, data_category, group_by=group, display_plot=False
            )
            print(
                f"Plot metric scatter for {data_category} by {group}: scatter plotted"
            )
            evaluator.plot_metric_distributions(
                df, save_path, data_category, group_by=group, display_plot=False
            )
            print(
                f"Plot metric distributions for {data_category} by {group}: distributions plotted"
            )

        return f"Completed: {data_category}"

    except Exception as e:
        return f"Failed: {data_category} due to {e}"


def run_extraction_evaluation(save_path):
    model = OllamaParaphraser(model_id=OLLAMA_VERSION)
    paraphrasers = {
        "TopicParaphraser": TopicParaphraser(model, model),
        "TaskParaphraser": TaskParaphraser(model, model),
        "TitleParaphraser": TitleParaphraser(model, model),
        "BulletPointParaphraser": BulletPointParaphraser(model, model),
    }

    evaluator = ParaphrasingEvaluator(
        paraphrasers=paraphrasers,
        prompts=[
            "Paraphrase the following text and output only the paraphrased version:"
        ],
        original_text="not used",
        n_responses=3,
        max_length=MAX_LENGTH,
        temperature=TEMPERATURE,
        config={"save_path": save_path},
    )
    print("Starting extraction evaluation")
    evaluator.evaluate_extractors(save_to_disk=True, display_plot=False)
    print("Extraction evaluation complete")


def run_evaluation():
    data_root, save_path = get_base_paths()
    print(f"Data root: {data_root}, Save path: {save_path}")
    results = []

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(evaluate_category, category, data_root, save_path): category
            for category in CATEGORY2DIRECTORY
        }
        for future in as_completed(futures):
            result = future.result()
            print(result)
            results.append(result)

    run_extraction_evaluation(save_path)

    return results


if __name__ == "__main__":
    run_evaluation()
