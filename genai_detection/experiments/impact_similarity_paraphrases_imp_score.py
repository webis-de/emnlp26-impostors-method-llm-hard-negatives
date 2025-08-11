import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.metrics import roc_curve
from genai_detection.config import CONFIG
from genai_detection.detectors.impostor import ImpostorDetector
from nltk.translate import bleu_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

SAVE_PATH = (
    Path(__file__).resolve().parents[2]
    / CONFIG.SAVE_PATH
    / "impostor_scores"
    / "experiments"
    / "paraphrase_similarity_imp_score_impact"
)
rouge = evaluate.load("rouge")


def get_dataset(path2dataset: str, split: str = "train") -> pd.DataFrame:
    """
    Load the cross-genre dataset from the specified path.
    The dataset is expected to be in a format compatible with the `load_from_disk` function.

    The construction of this cross-genre dataset is in file `genai_detection/dataset_util.py`.
    """
    assert os.path.exists(path2dataset), f"Dataset path {path2dataset} does not exist."
    dataset = load_from_disk(path2dataset)[split].to_pandas()
    print(f"Loaded dataset with {len(dataset)} examples.")
    return dataset


def _syn_sim(text1: str, text2: str):
    """
    Compute the syntactic similarity between two texts.
    This is a placeholder function and should be replaced with an actual implementation.
    """
    assert isinstance(text1, str) and isinstance(
        text2, str
    ), "Both inputs must be strings."
    bleu_val = bleu_score.sentence_bleu(  # in [0, 1]
        references=[text1.split()],
        hypothesis=text2.split(),
        smoothing_function=bleu_score.SmoothingFunction().method1,
    )
    rouge_val = rouge.compute(predictions=[text1], references=[text2])
    return np.mean([bleu_val, rouge_val["rouge1"], rouge_val["rougeL"]])


def _get_opt_imp_threshold(fpr, tpr, thresholds):
    """
    Computes the optimal threshold using Youden's J statistic.
    Youden's J statistic is used to select the optimal predicted probability cut off.
    It is the maximum vertical distance between ROC curve and diagonal line, where the idea is to maximise the difference between true positive rate (TPR) and false positive rate (FPR).
    Youden's J statistic is defined as J = TPR + TNR - 1 = TPR + (−FPR) = TPR - FPR.

    For more information see:
    - https://www.ibm.com/docs/en/spss-statistics/30.0.0?topic=schemes-area-under-curve (12.06.2025).
    - https://en.wikipedia.org/wiki/Youden%27s_J_statistic (12.06.2025).
    If none of the values are valid, it defaults to 0.5.

    :param fpr: False Positive Rate.
    :param tpr: True Positive Rate.
    :param thresholds: Thresholds used to compute fpr and tpr.
    :return: Optimal threshold.
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    thresholds = np.asarray(thresholds)

    # validity mask: finite and within (0, 1)
    valid_mask = (
        np.isfinite(fpr)
        & np.isfinite(tpr)
        & np.isfinite(thresholds)
        & (fpr > 0)  # do not include 0 and 1 to avoid best threshold being 0 or 1
        & (fpr < 1)
        & (tpr > 0)
        & (tpr < 1)
    )

    # check if there are any valid values
    if np.any(valid_mask):
        fpr_valid = fpr[valid_mask]
        tpr_valid = tpr[valid_mask]
        thresholds_valid = thresholds[valid_mask]

        # maximum vertical distance between ROC curve and diagonal line
        youden_j = tpr_valid - fpr_valid
        optimal_idx = np.argmax(youden_j)
        return thresholds_valid[optimal_idx]
    else:
        # fallback: No valid data
        print("Warning: No valid TPR/FPR data available. Defaulting to threshold = 0.5")
        return 0.5  # or np.nan, depending on your use case


def _avg_sim_ref_paraphrases(impostor_entry):
    """Compute average sim_sim between reference_text and each paraphrase."""
    if not impostor_entry or "text_pair_0" not in impostor_entry:
        return None

    scores = []
    for j, sub_entry in impostor_entry["text_pair_0"].items():
        ref_text = sub_entry["reference_text"]
        paraphrases = sub_entry["paraphrases"].values()

        for para in paraphrases:
            scores.append(_syn_sim(ref_text, para))

    return sum(scores) / len(scores) if scores else None


def _helper_impostor(path2imp, pair, training_mode=True):
    impostor_detector = ImpostorDetector(
        impostor_technique="fixed",
        n_impostors=50,
        rounds=100,  # cf. pg. 181, Koppel et al. (2014)
        top_n=100000,  # cf. pg. 179, Koppel et al. (2014)
        path2imp=path2imp,
        upsample=False,
    )
    impostor_detector.set_training_mode(training_mode)
    return impostor_detector._get_score_impl(pair)


def create_df(path2dataset: str, dataset_name: str, save_path: Path):
    train_dataset = get_dataset(path2dataset, split="train")
    test_dataset = get_dataset(path2dataset, split="test")
    train_dataset = pd.concat(
        [
            train_dataset.loc[train_dataset["same"]].head(15),
            train_dataset.loc[~train_dataset["same"]].head(15),
        ],
        ignore_index=False,
    )

    test_dataset = pd.concat(
        [
            test_dataset.loc[test_dataset["same"]].head(15),
            test_dataset.loc[~test_dataset["same"]].head(15),
        ],
        ignore_index=False,
    )
    path2imp = (
        Path(os.getcwd()).resolve() / CONFIG.PATH2BLOG
        if dataset_name == CONFIG.BLOG
        else Path(os.getcwd()).resolve() / CONFIG.PATH2STUDENT_ESSAYS
    )

    print(f"Running experiment for {dataset_name} dataset.")

    with ThreadPoolExecutor() as executor:  # do not nest ProcessPoolExecutor, use ThreadPoolExecutor instead in inner loop
        results = list(
            executor.map(
                _helper_impostor, [path2imp] * len(train_dataset), train_dataset["pair"]
            )
        )

    # results = []
    # for pair in train_dataset["pair"]:
    #     results.append(impostor_detector._get_score_impl(pair))

    # print(f"Computed impostor scores for {len(results)} pairs:", results)
    # results is a list of tuples: (impostor_score, impostor_dict)
    train_dataset["impostor_score"] = [score for score, _ in results]
    train_dataset["impostor_dict"] = [impostor_dict for _, impostor_dict in results]

    labels = train_dataset["same"]
    labels = labels.values
    scores = np.concatenate(
        train_dataset["impostor_score"].values
    )  # each entry in scores is a one-element list

    fpr, tpr, roc_thresholds = roc_curve(y_true=labels, y_score=scores)
    opt_thres = _get_opt_imp_threshold(fpr, tpr, roc_thresholds)

    test_dataset["thres"] = opt_thres
    print(f"Set threshold to optimal threshold for {dataset_name} dataset: {opt_thres}")
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                _helper_impostor,
                [path2imp] * len(test_dataset),
                test_dataset["pair"],
                [False] * len(test_dataset),
            )
        )
    # results is a list of tuples: (impostor_score, impostor_dict)
    test_dataset["impostor_score"] = [score for score, _ in results]
    test_dataset["impostor_dict"] = [impostor_dict for _, impostor_dict in results]
    test_dataset["impostor_prediction"] = test_dataset["impostor_score"] >= opt_thres

    # (1) Add syn_sim_ref_paraphrases column to train_dataset
    test_dataset["syn_sim_ref_paraphrases"] = test_dataset["impostor_dict"].apply(
        _avg_sim_ref_paraphrases
    )

    # (2) Add syn_sim_disputed_candidate column to test_dataset
    def sim_pair(pair):
        """Compute sim_sim for the two texts in the pair column."""
        if pair is None or len(pair) != 2:
            return None
        return _syn_sim(pair[0], pair[1])

    test_dataset["syn_sim_disputed_candidate"] = test_dataset["pair"].apply(sim_pair)
    test_dataset.to_csv(
        save_path / f"{dataset_name}_impostor_scores_syn_sim.csv", index=False
    )
    return test_dataset


def get_metric_by_bin(score_sim_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    def compute_metric(g):
        y_true = g["same"]
        y_pred = g["impostor_prediction"]

        if metric == "Accuracy":
            return (y_pred == y_true).mean()
        elif metric == "Precision":
            return precision_score(y_true, y_pred, zero_division=0)
        elif metric == "Recall":
            return recall_score(y_true, y_pred, zero_division=0)
        elif metric == "F1":
            return f1_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return (
        score_sim_df.groupby("diff_bin").apply(compute_metric).reset_index(name=metric)
    )


def vis_acc_per_syn_sim(score_sim_df: pd.DataFrame, save_path: Path, dataset_name: str):
    """
    Visualize the accuracy per syntactic similarity.
    """
    # group by difference of syntactic similarity of reference + paraphrases and disputed candidate pair
    score_sim_df["syn_sim_diff"] = (
        score_sim_df["syn_sim_ref_paraphrases"]
        - score_sim_df["syn_sim_disputed_candidate"]
    )

    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        for col in [
            "syn_sim_diff",
            "syn_sim_ref_paraphrases",
            "syn_sim_disputed_candidate",
        ]:
            # Choose number of bins (e.g., quartiles = 4 bins)
            n_bins = 4

            score_sim_df["diff_bin"] = pd.qcut(
                score_sim_df[col],
                q=n_bins,
                labels=[f"Bin {i+1}" for i in range(n_bins)],
            )
            bin_ranges = score_sim_df.groupby("diff_bin")[col].agg(["min", "max"])
            metric_by_bin = get_metric_by_bin(score_sim_df, metric=metric)
            bin_stats = metric_by_bin.merge(bin_ranges, on="diff_bin")
            fig, ax1 = plt.subplots(figsize=(8, 5))

            ax1.bar(
                bin_stats["diff_bin"],
                bin_stats[metric],
                color="skyblue",
                label=metric,
            )

            # Bin range annotations above bars
            for i, row in bin_stats.iterrows():
                ax1.text(
                    i,
                    row[metric] + 0.02,
                    f"[{row['min']:.2f}, {row['max']:.2f}]",
                    ha="center",
                    fontsize=9,
                    color="black",
                )

            ax1.set_ylabel(metric)
            ax1.set_ylim(0, 1.1)
            ax1.tick_params(axis="y")
            quantile_type = (
                "Syntactic Similarity Difference"
                if col == "syn_sim_diff"
                else (
                    "Syntactic Similarity of Reference & Paraphrases"
                    if col == "syn_sim_ref_paraphrases"
                    else "Syntactic Similarity of Disputed & Candidate"
                )
            )
            title = (
                f"{metric} Across {quantile_type} Quantiles\n({dataset_name.capitalize()} Dataset, Bin Ranges Annotated)"
                if col != "syn_sim_diff"
                else f"{metric} Across "
                + r"Syntactic Similarity Difference$^1$"
                + f"Quantiles\n({dataset_name.capitalize()} Dataset, Bin Ranges Annotated)"
            )
            plt.title(title)
            if col == "syn_sim_diff":
                plt.annotate(
                    "1: Difference between syntactic similarity of reference + paraphrases and disputed candidate pair",
                    xy=(1.0, -0.2),
                    xycoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize=10,
                )
            plt.tight_layout()
            path = save_path / dataset_name / metric
            path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                path
                / f"{dataset_name}_syn_sim_{quantile_type.replace(' ', '_')}_{metric.lower()}.svg"
            )


if __name__ == "__main__":
    # Run the experiment
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Student Essays
    print("Running experiment for Student Essays dataset.")
    path2student_df = SAVE_PATH / f"{CONFIG.STUDENT_ESSAYS}_impostor_scores_syn_sim.csv"
    if not path2student_df.exists():
        raise ValueError(
            f"Student Essays dataset not found at {CONFIG.PATH2STUDENT_ESSAYS}. Please ensure the dataset is available."
        )
        student_test_df = create_df(
            path2dataset=CONFIG.PATH2STUDENT_ESSAYS,
            dataset_name=CONFIG.STUDENT_ESSAYS,
            save_path=SAVE_PATH,
        )
    else:
        student_test_df = pd.read_csv(path2student_df)
    print("Visualizing accuracy per syntactic similarity for Student Essays dataset.")
    vis_acc_per_syn_sim(
        score_sim_df=student_test_df,
        save_path=SAVE_PATH,
        dataset_name=CONFIG.STUDENT_ESSAYS,
    )

    # Blog
    print("Running experiment for Blog dataset.")
    path2blog_df = SAVE_PATH / f"{CONFIG.BLOG}_impostor_scores_syn_sim.csv"
    if not path2blog_df.exists():
        raise ValueError(
            f"Blog dataset not found at {CONFIG.PATH2BLOG}. Please ensure the dataset is available."
        )
        blog_test_df = create_df(
            path2dataset=CONFIG.PATH2BLOG,
            dataset_name=CONFIG.BLOG,
            save_path=SAVE_PATH,
        )
    else:
        blog_test_df = pd.read_csv(path2blog_df)
    print("Visualizing accuracy per syntactic similarity for Blog dataset.")
    vis_acc_per_syn_sim(
        score_sim_df=blog_test_df,
        save_path=SAVE_PATH,
        dataset_name=CONFIG.BLOG,
    )

    print("Experiment completed. Results saved to:", SAVE_PATH)
