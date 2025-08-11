import argparse
from asyncio import sleep
import asyncio
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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

SAVE_PATH = (
    Path(__file__).resolve().parents[2]
    / CONFIG.SAVE_PATH
    / "impostor_scores"
    / "experiments"
    / "impostor_generator_comparison"
)
IMP_GEN_OPTIONS = [
    "llm",
    "text_len",
    "on-the-fly",
    "blogs",
    "fixed",
    "content",
    "naive_llm",
    "non_naive_llm",
    "mirror_minds",
]


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


def _helper_impostor(path2imp, pair, training_mode=True, imp_gen: str = "mirror_minds"):
    impostor_detector = ImpostorDetector(
        impostor_technique=imp_gen,
        n_impostors=50,
        rounds=100,  # cf. pg. 181, Koppel et al. (2014)
        top_n=100000,  # cf. pg. 179, Koppel et al. (2014)
        path2imp=path2imp,
        upsample=False,
    )
    impostor_detector.set_training_mode(training_mode)
    return impostor_detector._get_score_impl(pair)


async def create_df(path2dataset: str, dataset_name: str, save_path: Path):
    train_dataset = get_dataset(path2dataset, split="train")
    test_dataset = get_dataset(path2dataset, split="test")
    # TODO: Test on small data subsets
    train_dataset = pd.concat(
        [
            train_dataset.loc[train_dataset["same"]].head(5),
            train_dataset.loc[~train_dataset["same"]].head(5),
        ],
        ignore_index=False,
    )

    test_dataset = pd.concat(
        [
            test_dataset.loc[test_dataset["same"]].head(5),
            test_dataset.loc[~test_dataset["same"]].head(5),
        ],
        ignore_index=False,
    )

    path2imp = (
        Path(os.getcwd()).resolve() / CONFIG.PATH2BLOG
        if dataset_name == CONFIG.BLOG
        else Path(os.getcwd()).resolve() / CONFIG.PATH2STUDENT_ESSAYS
    )

    print(f"Running experiment for {dataset_name} dataset.")

    for imp_gen in IMP_GEN_OPTIONS:
        # SAIA API rate limits exceeded
        # with ThreadPoolExecutor() as executor:  # do not nest ProcessPoolExecutor, use ThreadPoolExecutor instead in inner loop
        #     results = list(
        #         executor.map(
        #             _helper_impostor,
        #             [path2imp] * len(train_dataset),
        #             train_dataset["pair"],
        #             [True] * len(train_dataset),  # training mode
        #             [imp_gen] * len(train_dataset),
        #         )
        #     )
        # TODO: maybe add sleep
        results = []
        res = None
        for pair in train_dataset["pair"]:
            i = 0
            while res is None:
                i += 10
                await sleep(i)
                res = _helper_impostor(
                    path2imp, pair, training_mode=True, imp_gen=imp_gen
                )
                print(
                    f"SAIA API Rate limit exceeded: {res is None}, sleeping for {i} seconds if True."
                )
                if i > 100:
                    break
            results.append(res)

        train_dataset[f"impostor_score_{imp_gen}"] = [score for score, _ in results]
        train_dataset[f"impostor_dict_{imp_gen}"] = [
            impostor_dict for _, impostor_dict in results
        ]

        labels = train_dataset["same"]
        labels = labels.values
        scores = np.concatenate(
            train_dataset[f"impostor_score_{imp_gen}"].values
        )  # each entry in scores is a one-element list

        fpr, tpr, roc_thresholds = roc_curve(y_true=labels, y_score=scores)
        opt_thres = _get_opt_imp_threshold(fpr, tpr, roc_thresholds)

        test_dataset["thres"] = opt_thres
        print(
            f"Set threshold to optimal threshold for {dataset_name} dataset: {opt_thres}"
        )
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
        test_dataset[f"impostor_score_{imp_gen}"] = [score for score, _ in results]
        test_dataset[f"impostor_dict_{imp_gen}"] = [
            impostor_dict for _, impostor_dict in results
        ]
        test_dataset[f"impostor_prediction_{imp_gen}"] = test_dataset[
            f"impostor_score_{imp_gen}"
        ]

        test_dataset.to_csv(
            save_path / f"{dataset_name}_impostor_scores_syn_sim.csv", index=False
        )
    return test_dataset


def df2optimal_metrics(score_dif_imp_gen_df: pd.DataFrame):
    """
    Computes the optimal metrics for each impostor generator based on the given DataFrame.
    The DataFrame is expected to have columns for impostor scores and predictions.

    :param score_dif_imp_gen_df: DataFrame containing impostor scores and predictions.
    :return: Dictionary with optimal metrics for each impostor generator:
    {
    imp_gen_name: {
        'accuracy': float,
        'f1': float,
        'precision': float,
        'recall': float
    },
    ...
    }
    """
    optimal_metrics = {}
    for imp_gen in IMP_GEN_OPTIONS:
        if f"impostor_prediction_{imp_gen}" not in score_dif_imp_gen_df.columns:
            continue
        pred = score_dif_imp_gen_df[f"impostor_prediction_{imp_gen}"]
        true_labels = score_dif_imp_gen_df["same"]

        acc = (pred == true_labels).mean()
        f1 = evaluate.load("f1").compute(predictions=pred, references=true_labels)["f1"]
        precision = evaluate.load("precision").compute(
            predictions=pred, references=true_labels
        )["precision"]
        recall = evaluate.load("recall").compute(
            predictions=pred, references=true_labels
        )["recall"]

        optimal_metrics[imp_gen] = {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    return optimal_metrics


def plot_optimal_threshold_bars(
    df: pd.DataFrame, dataset_name: str, save_path: Path = SAVE_PATH
):
    optimal_metrics = df2optimal_metrics(df)
    imp_gens = list(optimal_metrics.keys())
    accs = [optimal_metrics[ig]["accuracy"] for ig in imp_gens]
    f1s = [optimal_metrics[ig]["f1"] for ig in imp_gens]
    precs = [optimal_metrics[ig]["precision"] for ig in imp_gens]
    recs = [optimal_metrics[ig]["recall"] for ig in imp_gens]

    # Assign a unique color for each imp_gen
    cmap = plt.get_cmap("tab10")  # tab10 gives 10 distinct colors
    colors = {ig: cmap(i % 10) for i, ig in enumerate(imp_gens)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    bar_width = 0.8  # width of bars

    def plot_bar(ax, values, title):
        bars = ax.bar(
            imp_gens, values, color=[colors[ig] for ig in imp_gens], alpha=0.7
        )
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(True, axis="y")
        return bars

    # Plot each metric
    bars_acc = plot_bar(axes[0, 0], accs, "Accuracy at Optimal Threshold")
    bars_f1 = plot_bar(axes[0, 1], f1s, "F1 Score at Optimal Threshold")
    bars_prec = plot_bar(axes[1, 0], precs, "Precision at Optimal Threshold")
    bars_rec = plot_bar(axes[1, 1], recs, "Recall at Optimal Threshold")

    # Create one shared legend for all subplots
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[ig]) for ig in imp_gens]
    fig.legend(
        handles,
        imp_gens,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(imp_gens),
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for legend
    plt.savefig(save_path / f"{dataset_name}_optimal_threshold_bars.svg")
    plt.close(fig)


def compute_metrics_over_thresholds(y_true, y_scores, thresholds=None):
    """
    Compute F1, Accuracy, Precision, and Recall for a range of thresholds.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_scores : array-like
        Predicted scores/probabilities (continuous).
    thresholds : array-like, optional
        List or array of thresholds to evaluate. If None, will use np.linspace(0, 1, 101).

    Returns
    -------
    metrics_dict : dict
        {
            'thresholds': [...],
            'f1s': [...],
            'accs': [...],
            'precisions': [...],
            'recalls': [...]
        }
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)  # 0.00 to 1.00 in steps of 0.01

    f1s, accs, precisions, recalls = [], [], [], []

    for th in thresholds:
        y_pred = (y_scores >= th).astype(int)

        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        accs.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))

    return {
        "thresholds": np.array(thresholds),
        "f1s": np.array(f1s),
        "accs": np.array(accs),
        "precisions": np.array(precisions),
        "recalls": np.array(recalls),
    }


def get_scores_dict_for_diff_thres(df: pd.DataFrame):
    """{
        'llm': {
            'thresholds': [...],
            'f1s': [...],
            'accs': [...],
            'precision': [...],
            'recall': [...],
            'pr_thresholds': [...]
        },
        ...
    }"""
    scores_per_imp_gen_per_thres = {}
    for imp_gen in IMP_GEN_OPTIONS:
        if f"impostor_score_{imp_gen}" not in df.columns:
            continue
        y_true = df["same"].values
        y_scores = np.concatenate(df[f"impostor_score_{imp_gen}"].values)
        scores_per_imp_gen_per_thres[imp_gen] = compute_metrics_over_thresholds(
            y_true, y_scores
        )
    return scores_per_imp_gen_per_thres


def plot_threshold_curves_all(df: pd.DataFrame, save_path: Path, dataset_name: str):
    scores_per_imp_gen_per_thres_dict = get_scores_dict_for_diff_thres(df)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cmap = plt.get_cmap("tab10")
    colors = {
        ig: cmap(i % 10)
        for i, ig in enumerate(scores_per_imp_gen_per_thres_dict.keys())
    }

    # --- Plot 1: F1 vs Threshold ---
    for imp_gen, vals in scores_per_imp_gen_per_thres_dict.items():
        axes[0, 0].plot(
            vals["thresholds"], vals["f1s"], label=imp_gen, color=colors[imp_gen]
        )
    axes[0, 0].set_xlabel("Threshold")
    axes[0, 0].set_ylabel("F1 Score $\\frac{{2PR}}{{P+R}}$", fontsize=14)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Threshold vs F1 Score")
    axes[0, 0].grid(True)

    # --- Plot 2: Accuracy vs Threshold ---
    for imp_gen, vals in scores_per_imp_gen_per_thres_dict.items():
        axes[0, 1].plot(
            vals["thresholds"], vals["accs"], label=imp_gen, color=colors[imp_gen]
        )
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Accuracy Score $\\frac{{TP + TN}}{{N}}$", fontsize=14)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Threshold vs Accuracy Score")
    axes[0, 1].grid(True)

    # --- Plot 3: Precision vs Threshold ---
    for imp_gen, vals in scores_per_imp_gen_per_thres_dict.items():
        axes[1, 0].plot(
            vals["pr_thresholds"],
            vals["precision"][:-1],
            label=imp_gen,
            color=colors[imp_gen],
        )
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("Precision $\\frac{{TP}}{{TP + FP}}$", fontsize=14)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Threshold vs Precision Score")
    axes[1, 0].grid(True)

    # --- Plot 4: Recall vs Threshold ---
    for imp_gen, vals in scores_per_imp_gen_per_thres_dict.items():
        axes[1, 1].plot(
            vals["pr_thresholds"],
            vals["recall"][:-1],
            label=imp_gen,
            color=colors[imp_gen],
        )
    axes[1, 1].set_xlabel("Threshold")
    axes[1, 1].set_ylabel("Recall $\\frac{{TP}}{{TP + FN}}$", fontsize=14)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Threshold vs Recall Score")
    axes[1, 1].grid(True)

    # One shared legend
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color=colors[ig])
            for ig in scores_per_imp_gen_per_thres_dict.keys()
        ],
        labels=list(scores_per_imp_gen_per_thres_dict.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(scores_per_imp_gen_per_thres_dict),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"{dataset_name}_threshold_curves_all.svg")
    plt.close(fig)


if __name__ == "__main__":
    # Run the experiment
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Student Essays
    print("Running experiment for Student Essays dataset.")
    student_test_df = asyncio.run(
        create_df(
            path2dataset=CONFIG.PATH2STUDENT_ESSAYS,
            dataset_name=CONFIG.STUDENT_ESSAYS,
            save_path=SAVE_PATH,
        )
    )
    print("Visualizing accuracy per syntactic similarity for Student Essays dataset.")
    plot_optimal_threshold_bars(
        df=student_test_df,
        save_path=SAVE_PATH,
        dataset_name=CONFIG.STUDENT_ESSAYS,
    )
    plot_threshold_curves_all(
        df=student_test_df,
        save_path=SAVE_PATH,
        dataset_name=CONFIG.STUDENT_ESSAYS,
    )

    # Blog
    print("Running experiment for Blog dataset.")
    blog_test_df = asyncio.run(
        create_df(
            path2dataset=CONFIG.PATH2BLOG,
            dataset_name=CONFIG.BLOG,
            save_path=SAVE_PATH,
        )
    )
    print("Visualizing accuracy per syntactic similarity for Blog dataset.")
    plot_optimal_threshold_bars(
        df=blog_test_df,
        save_path=SAVE_PATH,
        dataset_name=CONFIG.BLOG,
    )
    plot_threshold_curves_all(
        df=blog_test_df,
        save_path=SAVE_PATH,
        dataset_name=CONFIG.BLOG,
    )

    print("Experiment completed. Results saved to:", SAVE_PATH)
