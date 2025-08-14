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
import os

# https://stackoverflow.com/questions/65744442/how-to-use-threads-for-huggingface-transformers

from genai_detection.impostor_generators.MirrorMinds_generator import (
    MirrorMindsGenerator,
)

SAVE_PATH = (
    Path(__file__).resolve().parents[2]
    / CONFIG.SAVE_PATH
    / "impostor_scores"
    / "experiments"
    / "impostor_generator_comparison"
)
IMP_GEN_OPTIONS = [
    "fixed",
    # "naive_llm",
    # "non_naive_llm",
    # "llm",
    "text_len",
    "content",
    # "on-the-fly",
    # "mirror_minds",
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
    print(f"Generating impostor with {imp_gen} generator")
    assert pair is not None and not (
        pd.isna(pair).any()
        if isinstance(pair, (list, tuple, np.ndarray))
        else pd.isna(pair)
    ), "Pair must not be None or NaN"

    impostor_detector = ImpostorDetector(
        impostor_technique=imp_gen,
        n_impostors=50,
        rounds=100,  # cf. pg. 181, Koppel et al. (2014)
        top_n=100000,  # cf. pg. 179, Koppel et al. (2014)
        path2imp=path2imp,
        upsample=False,
    )
    impostor_detector.set_training_mode(training_mode)
    # FIXME: debugging
    if pair is not None and not (
        pd.isna(pair).any()
        if isinstance(pair, (list, tuple, np.ndarray))
        else pd.isna(pair)
    ):
        if isinstance(pair, str):
            print(f"Pair is a string: {pair[:29]}...")  # Print first 30 characters
        elif isinstance(pair, (list, tuple, np.ndarray)):
            print(
                f"Pair is a list/tuple/ndarray: {[p[:20] for p in pair]}..."
            )  # Print first 3 elements
        else:
            raise ValueError(f"Unsupported type for pair: {type(pair)}/ {pair}...")
    return impostor_detector._get_score_impl(pair)


def _split_unhashable(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["pair", "authors"]:
        if col in df.columns:
            df[f"{col}_left"] = df[col].apply(
                lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray)) else x
            )
            df[f"{col}_right"] = df[col].apply(
                lambda x: x[1] if isinstance(x, (list, tuple, np.ndarray)) else x
            )
    return df


def create_df(path2dataset: str, dataset_name: str, save_path: Path):
    train_dataset = get_dataset(path2dataset, split="train")
    test_dataset = get_dataset(path2dataset, split="test")
    # TODO: Test on small data subsets
    train_dataset = pd.concat(
        [
            train_dataset.loc[train_dataset["same"]].head(1),
            train_dataset.loc[~train_dataset["same"]].head(1),
        ],
        ignore_index=False,
    )
    print("1 Number of elements in train dataset:", len(train_dataset))

    test_dataset = pd.concat(
        [
            test_dataset.loc[test_dataset["same"]].head(1),
            test_dataset.loc[~test_dataset["same"]].head(1),
        ],
        ignore_index=False,
    )
    matching_cols = [
        "pair_left",
        "pair_right",
        "authors_left",
        "authors_right",
        "same",
    ]
    assert not any(
        [
            pair is None
            or (
                pd.isna(pair).any()
                if hasattr(pair, "__iter__") and not isinstance(pair, str)
                else pd.isna(pair)
            )
            for pair in test_dataset["pair"]
        ]
    ), "Test dataset pairs must not be None or NaN"
    print(
        "Test dataset cols at beginning pair col",
        [type(p) for p in test_dataset["pair"]],
        type(test_dataset["pair"]),
    )

    path2imp = (
        Path(os.getcwd()).resolve() / CONFIG.PATH2BLOG
        if dataset_name == CONFIG.BLOG
        else Path(os.getcwd()).resolve() / CONFIG.PATH2STUDENT_ESSAYS
    )

    print(f"Running experiment for {dataset_name} dataset.")

    # FIXME: rate limit + mirror minds download problem
    for imp_gen in IMP_GEN_OPTIONS:
        col_score = f"impostor_score_{imp_gen}"
        col_dict = f"impostor_dict_{imp_gen}"

        test_scores_file_name = (
            f"{dataset_name}_{imp_gen}_test_impostor_scores_syn_sim.csv"
        )
        train_scores_file_name = (
            f"{dataset_name}_{imp_gen}_train_impostor_scores_syn_sim.csv"
        )

        if (save_path / train_scores_file_name).exists():
            print(
                f"Reading {imp_gen} for {dataset_name} dataset from existing file {save_path / train_scores_file_name}"
            )
            with open(save_path / train_scores_file_name, "r") as f:
                df_from_csv = pd.read_csv(f)
                # df_from_csv.reset_index(inplace=True)
                assert (
                    "pair" in df_from_csv.columns
                ), f"Expected 'pair' column in {train_scores_file_name}, but not found: {df_from_csv.columns}"

            train_dataset = _split_unhashable(train_dataset)
            assert (
                "pair" in train_dataset.columns
            ), f"Expected 'pair' column in training dataset, but not found: {train_dataset.columns}"
            if col_score not in train_dataset:
                train_dataset[col_score] = np.nan
            # print(f"Train dataset columns before merging: {train_dataset.columns}")
            # train_dataset.set_index(
            #     ["pair_left", "pair_right", "authors_left", "authors_right", "same"],
            #     inplace=True,
            #     # drop=False,
            # )
            # print(f"Train dataset columns before merging: {train_dataset.columns}")

            # df_from_csv = _split_unhashable(df_from_csv)
            # df_from_csv.drop(["pair", "authors"], axis=1, inplace=True)
            # df_from_csv.set_index(
            #     ["pair_left", "pair_right", "authors_left", "authors_right", "same"],
            #     inplace=True,
            #     # drop=False,
            # )
            df_from_csv = df_from_csv.set_index(matching_cols)
            df_from_csv = df_from_csv[~df_from_csv.index.duplicated(keep="last")]
            train_dataset = train_dataset.set_index(matching_cols)
            train_dataset = train_dataset[~train_dataset.index.duplicated(keep="last")]

            # Fill only missing col_score values
            train_dataset[col_score] = train_dataset[col_score].fillna(
                df_from_csv[col_score]
            )

            # train_dataset = train_dataset.combine_first(df_from_csv)
            print(f"Train dataset columns after merging: {train_dataset.columns}")
            assert (
                not train_dataset.empty
            ), f"Train dataset is empty after merging with {train_scores_file_name}"

        if (save_path / test_scores_file_name).exists():
            with open(save_path / test_scores_file_name, "r") as f:
                df_from_csv = pd.read_csv(f)
                # df_from_csv.reset_index(inplace=True)
            test_dataset = _split_unhashable(test_dataset)
            print(
                "Test dataset cols after split unhashable pair col",
                [type(p) for p in test_dataset["pair"]],
                type(test_dataset["pair"]),
            )
            if col_score not in test_dataset:
                test_dataset[col_score] = np.nan
            # test_dataset.set_index(
            #     ["pair_left", "pair_right", "authors_left", "authors_right", "same"],
            #     inplace=True,
            #     # drop=False,
            # )

            df_from_csv = _split_unhashable(df_from_csv)
            # df_from_csv.drop(["pair", "authors"], axis=1, inplace=True)
            # df_from_csv.set_index(
            #     ["pair_left", "pair_right", "authors_left", "authors_right", "same"],
            #     inplace=True,
            #     # drop=False,
            # )

            # FIXME: destroys integrity of the dataset

            # for df in (test_dataset, df_from_csv):
            #     df.index = pd.MultiIndex.from_frame(
            #         df.index.to_frame().astype(str).apply(lambda s: s.str.strip())
            #     )

            df_from_csv = df_from_csv.set_index(matching_cols)
            test_dataset = test_dataset.set_index(matching_cols)
            print("important")
            print(test_dataset.index.equals(df_from_csv.index))
            print(test_dataset.index.dtypes)
            print(df_from_csv.index.dtypes)
            # print(
            #     "Only in test_dataset:",
            #     len(test_dataset.index.difference(df_from_csv.index)),
            # )
            # print(
            #     "Only in df_from_csv:",
            #     len(df_from_csv.index.difference(test_dataset.index)),
            # )

            # Fill only missing col_score values
            test_dataset[col_score] = test_dataset[col_score].fillna(
                df_from_csv[col_score]
            )

            # FIXME: try this
            # for row in df_from_csv.itertuples():
            #     for row_test in test_dataset.itertuples():
            #         if all(
            #             [
            #                 getattr(row, col) == getattr(row_test, col)
            #                 for col in matching_cols
            #             ]
            #         ):
            #             test_dataset.loc[row_test.Index, col_score] = getattr(
            #                 row, col_score
            #             )
            #             continue
            # test_dataset = test_dataset.combine_first(df_from_csv)
            print(f"Test dataset cols after merging: {test_dataset.columns}")
            print(
                "Test dataset cols after merging pair col",
                [type(p) for p in test_dataset["pair"]],
                type(test_dataset["pair"]),
            )

            assert (
                not test_dataset.empty
            ), f"Test dataset is empty after merging with {test_scores_file_name}"
        # fill nan values of new rows with predicted values
        print(f"Running impostor detector for impostor generation approach: {imp_gen}")
        print("2 Number of elements in train dataset:", len(train_dataset))
        missing_mask = (
            train_dataset[col_score].isna()
            if col_score in train_dataset.columns
            else pd.Series(True, index=train_dataset.index)
        )
        print(
            f"Number of missing impostor scores in train dataset: {missing_mask.sum()}"
        )
        if missing_mask.sum() > 0:
            new_rows = train_dataset[missing_mask].copy()
            assert not new_rows[
                "pair"
            ].empty, "No new row pairs to process for training dataset"
            with ThreadPoolExecutor() as executor:  # do not nest ProcessPoolExecutor, use ThreadPoolExecutor instead in inner loop
                results = list(
                    executor.map(
                        _helper_impostor,
                        [path2imp] * len(new_rows),
                        new_rows["pair"],
                        [True] * len(new_rows),  # training mode
                        [imp_gen] * len(new_rows),
                    )
                )

            train_dataset.loc[missing_mask, col_score] = [score for score, _ in results]
            train_dataset.loc[missing_mask, col_dict] = [
                imp_dict for _, imp_dict in results
            ]
            assert results is not None, "Results should not be None"
            print(
                "DEBUG: Predictions in train dataset before:",
                train_dataset[col_score].values,
            )
            # FIXME: strange output in debugging
            print(
                "New scores (training dataset) 2:",
                train_dataset.loc[missing_mask, col_score],
            )
            print(
                "DEBUG: Predictions in train dataset after:",
                train_dataset[col_score].values,
            )
            train_dataset.reset_index(inplace=True)
            train_dataset.to_csv(
                save_path / train_scores_file_name,
                index=False,
            )

        labels = train_dataset["same"].values
        print("DEBUG: Predictions in train dataset:", train_dataset[col_score].values)
        scores = train_dataset[col_score].values
        fpr, tpr, roc_thresholds = roc_curve(y_true=labels, y_score=scores)
        opt_thres = _get_opt_imp_threshold(fpr, tpr, roc_thresholds)
        # do not use multi index
        missing_mask = (
            # pd.Series(
            [t != opt_thres for t in test_dataset["thres"]]
            # , index=test_dataset.index,
            # )
            if "thres" in test_dataset.columns
            else [True]
            * len(test_dataset)  # pd.Series(True)#, index=test_dataset.index)
        )
        if any(missing_mask):

            print("Missing mask for test dataset:", missing_mask.index, missing_mask)
            # print("test data before:", test_dataset.index, type(test_dataset["pair"]))
            print(
                "test data before:",
                [type(p) for p in test_dataset["pair"]],
                type(test_dataset["pair"]),
            )
            none_indices = test_dataset.index[
                test_dataset["pair"].isna()
                | (test_dataset["pair"].apply(lambda x: x is None))
            ].tolist()
            print("Indices with pair=None:", none_indices)
            new_rows = test_dataset.loc[missing_mask].copy()
            print("cols", new_rows.columns)
            for row in new_rows.itertuples():
                print("row", row.index, type(row.pair))
                # print(
                #     f"Row {row.Index} with pair {len(row.pair)} has threshold {row.thres}, setting to optimal threshold {opt_thres}"
                # )
            test_dataset["thres"] = opt_thres
            print(
                f"Set threshold to optimal threshold for {dataset_name} dataset: {opt_thres}"
            )
            with ThreadPoolExecutor() as executor:
                results = list(
                    executor.map(
                        _helper_impostor,
                        [path2imp] * len(new_rows),
                        new_rows["pair"],
                        [False] * len(new_rows),
                        [imp_gen] * len(new_rows),
                    )
                )
            # results is a list of tuples: (impostor_score, impostor_dict)
            test_dataset.loc[missing_mask, col_score] = [score for score, _ in results]
            test_dataset.loc[missing_mask, col_dict] = [
                imp_dict for _, imp_dict in results
            ]
            test_dataset[f"impostor_prediction_{imp_gen}"] = (
                test_dataset[col_score] > opt_thres
            )

            test_dataset.reset_index(inplace=True)
            test_dataset.to_csv(
                save_path / test_scores_file_name,
                index=False,
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
    # student_test_df = asyncio.run(
    student_test_df = create_df(
        path2dataset=CONFIG.PATH2STUDENT_ESSAYS,
        dataset_name=CONFIG.STUDENT_ESSAYS,
        save_path=SAVE_PATH,
    )
    # )
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
    # print("Running experiment for Blog dataset.")
    # # blog_test_df = asyncio.run(
    # blog_test_df = create_df(
    #     path2dataset=CONFIG.PATH2BLOG,
    #     dataset_name=CONFIG.BLOG,
    #     save_path=SAVE_PATH,
    # )
    # # )
    # print("Visualizing accuracy per syntactic similarity for Blog dataset.")
    # plot_optimal_threshold_bars(
    #     df=blog_test_df,
    #     save_path=SAVE_PATH,
    #     dataset_name=CONFIG.BLOG,
    # )
    # plot_threshold_curves_all(
    #     df=blog_test_df,
    #     save_path=SAVE_PATH,
    #     dataset_name=CONFIG.BLOG,
    # )

    # print("Experiment completed. Results saved to:", SAVE_PATH)
