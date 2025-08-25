"""
This experiments compares the performance of different AV detectors for different input settings:
(1) Human-LLM, (2) Human-LLM (3) LLM-LLM with same LLM, (4) LLM-LLM with differnt LLMs.
df used is Student Essays.
Similar to Koppel et al. (2014), we configured each pair of input pairs to originate from different tasks.
"""

# Code from comp_imp_generators.py and adapt such that unmasking and PPMD work as well.
# plot results for optimal thresholds, or compare threshold if possible else make horizontal line for baselines

import argparse
import json
from time import sleep
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

from genai_detection.detectors.impostor_supervised_baseline import (
    SupervisedImpostorBaseline,
)
from genai_detection.detectors.impostor_unsupervised_baseline import (
    UnSupervisedImpostorBaseline,
)
from genai_detection.detectors.ppmd import PPMdDetector
from genai_detection.detectors.unmasking import UnmaskingDetector
from genai_detection.impostor_generators.MirrorMinds_generator import (
    MirrorMindsGenerator,
)

SAVE_PATH = (
    Path(__file__).resolve().parents[2]
    / CONFIG.SAVE_PATH
    / "impostor_scores"
    / "experiments"
    / "detection_scenarios"
)
IMP_GEN_OPTIONS = [
    "content",  # no device error when running sequentially and with 256GB RAM, 4 CPU cores
    "fixed",
    "text_len",
    # "non_naive_llm",
    # "llm",
    # "on-the-fly", # no more api calls
    "mirror_minds",  # no device error when running sequentially and with 256GB RAM, 4 CPU cores
    "naive_llm",
]

BASELINES = [
    "unsupervised baseline min-max",
    "unsupervised baseline cosine",
    "supervised baseline",
    "unmasking baseline",
    "ppmd baseline",
]


def get_df(split: str = "train") -> pd.DataFrame:
    """
    Load the cross-genre df from the specified path.
    The df is expected to be in a format compatible with the `load_from_disk` function.

    The construction of this cross-genre df is in file `genai_detection/df_util.py`.
    """
    assert os.path.exists(
        CONFIG.PATH2ARTIFICIAL_STUDENT_ESSAYS
    ), f"df path {CONFIG.PATH2ARTIFICIAL_STUDENT_ESSAYS} does not exist."
    df = load_from_disk(CONFIG.PATH2ARTIFICIAL_STUDENT_ESSAYS)[split].to_pandas()
    print(f"Loaded {split} df with {len(df)} examples.")
    return df


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


def _get_num_impostors(imp_gen: str) -> int:
    """
    Returns the number of impostors to generate based on the impostor generator type.
    :param imp_gen: Impostor generator type.
    :return: Number of impostors to generate.
    """
    # return 50
    # TODO: max 2 calls per second/ 14 calls per minute for SAIA based generators
    if imp_gen in ["mirror_minds", "on-the-fly", "naive_llm", "non_naive_llm", "llm"]:
        return 2  # 25
    else:
        return 50  # default for other generators


def _helper_impostor(path2imp, pair, training_mode=True, imp_gen: str = "mirror_minds"):
    print(f"Generating impostor with {imp_gen} generator")
    assert pair is not None and not (
        pd.isna(pair).any()
        if isinstance(pair, (list, tuple, np.ndarray))
        else pd.isna(pair)
    ), "Pair must not be None or NaN"

    impostor_detector = ImpostorDetector(
        impostor_technique=imp_gen,
        n_impostors=_get_num_impostors(imp_gen),
        rounds=100,  # cf. pg. 181, Koppel et al. (2014)
        top_n=100000,  # cf. pg. 179, Koppel et al. (2014)
        path2imp=path2imp,
        upsample=True,  # some impostor texts are too short otherwise
    )
    impostor_detector.set_training_mode(training_mode)
    score = impostor_detector._get_score_impl(pair)
    if imp_gen in ["mirror_minds", "on-the-fly", "naive_llm", "non_naive_llm", "llm"]:
        sleep(61)  # wait for the API to be ready
    return score


def create_df(df_name: str, save_path: Path):
    train_df = get_df(split="train")
    test_df = pd.concat([get_df(split="test"), train_df], ignore_index=True)
    # TODO: Test on small data subsets, too small does not work for optimal threshold computation
    n_test = 100  # 10
    original_test_df = pd.concat(
        [
            test_df.loc[test_df["same"]].head(n_test),
            test_df.loc[~test_df["same"]].head(n_test),
        ],
        ignore_index=False,
    )
    print(
        f"DEBUG: Obtained train ({len(train_df)} items) + test df ({len(test_df)} items)."
    )

    matching_cols = [
        "disputed_text",
        "candidate_text",
        "disputed_author",
        "candidate_author",
        "same",
        "candidate_assignment",
        "disputed_assignment",
    ]

    path2imp = (
        Path(os.getcwd()).resolve() / CONFIG.PATH2BLOG
        if df_name == CONFIG.BLOG
        else Path(os.getcwd()).resolve() / CONFIG.PATH2STUDENT_ESSAYS
    )

    for imp_gen in IMP_GEN_OPTIONS:
        if imp_gen == "on-the-fly":
            test_df = pd.concat(
                [
                    original_test_df.loc[original_test_df["same"]].head(5),
                    original_test_df.loc[~original_test_df["same"]].head(5),
                ],
                ignore_index=False,
            )
        else:
            test_df = original_test_df.copy()

        print(f"Running experiment for {df_name} df and {imp_gen} Impostor generator.")
        col_score = f"impostor_score_{imp_gen}"
        col_dict = f"impostor_dict_{imp_gen}"

        test_scores_file_name = (
            f"{df_name}_{imp_gen}_test_impostor_scores_detection_scenarios.json"
        )

        if (save_path / test_scores_file_name).exists():
            with open(save_path / test_scores_file_name, "r") as f:
                df_from_json = json.load(f)
                df_from_json = pd.DataFrame(df_from_json)
            if col_score not in test_df:
                test_df[col_score] = np.nan

            df_from_json = df_from_json.set_index(matching_cols)
            df_from_json = df_from_json[~df_from_json.index.duplicated(keep="last")]
            test_df = test_df.set_index(matching_cols)

            test_df[col_score] = test_df[col_score].fillna(df_from_json[col_score])
            if "same" not in test_df.columns:
                test_df.index.names = matching_cols
                test_df.reset_index(inplace=True)

        missing_mask = (
            test_df[col_score].isna()
            if col_score in test_df.columns
            else pd.Series(True, index=test_df.index)
        )
        if missing_mask.sum() > 0:
            # TODO: uncomment this for generation
            # print(
            #     f"Only eval, skipping generation for {missing_mask.sum()} pairs und keep only existent scores."
            # )
            # test_df = test_df[test_df[col_score].notna()]

            print(
                f"DEBUG Missing {col_score} in {missing_mask.sum()} rows of test df for {imp_gen} generator."
            )
            new_rows = test_df.loc[missing_mask].copy()
            max_workers = (
                max(1, min(16, len(new_rows)))  # number of cpus cores = 16
                if imp_gen
                not in [
                    "mirror_minds",
                    "on-the-fly",
                    "naive_llm",
                    "non_naive_llm",
                    "llm",
                ]
                else max(1, 4 // _get_num_impostors(imp_gen))
            )  # 14 per minute, use fewer to be save
            if imp_gen in ["content", "mirror_minds"]:
                results = []
                for pair in new_rows["pair"]:
                    res = _helper_impostor(
                        path2imp, pair, training_mode=False, imp_gen=imp_gen
                    )
                    results.append(res)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
            test_df.loc[missing_mask, col_score] = [score for score, _ in results]
            test_df.loc[missing_mask, col_dict] = [imp_dict for _, imp_dict in results]

        if missing_mask.sum() > 0:
            test_df.to_json(
                save_path / test_scores_file_name, orient="records", indent=4
            )
            print(
                "DEBUG: Saved test df with predictions to JSON, path:",
                save_path / test_scores_file_name,
            )

        # NEVER COMMENT THIS: Add missing columns from test_df, initialized with NaN
        original_test_df = original_test_df.reindex(
            columns=original_test_df.columns.union(test_df.columns)
        )
        original_test_df = original_test_df.fillna(test_df)

    # FIXME
    print("Columns in original test df:", original_test_df.columns)

    for baseline_name, baseline in zip(
        BASELINES,
        [
            UnSupervisedImpostorBaseline(
                use_cosine_simiarity=False, dataset_name=df_name
            ),
            UnSupervisedImpostorBaseline(
                use_cosine_simiarity=True, dataset_name=df_name
            ),
            SupervisedImpostorBaseline(dataset_name=df_name),
            UnmaskingDetector(),
            PPMdDetector(),
        ],
    ):

        test_df = original_test_df.copy()
        if baseline_name in ["unmasking baseline", "ppmd baseline"]:
            preds = np.concatenate([baseline.get_score(p) for p in test_df["pair"]])
        else:
            preds = baseline.get_score(test_df["pair"])
        print(f"DEBUG: {baseline_name} predictions:", preds)
        test_df[f"{baseline_name.replace(' ','_')}_score"] = np.array(
            preds.tolist()
        ).ravel()

        original_test_df = original_test_df.reindex(
            columns=original_test_df.columns.union(test_df.columns)
        )

        # Fill NaNs in those columns with values from test_df
        original_test_df = original_test_df.fillna(test_df)

    original_test_df = original_test_df.dropna(axis=1, how="all")
    original_test_df.to_json(
        save_path
        / f"complete_{df_name}_detection_scenarios_incl_baselines_test_df.json",
        orient="records",
        indent=4,
    )
    return original_test_df


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
        print(f"Computing optimal metrics for impostor generator: {imp_gen}")
        if f"impostor_prediction_{imp_gen}" not in score_dif_imp_gen_df.columns:
            if f"impostor_score_{imp_gen}" not in score_dif_imp_gen_df.columns:
                continue
            else:
                score_dif_imp_gen_df[f"impostor_prediction_{imp_gen}"] = (
                    score_dif_imp_gen_df[f"impostor_score_{imp_gen}"]
                    > score_dif_imp_gen_df["thres"]
                )
        pred = score_dif_imp_gen_df[f"impostor_prediction_{imp_gen}"]
        non_nan_indices = pred.notna()
        pred = pred[non_nan_indices].to_list()  # filter out NaN values
        if all(p == 0 for p in pred):
            print(
                f"Warning: All predictions for {imp_gen} are 0. Prec, Recall, F1 will be 0."
            )
        true_labels = score_dif_imp_gen_df["same"]
        true_labels = true_labels[non_nan_indices].to_list()  # filter out NaN values

        acc = np.mean([p == t for p, t in zip(pred, true_labels)])
        assert isinstance(
            pred, list
        ), "pred should be a list of predictions but got: {}".format(type(pred))
        assert isinstance(
            true_labels, list
        ), "true_labels should be a list of true labels but got: {}".format(
            type(true_labels)
        )
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


def _f05(y_trues: list, y_preds: list):
    """
    Compute F0.5 score, which gives more weight to precision than recall.
    F0.5 = (1 + 0.5^2) * # TP / ((1 + 0.5^2) * # TP + 0.5^2 (# FN + # unanswered) + # FP)
    """
    n_unanswered = [y_pred == 0.5 for y_pred in y_preds].count(True)
    n_tp = sum(y_true == 1 and y_pred == 1 for y_true, y_pred in zip(y_trues, y_preds))
    n_fp = sum(y_true == 0 and y_pred == 1 for y_true, y_pred in zip(y_trues, y_preds))
    n_fn = sum(y_true == 1 and y_pred == 0 for y_true, y_pred in zip(y_trues, y_preds))
    nominator = (1 + 0.5**2) * n_tp
    denominator = (1 + 0.5**2) * n_tp + 0.5**2 * (n_fn + n_unanswered) + n_fp

    if denominator == 0:
        return 0.0
    return nominator / denominator


def _c_at_1(y_trues: list, y_preds: list):
    """
    Compute C@1 score, which is the proportion of instances where the top prediction is correct.
    C@1 = (# correct answer / # problems) * ( 1 + # unanswered / # problems )
    """
    n_correct = sum(y_true == y_pred for y_true, y_pred in zip(y_trues, y_preds))
    n_unanswered = [y_pred == 0.5 for y_pred in y_preds].count(True)
    n_total = len(y_trues)

    return (n_correct / n_total) * (1 + n_unanswered / n_total) if n_total > 0 else 0.0


def compute_metrics_over_thresholds(y_trues, y_scores, thresholds=None):
    """
    Compute F1, Accuracy, Precision, and Recall for a range of thresholds.

    Parameters
    ----------
    y_trues : array-like
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
            'recalls': [...],
            'f05s': [...],
            'c@1s': [...],
        }
    """
    print("DEBUG: y_trues:", y_trues)
    print("DEBUG: y_scores:", y_scores)
    y_scores = np.array(y_scores, dtype=np.float64)
    nan_indices = np.isnan(y_scores)
    print("DEBUG: Number of NaN in y_scores:", np.sum(nan_indices))
    y_scores = y_scores[~nan_indices]
    y_trues = y_trues[~nan_indices]

    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)  # 0.00 to 1.00 in steps of 0.01

    f1s, accs, precisions, recalls, f05s, c1s = [], [], [], [], [], []

    for th in thresholds:
        y_preds = (y_scores >= th).astype(int)

        f1s.append(f1_score(y_trues, y_preds, zero_division=0))
        accs.append(accuracy_score(y_trues, y_preds))
        precisions.append(precision_score(y_trues, y_preds, zero_division=0))
        recalls.append(recall_score(y_trues, y_preds, zero_division=0))
        f05s.append(_f05(y_trues, y_preds))
        c1s.append(_c_at_1(y_trues, y_preds))

    return {
        "thresholds": np.array(thresholds),
        "f1s": np.array(f1s),
        "accs": np.array(accs),
        "precisions": np.array(precisions),
        "recalls": np.array(recalls),
        "f05s": np.array(f05s),
        "c@1s": np.array(c1s),
    }


def get_scores_dict_for_diff_thres(df: pd.DataFrame):
    """{group_name: {
            'llm': {
                'thresholds': [...],
                'f1s': [...],
                'accs': [...],
                'precisions': [...],
                'recalls': [...],
                'f05s': [...],
                'c@1s': [...],
                'pr_thresholds': [...]
            },
            ...
            }
            ...
    }"""
    scores_per_imp_gen_per_thres = {}
    llm_names = [
        "qwen3-32b",
        "mistral-large-instruct",
        "openai-gpt-oss-120b",
        "meta-llama-3.1-8b-instruct",
    ]

    # group 3 scenarios together, i.e. same LLM, different LLMs, human-LLM
    group1 = df[df["artificial_generation"] == False]  # human-human (AV)
    group2a = df[
        df["candidate_author"].isin(llm_names)
    ]  # human-LLM: All text pairs where candidate is LLM (LLM detection)
    group3a = group2a  # candidate is LLM, LLM AV with human and llm disputed texts (other way arounf does not work, bc there are no human candidates for llm disputed texts in this dataset)
    group3b = df[
        df["candidate_author"].isin(llm_names) & df["disputed_author"].isin(llm_names)
    ]  # LLM Author Verification
    groups = {
        "Human-Human-(AV)": group1,  # disputed is always human
        "LLM-Detection": group2a,  # detect if LLM involved; candidate is always LLM
        "LLM-AV": group3a,  # candidate is LLM, LLM AV with human and llm disputed texts (other way arounf does not work, bc there are no human candidates for llm disputed texts in this dataset)
        "LLM-AV-(only-LLMs)": group3b,  # disputed + candidate is always LLM
    }

    for group_name, group_df in groups.items():
        print(f"DEBUG: Processing group {group_name} with {len(group_df)} examples.")
        if len(group_df) == 0:
            print(f"WARNING: Group {group_name} is empty, skipping.")
            continue

        # Compute metrics for each impostor generator in the group
        scores_per_imp_gen_per_thres[group_name] = {}
        for imp_gen in IMP_GEN_OPTIONS + BASELINES:
            if (
                f"impostor_score_{imp_gen}" not in group_df.columns
                and f"{imp_gen.replace(' ','_')}_score" not in group_df.columns
            ):
                print(
                    f"ERROR: Skipping {imp_gen} as it is not in the DataFrame columns."
                )
                continue
            if group_name == "LLM-Detection":
                # we only want to detect if artificial generation is involved in disputed text
                y_true = np.array(
                    [
                        author in llm_names
                        for author in group_df["disputed_author"].values
                    ]
                )
            else:
                y_true = group_df["same"].values
            y_scores = (
                group_df[f"impostor_score_{imp_gen}"].values
                if imp_gen in IMP_GEN_OPTIONS
                else group_df[f"{imp_gen.replace(' ','_')}_score"].values
            )
            print("---------------\nIMP GEN:", imp_gen)
            scores_per_imp_gen_per_thres[group_name][imp_gen] = (
                compute_metrics_over_thresholds(y_true, y_scores)
            )
    return scores_per_imp_gen_per_thres


def plot_rocs_threshold_imposter(
    df, savefig_basepath: str = None, pos_label: int = 1, title_kwargs: dict = None
):
    for imp_gen in IMP_GEN_OPTIONS + BASELINES:
        y_true = df["same"].values
        y_scores = (
            df[f"impostor_score_{imp_gen}"].values
            if imp_gen in IMP_GEN_OPTIONS
            else df[f"{imp_gen.replace(' ','_')}_score"].values
        )

        # ROC Curve: Balanced classes or when you care about TPR vs. FPR
        # displayed for different thresholds
        # scores: probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers)
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html (05.06.2025)
        fpr, tpr, roc_thresholds = roc_curve(
            y_true=y_true, y_score=y_scores, pos_label=pos_label
        )
        plt.plot(fpr, tpr, label=imp_gen)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate $FPR = 1 - \\frac{{TN}}{{TN + FP}}$", fontsize=14)
    plt.ylabel("True Positive Rate $TPR = R = \\frac{{TP}}{{TP + FN}}$", fontsize=14)
    title = "ROC Curve" if not title_kwargs else f"ROC Curve\n{title_kwargs}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savefig_basepath:
        figure_name = (
            "roc_curve.svg"
            if not title_kwargs
            else f"roc_curve_dif_impg_gen_{title_kwargs['dataset_name']}.svg"
        )
        savefig = os.path.join(savefig_basepath, figure_name)
        plt.savefig(savefig)
    plt.show()
    plt.close()


def plot_threshold_curves_all_single(df: pd.DataFrame, save_path: Path, df_name: str):
    # ROC curve for each impostor generator
    plot_rocs_threshold_imposter(
        df,
        pos_label=1,
        title_kwargs={"dataset_name": df_name},
        savefig_basepath=save_path,
    )
    # TODO: for testing
    return None

    scores_diff_scenarios = get_scores_dict_for_diff_thres(df)
    group_descriptions = {
        "Human-Human-(AV)": "no artificial generation",
        "LLM-Detection": "LLM candidate",
        "LLM-AV": r"$\geq 1 \text{ LLM text}$",
        "LLM-AV-(only-LLMs)": "disputed & candidate LLM",
    }
    for (
        scenario_name,
        scores_per_imp_gen_per_thres_dict,
    ) in scores_diff_scenarios.items():
        print(
            f"DEBUG: Processing group {scenario_name} with {len(scores_per_imp_gen_per_thres_dict)} examples."
        )
        if len(scores_per_imp_gen_per_thres_dict) == 0:
            print(f"WARNING: Group {scenario_name} is empty, skipping.")
            continue
        print(
            f"DEBUG: [{scenario_name} scenario] Scores per impostor generator per threshold for keys:",
            scores_per_imp_gen_per_thres_dict.keys(),
        )
        cmap = plt.get_cmap("tab10")
        colors = {
            ig: cmap(i % 10)
            for i, ig in enumerate(scores_per_imp_gen_per_thres_dict.keys())
        }
        line_styles = [
            "--",
            ":",
            "-.",
            (5, (10, 3)),
            (0, (3, 1, 1, 1, 1, 1)),
            # (0, (3, 5, 1, 5, 1, 5)),
            (0, (3, 1, 1, 1)),
        ]

        metrics = ["f1s", "accs", "precisions", "recalls", "f05s", "c@1s"]
        titles = [
            "Threshold vs F1 Score",
            "Threshold vs Accuracy Score",
            "Threshold vs Precision Score",
            "Threshold vs Recall Score",
            "Threshold vs $F_{0.5}$ Score",
            "Threshold vs C@1 Score",
        ]
        ylabels = [
            "F1 Score $\\frac{{2PR}}{{P+R}}$",
            "Accuracy Score $\\frac{{TP + TN}}{{N}}$",
            "Precision $\\frac{{TP}}{{TP + FP}}$",
            "Recall $\\frac{{TP}}{{TP + FN}}$",
            "$F_{0.5}$ Score $\\frac{{(1 + 0.5^2)TP}}{{(1 + 0.5^2)TP + 0.5^2(FN + unanswered) + FP}}$",
            "C@1 Score $\\frac{{TP + TN}}{{TP + TN + FP + FN}} \\cdot (1 + \\frac{{unanswered}}{{TP + TN + FP + FN}})$",
        ]
        for metric, title, ylabel in zip(metrics, titles, ylabels):
            plt.figure(figsize=(12, 6))
            for i, (imp_gen, vals) in enumerate(
                scores_per_imp_gen_per_thres_dict.items()
            ):
                print("DEBUG: Imp gen:", imp_gen)
                assert (
                    metric in vals
                ), f"Metric {metric} not found in values for {imp_gen}. Has only {vals.keys()} keys."
                plt.plot(
                    vals["thresholds"],
                    vals[metric],
                    label=imp_gen,
                    color=colors[imp_gen],
                    linewidth=1.5,
                    linestyle=line_styles[i % len(line_styles)],
                )
            plt.xlabel("Threshold", fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.ylim(0, 1)
            plt.title(
                f"{title}\nin {' '.join(scenario_name.split('-')[:2])} scenario\n({group_descriptions[scenario_name]})",
                fontsize=14,
            )
            plt.grid(False)
            plt.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                ncol=1,
                frameon=False,
            )

            plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space for legend
            plt.subplots_adjust(hspace=0.2)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                save_path
                / f"{df_name}_{scenario_name}_threshold_{metric}_curves_all_incl_baselines.svg",
                bbox_inches="tight",
            )
            plt.close()


if __name__ == "__main__":
    # works local (20.08.2025)
    # Run the experiment
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Student Essays
    print("Running experiment for Artificial Student Essays df.")
    student_test_df = create_df(
        df_name=CONFIG.STUDENT_ESSAYS,
        save_path=SAVE_PATH,
    )
    print(
        "Creating DataFrame for Artificial Student Essays df completed. Saved to:",
        SAVE_PATH,
    )
    print("DEBUG: Artificial Student Essays test DataFrame:", student_test_df.head())
    print(
        "DEBUG: Artificial Student Essays test DataFrame columns:",
        student_test_df.columns,
    )
    print(
        "Visualizing accuracy, prec, recall, f1 per syntactic similarity for Artificial Student Essays df."
    )
    plot_threshold_curves_all_single(
        df=student_test_df,
        save_path=SAVE_PATH / "single_metrics",
        df_name=CONFIG.STUDENT_ESSAYS,
    )
    print(
        "Experiment for Artificial Student Essays df completed. Plots saved to:",
        SAVE_PATH,
    )
