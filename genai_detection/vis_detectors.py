import argparse
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Literal, Optional
from datasets import load_from_disk
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import seaborn as sns
from genai_detection.detectors.detector_base import DetectorBase
from genai_detection.detectors.impostor import ImpostorDetector
from genai_detection.detectors.impostor_supervised_baseline import (
    SupervisedImpostorBaseline,
)
from genai_detection.detectors.impostor_unsupervised_baseline import (
    UnSupervisedImpostorBaseline,
)
from genai_detection.detectors.unmasking import UnmaskingDetector

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from genai_detection.config import CONFIG


class VisDetectors:
    """
    A class to compare detectors on the same dataset.
    It visualizes the detection results using ROC and Precision-Recall curves, confusion matrices, and decision thresholds.
    Some visualization approaches are inspired by Koppel et al. (2014) paper.

    References:
    ===========
    Koppel, Moshe, and Yaron Winter. ‘Determining If Two Documents Are Written by the Same Author’.
    Journal of the Association for Information Science and Technology 65, no. 1 (January 2014): 178–87. https://doi.org/10.1002/asi.22954.
    """

    def __init__(self, dataset_name: str, detectors: list = None) -> None:
        """
        Initializes the VisDetectors class.
        :param dataset_name: The name of the dataset to use for visualization.
        :param detectors: List of detectors objects to visualize. Currently supported: Unmasking and impostor methods.
        """
        self.dataset_name = dataset_name
        assert all(
            isinstance(detector, DetectorBase) for detector in detectors
        ), "All detectors must inheret from DetectorBase."
        self.detectors = detectors if detectors is not None else []
        self.savefig_base = Path(__file__).resolve().parent.parent / CONFIG.SAVE_PATH

    def _title2filename(self, title: str) -> str:
        """
        Converts a title string to a filename-friendly format.
        :param title: The title to convert.
        :return: A filename-friendly string.
        """
        return (
            title.replace("\n", "_")
            .replace(" ", "_")
            .replace("'", "")
            .replace(".", "_")
            .replace(",", "_")
        )

    def _format_title(self, base: str, kwargs: dict) -> str:
        """
        Formats the title for the plots.
        :param base: The base title.
        :param kwargs: Additional keyword arguments to include in the title. Structured as a dictionary.
        :return: A formatted title string excluding path2imp, because paths are too long.
        """
        assert isinstance(kwargs, dict), "kwargs must be a dictionary"
        assert (
            kwargs.get("dataset_name", self.dataset_name) == self.dataset_name
        ), f"dataset parameter from kwargs ({kwargs.get('dataset')}) must match self.dataset_name ({self.dataset_name})"
        assert len(base) > 0, "Base title must not be empty"
        items = [
            f"{k}={v:.2f}" if (isinstance(v, float) or k == "threshold") else f"{k}={v}"
            for k, v in kwargs.items()
            if k != "path2imp" and k != "fig2reproduce"
        ]
        # newline after every 2 items, for better readability
        lines = []
        for i in range(0, len(items), 2):
            lines.append(", ".join(items[i : i + 2]))
        return base + "\n" + "\n".join(lines)

    def _load_datasets(self, balanced: bool = False):
        train_dataset = self.load_data(split="train")
        test_dataset = self.load_data(split="test")  # scores obtained on test data
        if balanced:
            for df in [train_dataset, test_dataset]:
                size_smaller_class = df["same"].value_counts().min()
                # ensure target class 'same' is present after being used for grouping
                df = (
                    df.groupby("same", group_keys=False)
                    .apply(
                        lambda x: x.sample(size_smaller_class, random_state=42).assign(
                            same=x["same"].iloc[0]
                        )
                    )
                    .reset_index(drop=True)
                )
        if train_dataset.empty or test_dataset.empty:
            raise ValueError(
                "Train or test dataset is empty. Cannot visualize impostors."
            )
        return train_dataset, test_dataset

    def visualize(self, balanced: bool = False) -> None:
        """
        Visualizes the detection results.

        :param balanced: Whether the number of same and different author pairs from dataset should be balanced (i.e. sampling strategy).
        """
        print(f"Visualizing detectors on dataset: {self.dataset_name}")
        train_dataset, test_dataset = self._load_datasets(balanced=balanced)
        for detector in self.detectors:
            if isinstance(detector, ImpostorDetector):
                self.visualize_impostors(
                    impostor_detector=detector,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                )
            elif isinstance(detector, UnmaskingDetector):
                self.plot_unmasking_curves(
                    unmasking_detector=detector,
                    dataset=test_dataset,
                    dataset_name=self.dataset_name,
                )
            else:
                raise ValueError(
                    f"Detector {detector} is not supported for visualization."
                )

    def _get_opt_imp_threshold(self, fpr, tpr, thresholds):
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
            print(
                "Warning: No valid TPR/FPR data available. Defaulting to threshold = 0.5"
            )
            return 0.5  # or np.nan, depending on your use case

    def visualize_impostors(
        self,
        impostor_detector,
        train_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
    ) -> None:
        """
        Visualizes the impostor detection results.
        Uses training data to find a threshold for impostor detection and then visualizes the impostors in the test dataset.

        Parameters:
            train_dataset (pd.DataFrame): The training dataset.
            test_dataset (pd.DataFrame): The test dataset.
            impostor_detector (ImpostorDetector): The impostor detector to use.
        """
        with ProcessPoolExecutor() as executor:
            train_dataset["impostor_score"] = list(
                executor.map(impostor_detector.get_score, train_dataset["pair"])
            )
        print("Calculated impostor scores on training data.")

        # find threshold that best separates impostors from non-impostors in the training set (targets are in the 'same' column)
        args = {
            "rounds": impostor_detector.rounds,
            "top_n": impostor_detector.top_n,
            "impostor_technique": impostor_detector.impostor_technique,
            "upsample": impostor_detector.upsample,
            "n_impostors": impostor_detector.n_impostors,
            "dataset_name": self.dataset_name,
        }
        fpr, tpr, thresholds, best_f1_thres = self.plot_decision_threshold_impostor(
            scores=train_dataset["impostor_score"],
            labels=train_dataset["same"],
            title_kwargs=args,
        )

        best_f1_thres = np.round(best_f1_thres, 2)
        youdens_j_thres = np.round(self._get_opt_imp_threshold(fpr, tpr, thresholds), 2)
        print(
            f"Optimal threshold for impostor detection via Youden's J function: {youdens_j_thres:.2f}/ via best F1: {best_f1_thres:.2f}"
        )

        # work with test dataset
        impostor_detector.set_training_mode(
            False
        )  # set to False for validation: Use training set for impostor generation for fixed impostor technique‚
        with ProcessPoolExecutor() as executor:
            test_dataset["impostor_score"] = list(
                executor.map(impostor_detector.get_score, test_dataset["pair"])
            )

        for thres_name, thres in zip(
            ["Youden's J", "best F1"], [youdens_j_thres, best_f1_thres]
        ):
            args["threshold"] = thres
            print(f"Visualizing impostor scores with threshold: {thres_name} = {thres}")

            test_dataset["pred_same"] = test_dataset["impostor_score"] >= thres

            # 'same' is ground truth, 'pred_same' is prediction
            y_true = test_dataset["same"]
            y_pred = test_dataset["pred_same"]

            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=["Different authors", "Same author"]
            )

            disp.plot(cmap=plt.cm.Blues)
            title = self._format_title(
                base=f"Confusion Matrix on Test Data with threshold {thres_name}",
                kwargs=args,
            )
            plt.title(title)
            plt.tight_layout()
            save_path = self.savefig_base / "impostor_scores" / self.dataset_name
            filename = self._title2filename(title=title)
            save_path.mkdir(parents=True, exist_ok=True)
            for format in ["svg"]:  # "png",
                print(f"Saving confusion matrix to {save_path / filename}.{format}")
                plt.savefig(save_path / f"{filename}.{format}")
            plt.close()

    def plot_unmasking_curves(
        self, unmasking_detector: UnmaskingDetector, dataset, dataset_name: str = None
    ):
        """
        Plots unmasking curves for the given dataset.
        :param unmasking_detector: The UnmaskingDetector instance to use for plotting.
        :param dataset: The dataset containing pairs of texts and their labels.
        :param dataset_name: The name of the dataset for the plot title and save path.
        """
        with ProcessPoolExecutor() as executor:
            curves = list(executor.map(unmasking_detector.get_curves, dataset["pair"]))

        targets = dataset["same"].tolist()
        pal = sns.color_palette("husl", 2)
        seen_labels = set()
        for c, l in zip(curves, targets):
            if len(c) == 0:
                continue
            if type(c[0]) is not list:
                c = c[0].tolist()
            label = None
            if l not in seen_labels:
                label = f"Same Author"  #: {l}"
                seen_labels.add(l)
            plt.plot(c, color=pal[l], label=label, alpha=0.1, linewidth=1)

        plt.xlabel("Unmasking Rounds")
        title = self._format_title(
            base="Unmasking Curves", kwargs={"dataset_name": dataset_name}
        )
        plt.title(title)
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout(pad=0.8)
        plt.subplots_adjust(wspace=0.025, hspace=0.05)
        save_path = self.savefig_base / "unmasking_curves" / dataset_name
        save_path.mkdir(parents=True, exist_ok=True)
        filename = self._title2filename(title=title)
        plt.savefig(save_path / f"{filename}.svg")
        plt.close()

    def load_data(self, split: Literal["train", "test", "val"]):
        """
        Loads the dataset for visualization.

        :param split: The split of the dataset to load (train, test, or val).
        Returns:
            dataset: The loaded dataset.
        """
        try:
            if self.dataset_name == CONFIG.PAN25:
                return load_from_disk(
                    Path(__file__).resolve().parent.parent / CONFIG.PATH2PAN25
                )[split].to_pandas()
            elif self.dataset_name == CONFIG.PAN23:
                return load_from_disk(
                    Path(__file__).resolve().parent.parent / CONFIG.PATH2PAN23
                )[split].to_pandas()
            elif self.dataset_name == CONFIG.PAN20:
                return load_from_disk(
                    Path(__file__).resolve().parent.parent / CONFIG.PATH2PAN20
                )[split].to_pandas()
            elif self.dataset_name == CONFIG.KOPPEL:
                return load_from_disk(
                    Path(__file__).resolve().parent.parent / CONFIG.PATH2KOPPEL_WEBIS
                )[split].to_pandas()
            elif self.dataset_name == CONFIG.BLOG:
                return load_from_disk(
                    Path(__file__).resolve().parent.parent / CONFIG.PATH2BLOG
                )[split].to_pandas()
            elif self.dataset_name == CONFIG.STUDENT_ESSAYS:
                return load_from_disk(
                    Path(__file__).resolve().parent.parent / CONFIG.PATH2STUDENT_ESSAYS
                )[split].to_pandas()
            else:
                raise ValueError(
                    f"Dataset {self.dataset_name} is not supported for visualization."
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset {self.dataset_name} for split {split}: {e}"
            ) from e

    # ProcessPoolExecutor does not support self, so we need to use a static method
    @staticmethod
    def _metric_at_thresh(scores, targets, threshold):
        """
        Computes F1 and accuracy at a given threshold.

        :param scores: The scores to evaluate.
        :param targets: The true labels.
        :param threshold: The threshold to apply.
        :return: A tuple containing F1 score and accuracy.
        """
        preds = scores >= threshold
        return (f1_score(targets, preds), accuracy_score(targets, preds))

    def plot_decision_threshold_impostor(
        self, scores: pd.Series, labels: pd.Series, title_kwargs: dict = None
    ):
        """
        Plots the decision threshold for the impostor detector using ROC and Precision-Recall curves.
        :param scores: The impostor scores.
        :param labels: The true labels (same or different authors).
        :param title_kwargs: Additional keyword arguments for the plot title.
        :return: fpr, tpr, roc_thresholds, best threshold for F1 score.
        """
        assert isinstance(title_kwargs, dict), "title_kwargs must be a dictionary"
        assert len(scores) == len(
            labels
        ), f"scores and labels must have the same length, but len(scores)={len(scores)} != len(labels)={len(labels)}\nscores: {scores}\nlabels: {labels}"
        sys.path.append(os.path.abspath(".."))
        dataset_name = title_kwargs.get("dataset_name", self.dataset_name)
        n_impostors = title_kwargs.get("n_impostors", None)
        assert (
            dataset_name == self.dataset_name
        ), "dataset_name must match self.dataset_name"
        save_path = self.savefig_base / "impostor_scores" / dataset_name

        # pd.Series to numpy arrays
        labels = labels.values
        try:
            scores = np.concatenate(
                scores.values
            )  # each entry in scores is a one-element list
            if len(scores) != len(labels):
                print(
                    f"Warning: Length of scores ({len(scores)}) does not match length of labels ({len(labels)})."
                )
                print("scores:", scores)
                print("labels:", labels)
        except Exception as e:
            raise ValueError(f"Failed to concatenate (vis_detectors) scores: {e}.")

        # ROC Curve: Balanced classes or when you care about TPR vs. FPR
        # displayed for different thresholds
        # scores: probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers)
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html (05.06.2025)
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        fig = plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.gca().set_aspect("equal")
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel(
            "False Positive Rate $FPR = 1 - \\frac{{TN}}{{TN + FP}}$", fontsize=14
        )
        plt.ylabel(
            "True Positive Rate $TPR = R = \\frac{{TP}}{{TP + FN}}$", fontsize=14
        )
        title = self._format_title(base="ROC Curve", kwargs=title_kwargs)
        plt.title(title)
        plt.legend()

        # Precision-Recall Curve: 	Imbalanced
        # scores: non-thresholded measure of decisions, relative ranking of predictions
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html (05.06.2025)
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        plt.subplot(1, 2, 2)
        plt.gca().set_aspect("equal")
        plt.plot(recall, precision, label="PR Curve", color="orange")
        plt.ylim(0, 1)
        if self.dataset_name == CONFIG.BLOG:
            if not n_impostors or n_impostors == 50:
                plt.scatter(
                    0.34,
                    0.95,
                    marker="o",
                    color="red",
                    label=r"$\sigma*=unknown$, $50$ authors$^1$",
                )
            elif not n_impostors or n_impostors == 500:
                plt.scatter(
                    0.222,
                    0.902,
                    marker="x",
                    color="red",
                    label=r"$\sigma^*=0.8$, $500$ authors$^1$",
                )
            if not n_impostors or n_impostors in [50, 5000]:
                plt.annotate(
                    "1: Koppel et al. (2014)/ Blog dataset",
                    xy=(1.0, -0.2),
                    xycoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize=10,
                )

        plt.xlabel("Recall $\\frac{{TP}}{{TP + FN}}$", fontsize=14)
        plt.ylabel("Precision $\\frac{{TP}}{{TP + FP}}$", fontsize=14)
        title = self._format_title(base="Precision-Recall Curve", kwargs=title_kwargs)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        save_path.mkdir(parents=True, exist_ok=True)
        for format in ["svg"]:  # "png",
            figure_name = (
                f"roc_prec_recall_curve.{format}"
                if not title_kwargs
                else f"roc_prec_recall_curve_r{title_kwargs['rounds']}_top{title_kwargs['top_n']}_n_imp{title_kwargs['n_impostors']}.{format}"
            )
            plt.savefig(save_path / figure_name)
        plt.close(fig)

        # Threshold vs (1) F1 score, (2) Accuracy, (3) Precision, (4) Recall
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.subplots_adjust(hspace=0.5, wspace=0.25)  # hspace controls vertical spacing
        thresholds = np.linspace(min(scores), max(scores), 100)

        thresholds = np.linspace(min(scores), max(scores), 100)
        scores_np = np.array(scores)
        labels_np = np.array(labels)

        scores_list = [scores_np] * len(thresholds)
        labels_list = [labels_np] * len(thresholds)

        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    self._metric_at_thresh, scores_list, labels_list, thresholds
                )
            )
        f1s, accs = zip(*results)

        # Plot 1: F1 vs Threshold
        axes[0, 0].plot(thresholds, f1s)
        axes[0, 0].set_aspect("equal", "box")
        axes[0, 0].set_xlabel("Threshold")
        axes[0, 0].set_ylabel(
            "F1 Score $\\frac{{2 \\cdot P \\cdot R}}{{P + R}}$", fontsize=14
        )
        axes[0, 0].set_ylim(0, 1)
        title = self._format_title(base="Threshold vs F1 Score", kwargs=title_kwargs)
        axes[0, 0].set_title(title, fontsize=10)
        axes[0, 0].grid(True)

        # Plot 2: Accuracy vs Threshold
        axes[0, 1].plot(thresholds, accs)
        axes[0, 1].set_aspect("equal", "box")
        axes[0, 1].set_xlabel("Threshold")
        axes[0, 1].set_ylabel("Accuracy Score $\\frac{{TP + TN}}{{N}}$", fontsize=14)
        axes[0, 1].set_ylim(0, 1)
        title = self._format_title(
            base="Threshold vs Accuracy Score", kwargs=title_kwargs
        )
        axes[0, 1].set_title(title, fontsize=10)
        axes[0, 1].grid(True)
        if dataset_name == CONFIG.BLOG:
            axes[0, 1].axhline(
                y=0.874,
                linestyle="--",
                color="red",
                alpha=0.5,
                label=r"Fixed (Blog) Imposter Generation$^1$",
            )
            axes[0, 1].axhline(
                y=0.832,
                linestyle="--",
                color="violet",
                alpha=0.5,
                label=r"On-the-fly Imposter Generation$^1$",
            )
            axes[0, 1].annotate(
                "1: Koppel et al. (2014)/ Blog dataset",
                xy=(1.0, -0.2),
                xycoords="axes fraction",
                ha="right",
                va="center",
                fontsize=10,
            )
            axes[0, 1].legend()
        elif dataset_name == CONFIG.STUDENT_ESSAYS:
            axes[0, 1].axhline(
                y=0.731,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=r"Fixed (Student Essay) Imposter Generation$^1$",
            )
            axes[0, 1].annotate(
                "1: Koppel et al. (2014)/ Student Essay dataset",
                xy=(1.0, -0.2),
                xycoords="axes fraction",
                ha="right",
                va="center",
                fontsize=10,
            )
            axes[0, 1].legend()

        # Plot 3: Precision vs Threshold
        axes[1, 0].plot(pr_thresholds, precision[:-1])
        axes[1, 0].set_aspect("equal", "box")
        axes[1, 0].set_xlabel("Threshold")
        axes[1, 0].set_ylabel("Precision $\\frac{{TP}}{{TP + FP}}$", fontsize=14)
        axes[1, 0].set_ylim(0, 1)
        title = self._format_title(
            base="Threshold vs Precision Score", kwargs=title_kwargs
        )
        axes[1, 0].set_title(title, fontsize=10)
        axes[1, 0].grid(True)

        # Plot 4: Recall vs Threshold
        axes[1, 1].plot(pr_thresholds, recall[:-1])
        axes[1, 1].set_aspect("equal", "box")
        axes[1, 1].set_xlabel("Threshold")
        axes[1, 1].set_ylabel("Recall $\\frac{{TP}}{{TP + FN}}$", fontsize=14)
        axes[1, 1].set_ylim(0, 1)
        title = self._format_title(
            base="Threshold vs Recall Score", kwargs=title_kwargs
        )
        axes[1, 1].set_title(title, fontsize=10)
        axes[1, 1].grid(True)

        save_path.mkdir(parents=True, exist_ok=True)
        for format in ["svg"]:  # "png",
            filename = f"thres_vs_f1_acc_prec_recall.{format}"
            if title_kwargs:
                b = self._format_title(
                    base=f"thres_vs_f1_acc_prec_recall", kwargs=title_kwargs
                )
                filename = self._title2filename(title=b)
            plt.savefig(save_path / f"{filename}.{format}")

        plt.close(fig)
        return fpr, tpr, roc_thresholds, thresholds[1:-1][np.argmax(f1s[1:-1])]

    #####################################################################################################################
    def _run_fig_2_worker(
        self, n_imp, train_dataset, test_dataset, path2imp, args: dict
    ):
        print(f"Using {n_imp} impostors from {path2imp}")
        try:
            updated_args = args.copy()
            updated_args["n_impostors"] = n_imp
            precs, recs = self._fig_2_for_fixed_n_imposters(
                train_dataset, test_dataset, path2imp, n_imp, args=updated_args
            )
            return {"n_imp": n_imp, "precision": precs, "recall": recs}
        except Exception as e:
            print(f"[ERROR] Failed for n_imp = {n_imp}:\n{traceback.format_exc()}\n{e}")
            return None

    # ugly, but only for reproduction of Figure 2 from Koppel et al. (2014)
    def reproduce_fig2_prec_recall_dif_n_imp(self, args: dict) -> None:
        """
        Visualizes the impostor detection results via Precision-Recall curves for different numbers of impostors (cf. Figure 2 from Koppel et al. (2014)).
        """
        assert isinstance(args, dict), "args must be a dictionary"
        assert self.dataset_name in [
            CONFIG.BLOG,
            CONFIG.STUDENT_ESSAYS,
        ], "This method is only implemented for BLOG and Student Essays datasets."
        train_dataset, test_dataset = self._load_datasets(balanced=True)
        # TODO: Test on small data subsets
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
        print("Loaded datasets for Figure 2:", self.dataset_name)
        # could be initially different, bc args are from argparse which are irrespective from calling thsi function with defined dataset_name
        args["dataset_name"] = self.dataset_name
        n_imp_options = [50, 500, 5000]
        precisions, recalls = {}, {}
        path2imp = (
            Path(os.getcwd()).resolve() / CONFIG.PATH2BLOG
            if self.dataset_name == CONFIG.BLOG
            else Path(os.getcwd()).resolve() / CONFIG.PATH2STUDENT_ESSAYS
        )
        # Does not work, bc tfidf vectorizer isnt correctly initialized in ImpostorDetector
        # print("Start parallel computation for different n_impostors.")
        # with ProcessPoolExecutor() as executor:
        #     futures = {
        #         executor.submit(
        #             self._run_fig_2_worker,
        #             n_imp,
        #             train_dataset,
        #             test_dataset,
        #             path2imp,
        #             args,
        #         ): n_imp
        #         for n_imp in n_imp_options
        #     }

        #     for future in as_completed(futures):
        #         result = future.result()
        #         if result:
        #             precisions[result["n_imp"]] = result["precision"]
        #             recalls[result["n_imp"]] = result["recall"]
        #         else:
        #             print(
        #                 f"[ERROR] Failed to compute precision and recall for n_imp = {futures[future]}"
        #             )

        print("Start sequential computation for different n_impostors.")
        for n_imp in n_imp_options:
            result = self._run_fig_2_worker(
                n_imp, train_dataset, test_dataset, path2imp, args
            )
            if result:
                print(
                    f"Computed precision and recall for n_imp = {n_imp}: prec: {result['precision']}, recall: {result['recall']}"
                )
                precisions[result["n_imp"]] = result["precision"]
                recalls[result["n_imp"]] = result["recall"]
            else:
                print(
                    f"[ERROR] Failed to compute precision and recall for n_imp = {n_imp}"
                )

        fig = plt.figure()
        assert (
            list(precisions.keys()) == n_imp_options
        ), f"Expected precisions keys {n_imp_options}, but got {list(precisions.keys())}"
        assert (
            list(recalls.keys()) == n_imp_options
        ), f"Expected recalls keys {n_imp_options}, but got {list(recalls.keys())}"

        for n_imp in sorted(n_imp_options):  # start legend with 50, then 500, then 5000
            precision = precisions[n_imp]
            recall = recalls[n_imp]
            plt.plot(recall, precision, label="# impostors = " + str(n_imp))
        plt.ylim(0, 1)
        plt.gca().set_aspect("equal", "box")
        if self.dataset_name == CONFIG.BLOG:
            plt.scatter(
                0.34,
                0.95,
                marker="o",
                color="red",
                label=r"$\sigma*=unknown$, $50$ authors$^1$",
            )
            plt.scatter(
                0.222,
                0.902,
                marker="x",
                color="red",
                label=r"$\sigma^*=0.8$, $500$ authors$^1$",
            )
            plt.annotate(
                "1: Koppel et al. (2014)/ Blog dataset",
                xy=(1.0, -0.5),
                xycoords="axes fraction",
                ha="right",
                va="center",
                fontsize=10,
            )

        plt.xlabel("Recall $\\frac{{TP}}{{TP + FN}}$", fontsize=14)
        plt.ylabel("Precision $\\frac{{TP}}{{TP + FP}}$", fontsize=14)
        diff_n_imp_args = args.copy()
        # remove n_impostors from args for title, bc we compare different n_imp
        diff_n_imp_args.pop("n_impostors", None)
        title = self._format_title(
            base="Precision-Recall Curve", kwargs=diff_n_imp_args
        )
        plt.title(title)
        plt.legend(
            loc="best", bbox_to_anchor=(1.0, 0.5), borderaxespad=0.0, fontsize=10
        )
        plt.tight_layout()
        save_path = (
            self.savefig_base / "impostor_scores" / self.dataset_name / "koppel_fig2"
        )
        save_path.mkdir(parents=True, exist_ok=True)
        for format in ["svg"]:  # "png",
            figure_name = (
                f"roc_prec_recall_curve_dif_n_imp.{format}"
                if len(args.keys()) == 0
                else f"roc_prec_recall_curve_r{args['rounds']}_top{args['top_n']}_dif_n_imp.{format}"
            )
            plt.savefig(save_path / figure_name)
        plt.close(fig)

    def _fig_2_for_fixed_n_imposters(
        self, train_dataset, test_dataset, path2imp, n_imp, args: dict
    ):
        assert isinstance(args, dict), "args must be a dictionary"
        assert (
            args.get("n_impostors", n_imp) == n_imp
        ), f"n_impostors={args['n_impostors']} in args must match n_imp={n_imp} in function call"

        with ThreadPoolExecutor() as executor:  # do not nest ProcessPoolExecutor, use ThreadPoolExecutor instead in inner loop
            train_dataset["impostor_score"] = list(
                executor.map(
                    self._helper_impostor,
                    [path2imp] * len(train_dataset),
                    train_dataset["pair"],
                    ["fixed"] * len(train_dataset),
                    [True] * len(train_dataset),  # training mode
                    [n_imp] * len(train_dataset),  # n_imp
                )
            )
        print("Calculated impostor scores on training data.")

        fpr, tpr, thresholds, best_f1_thres = self.plot_decision_threshold_impostor(
            scores=train_dataset["impostor_score"],
            labels=train_dataset["same"],
            title_kwargs=args,
        )
        print(
            f"Plotted decision threshold for impostor detection for n_imp: {n_imp} (Fig 2)."
        )

        best_f1_thres = np.round(best_f1_thres, 2)
        youdens_j_thres = np.round(self._get_opt_imp_threshold(fpr, tpr, thresholds), 2)
        print(
            f"Optimal threshold for impostor detection via Youden's J function: {youdens_j_thres:.2f}/ via best F1: {best_f1_thres:.2f}"
        )

        # work with test dataset
        with ThreadPoolExecutor() as executor:
            test_dataset["impostor_score"] = list(
                executor.map(
                    self._helper_impostor,
                    [path2imp] * len(test_dataset),
                    test_dataset["pair"],
                    ["fixed"] * len(test_dataset),
                    [False] * len(test_dataset),  # testing mode
                    [n_imp] * len(test_dataset),  # n_imp
                )
            )

        precision, recall, pr_thresholds = precision_recall_curve(
            test_dataset["same"], test_dataset["impostor_score"]
        )

        for thres_name, thres in zip(
            ["Youden's J", "best F1"], [youdens_j_thres, best_f1_thres]
        ):
            args["threshold"] = thres
            print(f"Visualizing impostor scores with threshold: {thres_name} = {thres}")

            test_dataset["pred_same"] = test_dataset["impostor_score"] >= thres

            # 'same' is ground truth, 'pred_same' is prediction
            y_true = test_dataset["same"]
            y_pred = test_dataset["pred_same"]

            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Different authors", "Same author"],
            )

            disp.plot(cmap=plt.cm.Blues)
            title = self._format_title(
                base=f"Confusion Matrix on Test Data with threshold {thres_name}",
                kwargs=args,
            )
            plt.title(title)
            plt.tight_layout()
            save_path = (
                self.savefig_base
                / "impostor_scores"
                / self.dataset_name
                / "koppel_fig2"
            )
            filename = self._title2filename(title=title)
            save_path.mkdir(parents=True, exist_ok=True)
            for format in ["svg"]:  # "png",
                print(f"Saving confusion matrix to {save_path / filename}.{format}")
                plt.savefig(save_path / f"{filename}.{format}")
            plt.close()
        return precision, recall

    #################################################################################

    def _helper_impostor(
        self, path2imp, pair, imp_gen: str, training_mode=True, n_imp: int = 50
    ):
        impostor_detector = ImpostorDetector(
            impostor_technique=imp_gen,
            n_impostors=n_imp,
            rounds=100,  # cf. pg. 181, Koppel et al. (2014)
            top_n=100000,  # cf. pg. 179, Koppel et al. (2014)
            path2imp=path2imp,
            upsample=False,
            real_time_generation=False,  # TODO: turn True, otherwise on-the-fly generation is not possible (currently too much data for too few free api calls)
        )
        impostor_detector.set_training_mode(training_mode)
        res = impostor_detector._get_score_impl(pair)
        return [r[0] for r in res] if isinstance(res, list) else res[0]

    def _run_fig_4_worker(self, imp_gen, train_dataset, test_dataset, path2imp):
        try:
            print(
                f"Using {imp_gen} impostor generation with path to imposters: {path2imp}"
            )
            # initialize impostor detector
            impostor_detector = ImpostorDetector(
                impostor_technique=imp_gen,
                n_impostors=50,
                rounds=100,  # cf. pg. 181, Koppel et al. (2014)
                top_n=100000,  # cf. pg. 179, Koppel et al. (2014)
                path2imp=path2imp,
                upsample=False,
                real_time_generation=False,  # TODO: turn True, otherwise on-the-fly generation is not possible (currently too much data for too few free api calls)
            )
            print(
                f"Initialized impostor detector with {imp_gen} impostors generation technique."
            )
            with ThreadPoolExecutor() as executor:
                train_dataset["impostor_score"] = list(
                    executor.map(
                        self._helper_impostor,
                        [path2imp] * len(train_dataset),
                        train_dataset["pair"],
                        [imp_gen] * len(train_dataset),
                    )
                )
            print(
                "Calculated impostor scores on training data for imposter generation:",
                imp_gen,
            )

            # find threshold that best separates impostors from non-impostors in the training set (targets are in the 'same' column)
            args = {
                "rounds": impostor_detector.rounds,
                "top_n": impostor_detector.top_n,
                "impostor_technique": impostor_detector.impostor_technique,
                "upsample": impostor_detector.upsample,
                "n_impostors": impostor_detector.n_impostors,
            }
            args["dataset_name"] = self.dataset_name
            fpr, tpr, thresholds, best_f1_thres = self.plot_decision_threshold_impostor(
                scores=train_dataset["impostor_score"],
                labels=train_dataset["same"],
                title_kwargs=args,
            )

            best_f1_thres = np.round(best_f1_thres, 2)
            youdens_j_thres = np.round(
                self._get_opt_imp_threshold(fpr, tpr, thresholds), 2
            )
            print(
                f"Optimal threshold for impostor detection via Youden's J function: {youdens_j_thres:.2f}/ via best F1: {best_f1_thres:.2f}"
            )

            # work with test dataset
            impostor_detector.set_training_mode(
                False
            )  # set to False for validation: Use training set for impostor generation for fixed impostor technique
            with ProcessPoolExecutor() as executor:
                test_dataset["impostor_score"] = list(
                    executor.map(
                        self._helper_impostor,
                        [path2imp] * len(train_dataset),
                        test_dataset["pair"],
                        [imp_gen] * len(train_dataset),
                        [False]
                        * len(train_dataset),  # training_mode=False for test set
                    )
                )

            # each entry in scores is a one-element list
            test_dataset["impostor_score"] = np.concatenate(
                test_dataset["impostor_score"].values
            )

            # same author pairs
            same_author_precisions, same_author_recalls, same_pr_thresholds = (
                precision_recall_curve(
                    test_dataset["same"],
                    test_dataset["impostor_score"],
                    pos_label=1,
                )
            )

            # different author pairs
            (
                different_author_precisions,
                different_author_recalls,
                different_pr_thresholds,
            ) = precision_recall_curve(
                test_dataset["same"],
                test_dataset["impostor_score"],
                pos_label=0,
            )

            return {
                "imp_gen": imp_gen,
                "same_author_precisions": same_author_precisions,
                "same_author_recalls": same_author_recalls,
                "same_pr_thresholds": same_pr_thresholds,
                "different_author_precisions": different_author_precisions,
                "different_author_recalls": different_author_recalls,
                "different_pr_thresholds": different_pr_thresholds,
            }
        except Exception as e:
            print(f"[ERROR] Failed for imp_gen = {imp_gen}:\n{traceback.format_exc()}")
            return None

    # ugly, but only for reproduction of Figure 4 a, b from Koppel et al. (2014)
    def reproduce_fig4_prec_recall_dif_imp_appr(
        self,
        args: dict,
        save_path: Optional[Path],
        imp_gen_options: list = ["fixed", "on-the-fly"],
    ) -> None:
        """
        Visualizes the impostor detection results via Precision-Recall curves for different impostor generation techniques (cf. Figures 4 a, b from Koppel et al. (2014)).
        """
        print(
            "Reproducing Figure 4 with different impostor generation techniques:",
            imp_gen_options,
        )
        assert isinstance(args, dict), "args must be a dictionary"
        if not save_path:
            save_path = (
                self.savefig_base
                / "impostor_scores"
                / self.dataset_name
                / "koppel_fig4"
            )
        save_path.mkdir(parents=True, exist_ok=True)
        pr_save_path = save_path / "precision_recall_values"
        pr_save_path.mkdir(parents=True, exist_ok=True)
        train_dataset, test_dataset = self._load_datasets(balanced=True)

        # TODO: Test on small data subsets
        # train_dataset = pd.concat(
        #     [
        #         train_dataset.loc[train_dataset["same"]].head(5),
        #         train_dataset.loc[~train_dataset["same"]].head(5),
        #     ],
        #     ignore_index=False,
        # )

        # test_dataset = pd.concat(
        #     [
        #         test_dataset.loc[test_dataset["same"]].head(5),
        #         test_dataset.loc[~test_dataset["same"]].head(5),
        #     ],
        #     ignore_index=False,
        # )

        # could be initially different, bc args are from argparse which are irrespective from calling thsi function with defined dataset_name
        args["dataset_name"] = self.dataset_name
        baselines = [
            "unsupervised baseline min-max",
            "unsupervised baseline cosine",
            "supervised baseline",
        ]
        same_author_precisions, same_author_recalls = {}, {}
        different_author_precisions, different_author_recalls = {}, {}
        path2imp = (
            Path(os.getcwd()).resolve() / CONFIG.PATH2BLOG
            if self.dataset_name == CONFIG.BLOG
            else Path(os.getcwd()).resolve() / CONFIG.PATH2STUDENT_ESSAYS
        )
        print(
            "Start sequential computation (else OOM) for different impostor generation techniques."
        )
        # with ProcessPoolExecutor() as executor:
        #     futures = {
        #         executor.submit(
        #             self._run_fig_4_worker,
        #             imp_gen,
        #             train_dataset,
        #             test_dataset,
        #             path2imp,
        #         ): imp_gen
        #         for imp_gen in imp_gen_options
        #     }

        #     for future in as_completed(futures):
        #         result = future.result()
        #         if result:
        for imp_gen in imp_gen_options:
            result = self._run_fig_4_worker(
                imp_gen, train_dataset, test_dataset, path2imp
            )
            assert result is not None, f"Failed for imp_gen: {imp_gen}"
            same_author_precisions[result["imp_gen"]] = result[
                "same_author_precisions"
            ][:-1]
            same_author_recalls[result["imp_gen"]] = result["same_author_recalls"][:-1]
            # save precision and recall for same pairs
            self._save_prec_recall_values(
                pr_save_path=pr_save_path,
                appr_name=result["imp_gen"],
                precision_vals=same_author_precisions[result["imp_gen"]],
                recall_vals=same_author_recalls[result["imp_gen"]],
                pr_thresholds=result["same_pr_thresholds"],
                portion="same",
            )

            different_author_precisions[result["imp_gen"]] = result[
                "different_author_precisions"
            ][:-1]
            different_author_recalls[result["imp_gen"]] = result[
                "different_author_recalls"
            ][:-1]
            # save precision and recall for different pairs
            self._save_prec_recall_values(
                pr_save_path=pr_save_path,
                appr_name=result["imp_gen"],
                precision_vals=different_author_precisions[result["imp_gen"]],
                recall_vals=different_author_recalls[result["imp_gen"]],
                pr_thresholds=result["different_pr_thresholds"],
                portion="different",
            )
            print(
                f"Computed and saved precision and recall for {result['imp_gen']} impostor generation. "
            )

        for baseline_name, baseline in zip(
            baselines,
            [
                UnSupervisedImpostorBaseline(
                    use_cosine_simiarity=False, dataset_name=self.dataset_name
                ),
                UnSupervisedImpostorBaseline(
                    use_cosine_simiarity=True, dataset_name=self.dataset_name
                ),
                SupervisedImpostorBaseline(dataset_name=self.dataset_name),
            ],
        ):
            try:
                preds = baseline.get_score(test_dataset["pair"])
                # get_prediction(test_dataset["pair"])
            except Exception as e:
                print(
                    f"[ERROR] Failed for baseline {baseline_name}:\n{e}\n{traceback.format_exc()}\nReloading datasets..."
                )
                train_dataset, test_dataset = self._load_datasets(balanced=True)
                # TODO: Test on small data subsets
                # test_dataset = pd.concat(
                #     [
                #         test_dataset.loc[test_dataset["same"]].head(5),
                #         test_dataset.loc[~test_dataset["same"]].head(5),
                #     ],
                #     ignore_index=False,
                # )
                preds = baseline.get_score(test_dataset["pair"])

            test_dataset[f"{baseline_name.replace(' ','_')}_score"] = preds

            # same author pairs
            precision, recall, pr_thresholds = precision_recall_curve(
                test_dataset["same"],
                test_dataset[f"{baseline_name.replace(' ','_')}_score"],
                pos_label=1,
            )
            # save precision and recall for same pairs
            self._save_prec_recall_values(
                pr_save_path=pr_save_path,
                appr_name=baseline_name,
                precision_vals=precision[:-1],
                recall_vals=recall[:-1],
                pr_thresholds=pr_thresholds,
                portion="same",
            )
            print(
                "Number of same author pairs:", len(test_dataset[test_dataset["same"]])
            )
            print(
                "Number of same author precisions/recalls:",
                len(precision),
                "/",
                len(recall),
            )
            same_author_precisions[baseline_name.replace(" ", "_")] = precision[:-1]
            same_author_recalls[baseline_name.replace(" ", "_")] = recall[:-1]

            # different author pairs
            precision, recall, pr_thresholds = precision_recall_curve(
                test_dataset["same"],
                test_dataset[f"{baseline_name.replace(' ','_')}_score"],
                pos_label=0,
            )
            # save precision and recall for different pairs
            self._save_prec_recall_values(
                pr_save_path=pr_save_path,
                appr_name=baseline_name,
                precision_vals=precision[:-1],
                recall_vals=recall[:-1],
                pr_thresholds=pr_thresholds,
                portion="different",
            )
            print(
                "Number of different author pairs:",
                len(test_dataset[~test_dataset["same"]]),
            )
            print(
                "Number of different author precisions/recalls:",
                len(precision),
                "/",
                len(recall),
            )
            different_author_precisions[baseline_name.replace(" ", "_")] = precision[
                :-1
            ]
            different_author_recalls[baseline_name.replace(" ", "_")] = recall[:-1]

        # Precision-Recall Curve: 	Imbalanced
        # scores: non-thresholded measure of decisions, relative ranking of predictions
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html (05.06.2025)
        print(
            "DEBUG Plotting Precision-Recall Curves for different impostor generation techniques:",
            same_author_precisions.keys(),
        )

        line_styles = dict(zip(baselines, [":", "--", "-."]))
        for kind, data in zip(
            ["Same Author", "Different Author"],
            [
                (same_author_precisions, same_author_recalls),
                (different_author_precisions, different_author_recalls),
            ],
        ):
            print(f"Plotting Precision-Recall Curve for {kind} pairs")
            precisions, recalls = data
            fig = plt.figure()
            for imp_gen, precision in precisions.items():
                recall = recalls[imp_gen]
                plt.plot(
                    recall,
                    precision,
                    label=imp_gen,
                    linestyle=(
                        "-"
                        if imp_gen in imp_gen_options
                        else (
                            line_styles[imp_gen]
                            if imp_gen in line_styles.keys()
                            else "--"
                        )
                    ),
                )

            plt.ylim(0, 1)
            plt.gca().set_aspect("equal")
            plt.xlabel("Recall $\\frac{{TP}}{{TP + FN}}$", fontsize=14)
            plt.ylabel("Precision $\\frac{{TP}}{{TP + FP}}$", fontsize=14)
            title = self._format_title(
                base=f"Precision-Recall Curve for {kind} Data", kwargs=args
            )
            plt.title(title)
            plt.legend()
            plt.tight_layout()

            for format in ["svg"]:  # "png",
                figure_name = (
                    f"roc_prec_recall_curve_dif_{kind.replace(' ', '_')}_imp_gen.{format}"
                    if len(args.keys()) == 0
                    else f"roc_prec_recall_curve_r{args['rounds']}_top{args['top_n']}_{kind.replace(' ', '_')}_dif_imp_gen.{format}"
                )
                plt.savefig(save_path / figure_name)
            plt.close(fig)

    def _save_prec_recall_values(
        self,
        pr_save_path: Path,
        appr_name: str,
        precision_vals: list,
        recall_vals: list,
        pr_thresholds: list,
        portion: str = "same",
    ):
        print("Saving precision and recall values for", appr_name, portion)
        assert (
            len(precision_vals) == len(recall_vals) == len(pr_thresholds)
        ), "Precision, recall, and thresholds must have the same length."
        assert portion in [
            "same",
            "different",
        ], "Portion must be one of 'same', or 'different'."
        df = pd.DataFrame(
            {
                "threshold": pr_thresholds,
                "precision": precision_vals,
                "recall": recall_vals,
            }
        )
        df.to_csv(
            pr_save_path / f"{appr_name.replace(' ', '_')}_{portion}_prec_rec.csv",
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare detectors.")
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Number of rounds (default: %(default)s)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=100000,
        help="Number of top space-free ngrams to consider (default: %(default)s)",
    )

    parser.add_argument(
        "--n_impostors",
        type=int,
        default=50,
        help="Number of impostors to generate per candidate (default: %(default)s)",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=[CONFIG.PAN20, CONFIG.PAN23, CONFIG.PAN25, CONFIG.KOPPEL, CONFIG.BLOG],
        default=CONFIG.BLOG,
        help="Dataset to use for visualization (default: %(default)s)",
    )

    parser.add_argument(
        "--impostor_technique",
        type=str,
        choices=[
            "llm",
            "text_len",
            "n_docs",
            "on-the-fly",
            "blogs",
            "fixed",
            "content",
        ],
        default="fixed",
        help="impostor technique to use (default: %(default)s)",
    )

    parser.add_argument(
        "--path2imp",
        type=str,
        default=Path(os.getcwd()).resolve().parent / CONFIG.PATH2BLOG,
        help="Path to the impostor dataset (default: %(default)s)",
    )

    parser.add_argument(
        "--balanced",
        type=bool,
        default=True,
        help="Whether the number of same and different author pairs from dataset is balanced (default: %(default)s)",
    )

    parser.add_argument(
        "--upsample",
        type=bool,
        default=True,
        help="Whether texts below 500 words should be skipped or upsampled (default: %(default)s)",
    )

    parser.add_argument(
        "--fig2reproduce",
        type=int,
        default=2,
        help="Number of Figure from Koppel et al. (2014) to reproduce (default: %(default)s)",
    )

    args = parser.parse_args()
    print("Arguments for impostor detector:", args)

    impostor = ImpostorDetector(
        impostor_technique="fixed",
        n_impostors=args.n_impostors,
        rounds=args.rounds,
        top_n=args.top_n,
        path2imp=args.path2imp,
        upsample=args.upsample,
    )

    # vis_det = VisDetectors(
    #     dataset_name=args.dataset_name,
    #     detectors=[impostor],
    # )
    # print("impostor Detector initialized.")
    # vis_det.visualize(balanced=args.balanced)

    # reproduction of Figure 2/ 4 from Koppel et al. (2014)
    our_figure_impostor_options = [
        # "on-the-fly",
        "fixed",
        "naive_llm",
        "non_naive_llm",
    ]
    fig = args.fig2reproduce
    assert fig in [
        2,
        4,
        5,  # not really figure 5, but figure 4 with our contributions (LLM based impostors)
    ], "Only Figures 2 and 4 can be reproduced from Koppel et al. (2014)."
    args = vars(args)  # namespace to dict conversion
    print(
        f"Reproducing Figure {fig} from Koppel et al. (2014) on BLOG and STUDENT data."
    )
    vis_det = VisDetectors(
        dataset_name=CONFIG.BLOG,
        detectors=[impostor],
    )
    print(f"impostor Detector initialized for figure {fig}.")
    print(
        "Figr 2 currently sequential computation for different n_impostors to find bottleneck."
    )
    # if fig == 2:
    #     vis_det.reproduce_fig2_prec_recall_dif_n_imp(args=args)
    # elif fig == 4:
    #     vis_det.reproduce_fig4_prec_recall_dif_imp_appr(args=args, save_path=None)
    # elif fig == 5:
    #     # not really figure 5, but figure 4 with our contributions (LLM based impostors)
    #     vis_det.reproduce_fig4_prec_recall_dif_imp_appr(
    #         args=args,
    #         imp_gen_options=our_figure_impostor_options,
    #         save_path=vis_det.savefig_base
    #         / "impostor_scores"
    #         / vis_det.dataset_name
    #         / "our_contributions_scores",
    #     )
    # print(f"Finished reproducing Figure {fig} from Koppel et al. (2014) on BLOG data.")

    vis_det = VisDetectors(
        dataset_name=CONFIG.STUDENT_ESSAYS,
        detectors=[impostor],
    )
    print(f"impostor Detector initialized for fig {fig}.")
    if fig == 2:
        vis_det.reproduce_fig2_prec_recall_dif_n_imp(args=args)
    elif fig == 4:
        vis_det.reproduce_fig4_prec_recall_dif_imp_appr(args=args, save_path=None)
    elif fig == 5:
        # not really figure 5, but figure 4 with our contributions (LLM based impostors)
        vis_det.reproduce_fig4_prec_recall_dif_imp_appr(
            args=args,
            imp_gen_options=our_figure_impostor_options,
            save_path=vis_det.savefig_base
            / "impostor_scores"
            / vis_det.dataset_name
            / "our_contributions_scores",
        )
    print(
        f"Finished reproducing Figure {fig} from Koppel et al. (2014) on STUDENT data."
    )
