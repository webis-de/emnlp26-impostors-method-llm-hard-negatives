import argparse
import os
from pathlib import Path
import re
import sys
from typing import Literal
from datasets import load_from_disk
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_recall_curve, roc_curve, accuracy_score
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CONFIG
from detectors.impostor import ImpostorDetector
from detectors.unmasking import UnmaskingDetector

class VisDetectors:
    """
    A class to compare detectors on the same dataset.
    It visualizes the detection results using ROC and Precision-Recall curves, confusion matrices, and decision thresholds.
    """

    def __init__(self, dataset, imposter_args=None, detectors:list = None) -> None:
        """
        Initializes the VisDetectors class.
        :param dataset: The name of the dataset to use for visualization.
        :param imposter_args: Arguments for the ImpostorDetector.
        :param detectors: List of detectors to visualize. Currenrly implemented: Unmasking and Imposter methods.
        """
        self.dataset = dataset
        self.imposter_args = imposter_args 
        self.detectors = detectors if detectors is not None else []
        self.savefig_base = Path.cwd() / CONFIG.SAVE_PATH   # save path relative to the current working directory
  
    def _format_title(self, base, kwargs):
        """
        Formats the title for the plots.
        :param base: The base title.
        :param kwargs: Additional keyword arguments to include in the title. Structured as a dictionary.
        :return: A formatted title string excluding path2imp, because paths are too long.
        """
        items = [f"{k}={v}" for k, v in kwargs.items() if k != 'path2imp']
        # newline after every 2 items, for better readability
        lines = []
        for i in range(0, len(items), 2):
            lines.append(", ".join(items[i:i+2]))
        return base + "\n" + "\n".join(lines)


    def visualize(self, balanced:bool = False) -> None:
        """
        Visualizes the detection results.

        :param balanced: Whether the number of same and different author pairs from dataset should be balanced (i.e. sampling strategy).
        """
        datasets = {}
        for detector in self.detectors:
            if detector == CONFIG.IMPOSTER: # imposter finds threshold on training data
                datasets['train'] = self.load_data(split='train')
            datasets['test'] = self.load_data(split='test') # scores obtained on test data
            if balanced:
                for split, df in datasets.items():
                    size_smaller_class = df['same'].value_counts().min()
                    # ensure target class 'same' is present after being used for grouping
                    datasets[split] = (
                        df.groupby('same', group_keys=False)
                        .apply(lambda x: x.sample(size_smaller_class, random_state=42).assign(same=x['same'].iloc[0]))
                        .reset_index(drop=True)
                    )

                if detector == CONFIG.IMPOSTER:
                    train_dataset = datasets['train']
                    test_dataset = datasets['test']
                    if train_dataset.empty or test_dataset.empty:
                        raise ValueError("Train or test dataset is empty. Cannot visualize imposters.")
                    self.visualize_imposters(train_dataset, test_dataset, dataset_name=self.dataset)
                elif detector == CONFIG.UNMASKING:
                    assert not datasets['test'].empty, "Test dataset is empty. Cannot visualize unmasking curves."
                    self.plot_unmasking_curves(dataset=datasets['test'], dataset_name=self.dataset)
                else:
                    raise ValueError(f"Detector {detector} is not supported for visualization.")
    
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

        # validity mask: finite and within [0, 1]
        valid_mask = (
            np.isfinite(fpr) & np.isfinite(tpr) & np.isfinite(thresholds) &
            (fpr >= 0) & (fpr <= 1) & (tpr >= 0) & (tpr <= 1)
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


    def visualize_imposters(self, train_dataset, test_dataset, dataset_name:str) -> None:
        """
        Visualizes the imposter detection results.
        Uses training data to find a threshold for imposter detection and then visualizes the imposters in the test dataset.

        Parameters:
            train_dataset (pd.DataFrame): The training dataset.
            test_dataset (pd.DataFrame): The test dataset.
        """
        impostor_det = ImpostorDetector(**self.imposter_args)
        with ProcessPoolExecutor() as executor:
            train_dataset['imposter_score'] = list(executor.map(impostor_det.get_scores, train_dataset['pair']))

        # find threshold that best separates imposters from non-imposters in the training set (targets are in the 'same' column)
        args = self.imposter_args.copy()
        args['dataset'] =  dataset_name
        fpr, tpr, thresholds = self.plot_decision_threshold_imposter(scores=train_dataset['imposter_score'], labels=train_dataset['same'], title_kwargs=args)

        self.imposter_args['threshold'] = self._get_opt_imp_threshold(fpr, tpr, thresholds)

        # work with test dataset
        with ProcessPoolExecutor() as executor:
            test_dataset['imposter_score'] = list(executor.map(impostor_det.get_scores, test_dataset['pair']))
        test_dataset['pred_same'] = test_dataset['imposter_score'] >= self.imposter_args['threshold']

        # 'same' is ground truth, 'pred_same' is prediction
        y_true = test_dataset['same']
        y_pred = test_dataset['pred_same']

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different authors", "Same author"])

        disp.plot(cmap=plt.cm.Blues)
        title = self._format_title(base="Confusion Matrix on Test Data", kwargs=self.imposter_args)
        plt.title(title)
        plt.tight_layout()
        save_path = self.savefig_base / 'impostor_scores' / self.dataset
        filename = title.replace(',','').replace('\n', '_').replace(' ', '_')
        plt.savefig((save_path / filename).with_suffix('.png'))
        plt.close()

    def plot_unmasking_curves(self, dataset, dataset_name:str=None):
        """
        Plots unmasking curves for the given dataset.
        :param dataset: The dataset containing pairs of texts and their labels.
        :param dataset_name: The name of the dataset for the plot title and save path.
        """
        unmask_det = UnmaskingDetector()
        with ProcessPoolExecutor() as executor:
            curves = list(executor.map(unmask_det.get_curves, dataset['pair']))

        targets = dataset['same'].tolist()
        pal = sns.color_palette('husl', 2) 
        seen_labels = set()
        for c,l in zip(curves, targets):
            if len(c) == 0:
                continue
            c = c[0].tolist()
            label = None
            if l not in seen_labels:
                label = f"Same Author: {l}"
                seen_labels.add(l)
            plt.plot(c, color=pal[l], label=label, alpha=0.1, linewidth=1)

        plt.xlabel('Unmasking Rounds')
        title = self._format_title(base="Unmasking Curves", kwargs={'dataset_name':dataset_name})
        plt.title(title)
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout(pad=0.8)
        plt.subplots_adjust(wspace=.025, hspace=.05)
        save_path = self.savefig_base / 'unmasking_curves' / dataset_name
        save_path.mkdir(parents=True, exist_ok=True)
        filename = title.replace('\n', '_').replace(' ', '_')
        plt.savefig((save_path / filename).with_suffix('.png'))
        plt.close()

    def load_data(self, split:Literal['train', 'test', 'val']):
        """
        Loads the dataset for visualization.

        :param split: The split of the dataset to load (train, test, or val).
        Returns:
            dataset: The loaded dataset.
        """
        try:
            if self.dataset == CONFIG.PAN25:
                return load_from_disk(os.path.join(os.path.abspath("."), CONFIG.PATH2PAN25))[split].to_pandas()
            elif self.dataset == CONFIG.PAN23:
                return load_from_disk(os.path.join(os.path.abspath("."), CONFIG.PATH2PAN23))[split].to_pandas()
            elif self.dataset == CONFIG.PAN20:
                return load_from_disk(os.path.join(os.path.abspath("."), CONFIG.PATH2PAN20))[split].to_pandas()
            elif self.dataset == CONFIG.KOPPEL:
                return load_from_disk(os.path.join(os.path.abspath("."), CONFIG.PATH2KOPPEL_WEBIS))[split].to_pandas()
            elif self.dataset == CONFIG.BLOG:
                return load_from_disk(os.path.join(os.path.abspath("."), CONFIG.PATH2BLOG))[split].to_pandas()
            else:
                raise ValueError(f"Dataset {self.dataset} is not supported for visualization.")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {self.dataset} for split {split}: {e}") from e
        
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
        return (
            f1_score(targets, preds),
            accuracy_score(targets, preds)
        )
    
    def plot_decision_threshold_imposter(self, scores:pd.Series, labels:pd.Series, title_kwargs:dict=None):
        """
        Plots the decision threshold for the imposter detector using ROC and Precision-Recall curves.
        :param scores: The imposter scores.
        :param labels: The true labels (same or different authors).
        :param title_kwargs: Additional keyword arguments for the plot title.
        :return: fpr, tpr, roc_thresholds
        """
        sys.path.append(os.path.abspath(".."))
        save_path = self.savefig_base / 'impostor_scores' / title_kwargs.get('dataset', 'unknown')
        save_path.mkdir(parents=True, exist_ok=True)

        # pd.Series to numpy arrays
        labels = labels.values
        scores = np.concatenate(scores.values)  # each entry in scores is a one-element list

        # ROC Curve: Balanced classes or when you care about TPR vs. FPR
        # displayed for different thresholds
        # scores: probability estimates of the positive class, confidence values, or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers)
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html (05.06.2025)
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate $FPR = 1 - \\frac{{TN}}{{TN + FP}}$", fontsize=14)
        plt.ylabel("True Positive Rate $TPR = R = \\frac{{TP}}{{TP + FN}}$", fontsize=14)
        title = self._format_title(base="ROC Curve", kwargs=title_kwargs)
        plt.title(title)
        plt.legend()

        # Precision-Recall Curve: 	Imbalanced 
        # scores: non-thresholded measure of decisions, relative ranking of predictions
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html (05.06.2025)
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label='PR Curve', color='orange')
        plt.scatter(0.34, 0.95, marker='o', color='red', label=r'Koppel et. Al. (2014) for $\sigma*=unknown$, $50$ authors') 
        plt.scatter(0.222, 0.902, marker='x', color='red', label=r'Koppel et. Al. (2014) for $\sigma*=0.8$, $500$ authors') 
        plt.xlabel("Recall $\\frac{{TP}}{{TP + FN}}$", fontsize=14)
        plt.ylabel("Precision $\\frac{{TP}}{{TP + FP}}$", fontsize=14)
        title = self._format_title(base="Precision-Recall Curve", kwargs=title_kwargs)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        figure_name = "roc_prec_recall_curve.png" if not title_kwargs else f"roc_prec_recall_curve_r{title_kwargs['rounds']}_top{title_kwargs['top_n']}.png"
        savefig = os.path.join(save_path, figure_name)
        plt.savefig(savefig)
        plt.close(fig)

        # Threshold vs (1) F1 score, (2) Accuracy, (3) Precision, (4) Recall
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)  # hspace controls vertical spacing
        thresholds = np.linspace(min(scores), max(scores), 100)

        thresholds = np.linspace(min(scores), max(scores), 100)
        scores_np = np.array(scores)
        labels_np = np.array(labels)
        
        scores_list = [scores_np] * len(thresholds)
        labels_list = [labels_np] * len(thresholds)

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self._metric_at_thresh, scores_list, labels_list, thresholds))
        f1s, accs = zip(*results)

        # Plot 1: F1 vs Threshold
        axes[0, 0].plot(thresholds, f1s)
        axes[0, 0].set_xlabel("Threshold")
        axes[0, 0].set_ylabel("F1 Score $\\frac{{2 \\cdot P \\cdot R}}{{P + R}}$", fontsize=14)
        title = self._format_title(base="Threshold vs F1 Score", kwargs=title_kwargs)
        axes[0, 0].set_title(title)
        axes[0, 0].grid(True)

        # Plot 2: Accuracy vs Threshold
        axes[0, 1].plot(thresholds, accs)
        axes[0, 1].set_xlabel("Threshold")
        axes[0, 1].set_ylabel("Accuracy Score $\\frac{{TP + TN}}{{N}}$", fontsize=14)
        title = self._format_title(base="Threshold vs Accuracy Score", kwargs=title_kwargs)
        axes[0, 1].set_title(title)
        axes[0, 1].grid(True)

        # Plot 3: Precision vs Threshold
        axes[1, 0].plot(pr_thresholds, precision[:-1])
        axes[1, 0].set_xlabel("Threshold")
        axes[1, 0].set_ylabel("Precision $\\frac{{TP}}{{TP + FP}}$", fontsize=14)
        title = self._format_title(base="Threshold vs Precision Score", kwargs=title_kwargs)
        axes[1, 0].set_title(title)
        axes[1, 0].grid(True)

        # Plot 4: Recall vs Threshold
        axes[1, 1].plot(pr_thresholds, recall[:-1])
        axes[1, 1].set_xlabel("Threshold")
        axes[1, 1].set_ylabel("Recall $\\frac{{TP}}{{TP + FN}}$", fontsize=14)
        title = self._format_title(base="Threshold vs Recall Score", kwargs=title_kwargs)
        axes[1, 1].set_title(title)
        axes[1, 1].grid(True)


        filename = "thres_vs_f1_acc_prec_recall.png"
        if title_kwargs:
            filename = f"thres_vs_f1_acc_prec_recall_r{title_kwargs['rounds']}_top{title_kwargs['top_n']}.png"
        savefig = os.path.join(save_path, filename)
        plt.savefig(savefig)

        plt.close(fig)
        return fpr, tpr, roc_thresholds
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare detectors."
    )
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
        "--dataset",
        type=str,
        choices=[CONFIG.PAN20, CONFIG.PAN23, CONFIG.PAN25, CONFIG.KOPPEL, CONFIG.BLOG],
        default=CONFIG.PAN20,
        help="Dataset to use for visualization (default: %(default)s)",
    )

    parser.add_argument(
        "--imposter_technique",
        type=str,
        choices=["llm", "text_len", "n_docs", "on-the-fly", "blogs", "fixed", "content"],
        default='fixed',
        help="Imposter technique to use (default: %(default)s)",
    )

    parser.add_argument(
        "--path2imp",
        type=str,
        default=CONFIG.PATH2PAN20,
        help="Path to the imposter dataset (default: %(default)s)",
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
   
    args = parser.parse_args()
    
    vis_det = VisDetectors(dataset=args.dataset, imposter_args={'rounds': args.rounds, 'top_n': args.top_n, 'path2imp': args.path2imp, 'imposter_technique':args.imposter_technique, 'upsample':args.upsample}, detectors=[CONFIG.IMPOSTER, CONFIG.UNMASKING])
    vis_det.visualize(balanced=args.balanced)