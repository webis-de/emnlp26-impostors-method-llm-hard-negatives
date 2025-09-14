import argparse
import json
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

# TODO: Bigger
NUM_SAMPLES = 5


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
        self.savefig_base = (
            Path(__file__).resolve().parent.parent.parent / CONFIG.SAVE_PATH
        )

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
                    Path(__file__).resolve().parent.parent.parent
                    / CONFIG.PATH2STUDENT_ESSAYS
                )[split].to_pandas()
            else:
                raise ValueError(
                    f"Dataset {self.dataset_name} is not supported for visualization."
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset {self.dataset_name} for split {split}: {e}"
            ) from e

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

    def _load_datasets(
        self,
        save_path: Path,
        min_samples: int = 20,
        dataset_name: str = "student",
    ):
        """
        Loads the train and test datasets from disk.
        Ensures the same items are used for all runs.
        """
        assert save_path.exists(), f"Save path {save_path} does not exist."
        datapath = save_path / "data"
        datapath.mkdir(parents=True, exist_ok=True)
        datafile_name = f"{dataset_name}_subset.json"

        print(f"Checking for existing dataset at {datapath / datafile_name}")
        if (datapath / datafile_name).exists() and (datapath / datafile_name).is_file():
            print(f"Loading dataset from {datafile_name}")
            dataset = pd.read_json(datapath / datafile_name)
            return dataset

        train_dataset = self.load_data(split="train")
        test_dataset = self.load_data(split="test")

        def downsample(df, target_col="same"):
            """Downsample each class in df to the size of the smallest class."""
            size_smaller_class = min(min_samples, df[target_col].value_counts().min())
            return (
                df.groupby(target_col, group_keys=False)
                .apply(lambda x: x.sample(size_smaller_class, random_state=42))
                .reset_index(drop=True)
            )

        # Apply to both train and test sets
        train_dataset = downsample(train_dataset, target_col="same")
        test_dataset = downsample(test_dataset, target_col="same")

        if train_dataset.empty or test_dataset.empty:
            raise ValueError(
                "Train or test dataset is empty. Cannot visualize impostors."
            )
        data = dict()
        for df, split in zip([train_dataset, test_dataset], ["train", "test"]):
            data[split] = []
            for row in df.itertuples():
                text_l, text_r = row.pair
                gt_label = (
                    bool(row.same) if isinstance(row.same, (np.generic,)) else row.same
                )
                gt_authors = (
                    row.authors.tolist()
                    if isinstance(row.authors, np.ndarray)
                    else row.authors
                )
                pair = (
                    tuple(row.pair.tolist())
                    if isinstance(row.pair, np.ndarray)
                    else row.pair
                )

                data[split].append(
                    {
                        "disputed_text": str(text_l),
                        "known_text": str(text_r),
                        "same": gt_label,
                        "authors": gt_authors,
                        "pair": pair,
                    }
                )

        # print("Data type:", type(data), data["train"].keys())
        with open(datapath / datafile_name, "w") as f:
            json.dump(data, f, indent=4)
        return data

    #################################################################################

    def _get_missing_scores_indices(
        self, dataset: pd.DataFrame, loaded_data: dict, imp_gen: str, n_imp: int = 50
    ):
        mask = []
        for row in dataset.itertuples():
            pair = str(row.pair)
            if pair not in loaded_data:
                loaded_data[pair] = {}
            if imp_gen not in loaded_data[pair]:
                loaded_data[pair][imp_gen] = {}
            if n_imp not in loaded_data[pair][imp_gen]:
                mask.append(row.Index)  # store index of row where score is missing
        return mask

    def _helper_missing_scores(
        self,
        dataset: pd.DataFrame,
        loaded_data: dict,
        n_imp: int = 50,
        imp_gen: str = "fixed",
    ):
        def fill_impostor_score(row):
            score_attr = f"{imp_gen}_score"
            if pd.isna(row[score_attr]):
                pair = str(row["pair"])
                if (
                    pair in loaded_data
                    and imp_gen in loaded_data[pair]
                    and str(n_imp) in loaded_data[pair][imp_gen]
                ):
                    return loaded_data[pair][imp_gen][str(n_imp)]
            return row[score_attr]

        dataset[f"{imp_gen}_score"] = dataset.apply(fill_impostor_score, axis=1)

        missing_scores_indices = self._get_missing_scores_indices(
            dataset, loaded_data, imp_gen, n_imp
        )

        return dataset.loc[missing_scores_indices], missing_scores_indices

    def _update_loaded_data(
        self,
        loaded_data: dict,
        missing_scores: list,
        n_imp: int,
        imp_gen: str = "fixed",
    ):
        for idx, pair in enumerate(missing_scores):
            if pair not in loaded_data:
                loaded_data[pair] = {}
            if imp_gen not in loaded_data[pair]:
                loaded_data[pair][imp_gen] = {}
            loaded_data[pair][imp_gen][str(n_imp)] = missing_scores[pair]
        return loaded_data

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
            real_time_generation=True,  # TODO: turn True, otherwise on-the-fly generation is not possible (currently too much data for too few free api calls)
        )
        impostor_detector.set_training_mode(training_mode)
        # TODO
        res = impostor_detector._get_score_impl(pair)
        return [r[0] for r in res] if isinstance(res, list) else res[0]

    def _run_fig_4_worker(self, imp_gen, test_dataset, path2imp, save_path):
        assert save_path.exists(), f"Save path {save_path} does not exist."
        print(f"Using {imp_gen} impostor generation with path to imposters: {path2imp}")
        save_path = save_path / "existing_generated_imps"
        save_path.mkdir(parents=True, exist_ok=True)
        existing_scores_filename = save_path / f"impostor_{imp_gen}_scores.json"

        # load existing scores if available
        if f"{imp_gen}_score" not in test_dataset.columns:
            test_dataset[f"{imp_gen}_score"] = np.nan

        if existing_scores_filename.exists():
            with open(existing_scores_filename, "r") as f:
                loaded_data = json.load(f)

        else:
            loaded_data = {}

        impostor_detector = ImpostorDetector(
            impostor_technique=imp_gen,
            n_impostors=50,
            rounds=100,  # cf. pg. 181, Koppel et al. (2014)
            top_n=100000,  # cf. pg. 179, Koppel et al. (2014)
            path2imp=path2imp,
            upsample=False,
            real_time_generation=True,  # TODO: turn True, otherwise on-the-fly generation is not possible (currently too much data for too few free api calls)
        )
        # work with test dataset
        impostor_detector.set_training_mode(
            False
        )  # set to False for validation: Use training set for impostor generation for fixed impostor technique

        rows_without_scores, missing_scores_indices = self._helper_missing_scores(
            test_dataset, loaded_data, 50, imp_gen
        )

        if len(rows_without_scores) > 0:
            print(
                f"Found {len(rows_without_scores)} rows without impostor scores in test data."
            )
            missing_scores = {}

            for idx, row in rows_without_scores.iterrows():
                try:
                    print("IMP Gen:", imp_gen)
                    score = self._helper_impostor(path2imp, row["pair"], imp_gen, False)
                    test_dataset.at[idx, f"{imp_gen}_score"] = score
                    missing_scores_indices.remove(idx)
                    # TODO: tuple as key
                    missing_scores[str(test_dataset.loc[idx, "pair"])] = score
                    print(
                        f"Computed impostor score for row {idx} using imp gen {imp_gen}."
                    )
                except Exception as e:
                    print(
                        f"Error computing impostor score for row {idx} with pair {row['pair'][0][:100]}..., {row['pair'][1][:100]}...: {e}"
                    )
                    traceback.print_exc()

            # print(
            #     "Missing scores computed for",
            #     len(missing_scores),
            #     "rows:",
            #     missing_scores,
            # )
            loaded_data = self._update_loaded_data(
                loaded_data=loaded_data,
                missing_scores=missing_scores,
                n_imp=50,
                imp_gen=imp_gen,
            )
            with open(existing_scores_filename, "w") as f:
                json.dump(loaded_data, f, indent=2)
            print(
                f"Saved missing test scores to {existing_scores_filename} for {len(rows_without_scores)} rows."
            )

        # each entry in scores is a one-element list
        test_dataset[f"{imp_gen}_score"] = np.array(
            test_dataset[f"{imp_gen}_score"].tolist()
        ).ravel()

        # same author pairs
        same_author_precisions, same_author_recalls, same_pr_thresholds = (
            precision_recall_curve(
                test_dataset["same"],
                test_dataset[f"{imp_gen}_score"],
                pos_label=1,
            )
        )
        return {
            "imp_gen": imp_gen,
            "same_author_precisions": same_author_precisions,
            "same_author_recalls": same_author_recalls,
            "same_pr_thresholds": same_pr_thresholds,
        }

    # ugly, but only for reproduction of Figure 4 a, b from Koppel et al. (2014)
    def reproduce_fig4_prec_recall_dif_imp_appr(
        self,
        args: dict,
        save_path: Optional[Path],
        imp_gen_options: list = ["fixed"],
    ) -> None:
        """
        Visualizes the impostor detection results via Precision-Recall curves for different impostor generation techniques (cf. Figures 4 a, b from Koppel et al. (2014)).
        """
        print(
            "Reproducing Figure 4 with different impostor generation techniques:",
            imp_gen_options,
        )
        # assert isinstance(args, dict), "args must be a dictionary"
        if not save_path:
            save_path = (
                self.savefig_base
                / "impostor_scores"
                / self.dataset_name
                / "koppel_fig4_gen_imps"
            )

        save_path.mkdir(parents=True, exist_ok=True)
        pr_save_path = save_path / "precision_recall_values"
        pr_save_path.mkdir(parents=True, exist_ok=True)

        data = self._load_datasets(
            save_path=save_path,
            min_samples=NUM_SAMPLES,
            dataset_name=self.dataset_name.lower(),
        )
        # we do not need training set for unsupervised baselines, or impostor methods bc precision-recall across all thresholds
        test_dataset = pd.json_normalize(data["test"])
        print("Test dataset shape:", test_dataset.shape)

        # could be initially different, bc args are from argparse which are irrespective from calling thsi function with defined dataset_name

        baselines = [
            "unsupervised baseline min-max",
            "unsupervised baseline cosine",
            "supervised baseline",
        ]
        same_author_precisions, same_author_recalls = {}, {}

        print(
            "Start sequential computation (else OOM) for different impostor generation techniques."
        )
        for imp_gen in imp_gen_options:
            if imp_gen not in ["on-the-fly", "naive_llm", "llm", "non_naive_llm"]:
                path2imp = (
                    Path(os.getcwd()).resolve() / CONFIG.PATH2BLOG
                    if self.dataset_name == CONFIG.BLOG
                    else Path(os.getcwd()).resolve() / CONFIG.PATH2STUDENT_ESSAYS
                )
            else:
                path2imp = (
                    self.savefig_base
                    # / "impostor_scores"
                    # / self.dataset_name
                    # / "modularized"
                    / "dumps"
                    / f"impostor_{imp_gen}.json"
                    # / f"modularized_impostors_{imp_gen}_old.json"  # FIXME TODO
                )

            result = self._run_fig_4_worker(imp_gen, test_dataset, path2imp, save_path)
            print("\n\nResult for imp gen:", imp_gen, result)
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
            preds = baseline.get_score(test_dataset["pair"])

            test_dataset[f"{baseline_name.replace(' ','_')}_score"] = np.array(
                preds.tolist()
            ).ravel()

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

        # Precision-Recall Curve: 	Imbalanced
        # scores: non-thresholded measure of decisions, relative ranking of predictions
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html (05.06.2025)
        print(
            "DEBUG Plotting Precision-Recall Curves for different impostor generation techniques:",
            same_author_precisions.keys(),
        )
        label_translations = {
            "fixed": "Fixed",
            "on-the-fly": "On-the-Fly",
            "naive_llm": "Naive LLM",
            "non_naive_llm": "Non-Naive LLM",
            "unsupervised_baseline_min-max": "Unsup. Min-Max (B)",
            "unsupervised_baseline_cosine": "Unsup. Cosine (B)",
            "supervised_baseline": "Sup. SVC (B)",
        }

        line_styles = dict(zip(baselines, [":", "--", "-."]))
        for kind, data in zip(
            ["Same Author"],
            [
                (same_author_precisions, same_author_recalls),
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
                    label=label_translations[imp_gen],
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
            title = f"Precision-Recall Curve for {kind} Data"
            plt.title(title)
            plt.legend()
            plt.tight_layout()

            for format in ["svg"]:  # "png",
                args_dict = vars(args)  # convert Namespace -> dict
                figure_name = (
                    f"roc_prec_recall_curve_dif_{kind.replace(' ', '_')}_imp_gen.{format}"
                    if len(args_dict) == 0
                    else f"roc_prec_recall_curve_r{args_dict.get('rounds', 'NA')}_top{args_dict.get('top_n', 'NA')}_{kind.replace(' ', '_')}_dif_imp_gen.{format}"
                )
                plt.savefig(save_path / figure_name)
                print(f"Saved figure {figure_name} to {save_path}.")
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

    def generate_impostors(self, imp_gen: str = "naive_llm") -> dict:
        """
        Paraphrases texts (i.e. generates impostors) in the given dataset using a paraphrasers.
        Generated paraphrases are also saved to disk for future use.

        :param dataset: The dataset containing texts to paraphrase.
        :param text_column: The column name in the dataset that contains the texts.
        :return: A list of paraphrased texts.
        """
        save_path = (
            self.savefig_base / "impostor_scores" / self.dataset_name / "modularized"
        )

        save_path.mkdir(parents=True, exist_ok=True)
        paraphrases_save_path = save_path / "paraphrases"
        paraphrases_save_path.mkdir(parents=True, exist_ok=True)
        pr_save_path = save_path / "precision_recall_values"
        pr_save_path.mkdir(parents=True, exist_ok=True)
        data = self._load_datasets(
            save_path=paraphrases_save_path,
            min_samples=NUM_SAMPLES,
            dataset_name=self.dataset_name.lower(),
        )
        if "test" in data:
            data = data["test"]
        # only use test dataset
        test_dataset = pd.json_normalize(data)
        existing_paraphrases_filename = save_path / "dumps"
        existing_paraphrases_filename.mkdir(parents=True, exist_ok=True)
        existing_paraphrases_filename = (
            existing_paraphrases_filename / f"modularized_impostors_{imp_gen}.json"
        )

        # two iterations, generating impostors for each candidate once
        if existing_paraphrases_filename.exists():
            with open(existing_paraphrases_filename, "r") as f:
                loaded_data = json.load(f)
        else:
            loaded_data = {}

        impostor_detector = ImpostorDetector(
            impostor_technique=imp_gen,
            n_impostors=50,
            rounds=100,  # cf. pg. 181, Koppel et al. (2014)
            top_n=100000,  # cf. pg. 179, Koppel et al. (2014)
            path2imp="path2imp",
            upsample=False,
            real_time_generation=True,
        )

        for item in test_dataset.itertuples():
            disputed_text = item.disputed_text
            known_text = item.known_text
            for text in [disputed_text, known_text]:
                # generate 50 paraphrases for each text if not already done
                impostor_candidates = (
                    impostor_detector.impostor_generator.generate_impostors(
                        text=text,
                        real_time_generation=True,
                        path2imp=existing_paraphrases_filename,
                    )
                )
                if known_text not in loaded_data:
                    loaded_data[known_text] = {}
                # add new keys if not already present
                new_keys = impostor_candidates.keys() - loaded_data[known_text].keys()
                for k in new_keys:
                    paraphrase = impostor_candidates[k]
                    if (len(paraphrase.split()) < 700) or (
                        (len(paraphrase.split()) / len(known_text.split())) < 0.7
                    ):
                        continue
                    loaded_data[known_text][k] = paraphrase
                with open(existing_paraphrases_filename, "w") as f:
                    json.dump(loaded_data, f, indent=2)

        return loaded_data


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
        impostor_technique="naive_llm",
        n_impostors=args.n_impostors,
        rounds=args.rounds,
        top_n=args.top_n,
        path2imp=args.path2imp,
        upsample=args.upsample,
    )

    # reproduction of Figure 2/ 4 from Koppel et al. (2014)
    our_figure_impostor_options = [
        # "on-the-fly",
        "naive_llm",
        "fixed",
        # "non_naive_llm",
    ]
    fig = 4
    print(
        f"Reproducing Figure {fig} from Koppel et al. (2014) on BLOG and STUDENT data."
    )

    vis_det = VisDetectors(
        dataset_name=CONFIG.STUDENT_ESSAYS,
        detectors=[impostor],
    )
    print(
        f"impostor Detector initialized for fig {fig} and dataset {CONFIG.STUDENT_ESSAYS}."
    )
    # {reference: {paraphraser_prompt: paraphrase, ...}, ...}
    # TODO: uncomment
    # impostors_dict = vis_det.generate_impostors(imp_gen="naive_llm")

    # run impostor approach with pre-generated impostors (loaded automatically from disk in impostor generator)
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

    # vis_det = VisDetectors(
    #     dataset_name=CONFIG.BLOG,
    #     detectors=[impostor],
    # )
    # print(f"impostor Detector initialized for fig {fig} and dataset {CONFIG.BLOG}.")

    # # figure 4 with our contributions (LLM based impostors)
    # vis_det.reproduce_fig4_prec_recall_dif_imp_appr(
    #     args=args,
    #     imp_gen_options=our_figure_impostor_options,
    #     save_path=vis_det.savefig_base
    #     / "impostor_scores"
    #     / vis_det.dataset_name
    #     / "our_contributions_scores",
    # )
    # print(f"Finished reproducing Figure {fig} from Koppel et al. (2014) on BLOG data.")
