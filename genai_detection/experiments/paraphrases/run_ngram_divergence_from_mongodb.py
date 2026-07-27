# Copyright 2026 Klara M. Gutekunst, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run 4-gram distribution divergence experiments for non-naive paraphrases."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from genai_detection.config import CONFIG
from genai_detection.experiments.paraphrases.ngram_distribution_metrics import (
    NGramDistributionDivergenceCalculator,
)
from genai_detection.experiments.paraphrases.paraphrase_data_loader import (
    ParaphraseDataLoader,
)
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)


class MongoNGramDivergenceExperiment:
    """Load paraphrases from MongoDB, compute divergences, and plot aggregates."""

    DEFAULT_FEATURES = ("assignment", "ethnicity", "teacher", "political_orientation")
    METRICS = ("kl_divergence", "js_divergence")

    def __init__(
        self,
        save_base_path: Optional[Path] = None,
        local_ray: Optional[bool] = None,
        ngram_size: int = 4,
        max_features: int = 100_000,
        smoothing: float = 1e-12,
    ):
        if local_ray is None:
            local_ray = os.path.exists("/Users/klara")

        self.mongodb = ParaphraseMongoDB(local_ray=local_ray)
        self.data_loader = ParaphraseDataLoader(mongodb=self.mongodb, base_dirs={})
        self.calculator = NGramDistributionDivergenceCalculator(
            n=ngram_size,
            max_features=max_features,
            smoothing=smoothing,
        )

        default_base = (
            Path(__file__).resolve().parents[3]
            / CONFIG.SAVE_PATH
            / "paraphrasing"
            / "ngram_distribution_divergence"
        )
        self.save_base_path = save_base_path or default_base
        self.save_base_path.mkdir(parents=True, exist_ok=True)

        sns.set_theme(context="paper", style="whitegrid", palette="colorblind")
        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "axes.titlesize": 17,
                "axes.labelsize": 15,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
            }
        )

    def load_non_naive_paraphrases(self) -> pd.DataFrame:
        """Load non-naive paraphrases and their original texts/metadata."""
        projection = {
            "_id": True,
            "text": True,
            "dataset_name": True,
            "assignment": True,
            "ethnicity": True,
            "teacher": True,
            "political_orientation": True,
            "gender": True,
            "age": True,
            "sign": True,
        }
        df = self.data_loader.obtain_complete_paraphrase_df_from_mongodb(
            original_projection=projection
        )
        if df.empty:
            raise ValueError("No paraphrases found in MongoDB.")

        df = df[df["paraphrase_approach"] == "non_naive"].copy()
        if df.empty:
            raise ValueError("No rows found in non_naive_paraphrases.")

        df.rename(columns={"text": "original_text"}, inplace=True)
        if "dataset_name" not in df.columns and "dataset" in df.columns:
            df.rename(columns={"dataset": "dataset_name"}, inplace=True)
        if "paraphrase" not in df.columns:
            raise ValueError("Expected a 'paraphrase' column in non_naive_paraphrases.")

        logger.info("Loaded %d non-naive paraphrase rows with columns: %s.", len(df), df.columns)
        df = df.dropna(subset=["text_id", "dataset_name", "original_text", "paraphrase"]).copy()
        logger.info("Loaded %d non-naive paraphrase rows with original texts.", len(df))
        return df

    def compute_pairwise_divergences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute KL and JS divergence for every original/paraphrase pair."""
        rows = []
        for index, row in df.iterrows():
            try:
                result = self.calculator.compare(
                    original_text=row["original_text"],
                    paraphrase_text=row["paraphrase"],
                )
            except Exception as e:
                logger.error("Failed divergence computation for row index=%s: %s", index, e)
                continue

            out = row.to_dict()
            out.update(result.__dict__)
            rows.append(out)

        result_df = pd.DataFrame(rows)
        if result_df.empty:
            raise ValueError("No divergence rows could be computed.")
        logger.info("Computed divergences for %d original/paraphrase pairs.", len(result_df))
        return result_df

    def aggregate_per_text(self, pairwise_df: pd.DataFrame) -> pd.DataFrame:
        """Average all paraphrase comparisons belonging to the same original text."""
        group_cols = ["dataset_name", "text_id"]
        metadata_cols = [
            feature
            for feature in self.DEFAULT_FEATURES
            if feature in pairwise_df.columns
        ]
        aggregations = {
            "kl_divergence": "mean",
            "js_divergence": "mean",
            "original_ngram_count": "mean",
            "paraphrase_ngram_count": "mean",
            "vocabulary_size": "mean",
            "paraphrase": "count",
        }
        for col in metadata_cols:
            aggregations[col] = "first"

        aggregate_df = (
            pairwise_df.groupby(group_cols, dropna=False)
            .agg(aggregations)
            .rename(columns={"paraphrase": "n_paraphrases"})
            .reset_index()
        )
        logger.info("Aggregated divergences to %d original texts.", len(aggregate_df))
        return aggregate_df

    @staticmethod
    def _safe_name(value: str) -> str:
        return str(value).replace(" ", "_").replace("/", "_").replace(":", "_")

    @staticmethod
    def _readable_dataset(dataset_name: str) -> str:
        return CONFIG.DATASET_TRANSLATIONS.get(dataset_name, dataset_name)

    @staticmethod
    def _readable_metric(metric: str) -> str:
        return {
            "kl_divergence": "KL Divergence",
            "js_divergence": "Jensen-Shannon Divergence",
        }.get(metric, metric)

    def _plot_metric_boxplot(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        metric: str,
        group_by: str,
    ) -> None:
        plot_df = data[[group_by, metric]].dropna().copy()
        if plot_df.empty:
            logger.warning(
                "Skipping plot for dataset=%s metric=%s group_by=%s because no values remain.",
                dataset_name,
                metric,
                group_by,
            )
            return

        counts = plot_df.groupby(group_by, sort=True)[metric].count()
        order = counts.index.tolist()
        width = max(8, len(order) * 1.35)
        fig, ax = plt.subplots(figsize=(width, 5.8), constrained_layout=True)
        sns.boxplot(
            data=plot_df,
            x=group_by,
            y=metric,
            hue=group_by,
            order=order,
            dodge=False,
            showfliers=False,
            width=0.45,
            linewidth=1.1,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()

        metric_label = self._readable_metric(metric)
        ax.set_xlabel(group_by.replace("_", " ").title())
        ax.set_ylabel(metric_label)
        max_features = rf"${self.calculator.max_features:,}".replace(",", r"\,") + "$"

        ax.set_title(f"{metric_label} of Top-{max_features} "
                     f"{self.calculator.n}-Gram Distributions\n"
                     f"{self._readable_dataset(dataset_name)} Dataset")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        sns.despine(ax=ax)

        out_dir = self.save_base_path / self._safe_name(dataset_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        for file_format in ("svg", "pdf"):
            out = (
                out_dir
                / f"{self._safe_name(dataset_name)}_{metric}_boxplot_grouped_by_{self._safe_name(group_by)}.{file_format}"
            )
            fig.savefig(out, bbox_inches="tight", transparent=True, format=file_format)
            logger.info("Plot saved to %s", out)
        plt.close(fig)

    def plot(self, aggregate_df: pd.DataFrame, features: tuple[str, ...]) -> None:
        """Create per-dataset boxplots for each divergence metric and feature."""
        for dataset_name in sorted(aggregate_df["dataset_name"].dropna().astype(str).unique()):
            dataset_df = aggregate_df[aggregate_df["dataset_name"].astype(str) == dataset_name].copy()
            if dataset_df.empty:
                continue

            available_features = [
                feature
                for feature in features
                if feature in dataset_df.columns and dataset_df[feature].notna().any()
            ]
            if not available_features:
                logger.warning(
                    "No requested feature columns found for dataset=%s; plotting all texts together.",
                    dataset_name,
                )
                dataset_df["all_texts"] = "all"
                available_features = ["all_texts"]

            for feature in available_features:
                if dataset_df[feature].nunique(dropna=True) <= 1 and feature != "all_texts":
                    logger.info(
                        "Feature '%s' has one value for dataset=%s; plot is still generated.",
                        feature,
                        dataset_name,
                    )
                for metric in self.METRICS:
                    self._plot_metric_boxplot(
                        data=dataset_df,
                        dataset_name=dataset_name,
                        metric=metric,
                        group_by=feature,
                    )

    def run(
        self,
        datasets: Optional[list[str]] = None,
        features: tuple[str, ...] = DEFAULT_FEATURES,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        source_df = self.load_non_naive_paraphrases()
        if datasets:
            source_df = source_df[source_df["dataset_name"].isin(datasets)].copy()
            if source_df.empty:
                raise ValueError(f"None of the requested datasets were found: {datasets}")
            logger.info("Filtered source rows to requested datasets: %s", datasets)

        pairwise_df = self.compute_pairwise_divergences(source_df)
        aggregate_df = self.aggregate_per_text(pairwise_df)

        pairwise_path = self.save_base_path / "pairwise_ngram_divergences.csv"
        aggregate_path = self.save_base_path / "per_text_mean_ngram_divergences.csv"
        pairwise_df.to_csv(pairwise_path, index=False)
        aggregate_df.to_csv(aggregate_path, index=False)
        logger.info("Saved pairwise results to %s", pairwise_path)
        logger.info("Saved per-text aggregate results to %s", aggregate_path)

        self.plot(aggregate_df=aggregate_df, features=features)
        return pairwise_df, aggregate_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute KL and Jensen-Shannon divergence between original and "
            "non-naive LLM paraphrase 4-gram distributions from MongoDB."
        )
    )
    parser.add_argument("--dataset_name", dest="datasets", action="append", default=None)
    parser.add_argument(
        "--feature",
        dest="features",
        action="append",
        default=None,
        help=(
            "Original-text metadata feature used for boxplot grouping. "
            "Repeat to pass multiple features."
        ),
    )
    parser.add_argument("--ngram-size", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--smoothing", type=float, default=1e-12)
    parser.add_argument(
        "--save-base-path",
        type=Path,
        default=None,
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument(
        "--remote-ray",
        action="store_true",
        help="Use remote Ray Mongo connection instead of local forwarded Mongo.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    experiment = MongoNGramDivergenceExperiment(
        save_base_path=args.save_base_path,
        local_ray=not args.remote_ray,
        ngram_size=args.ngram_size,
        max_features=args.max_features,
        smoothing=args.smoothing,
    )
    experiment.run(
        datasets=args.datasets,
        features=tuple(args.features) if args.features else MongoNGramDivergenceExperiment.DEFAULT_FEATURES,
    )


if __name__ == "__main__":
    main()
