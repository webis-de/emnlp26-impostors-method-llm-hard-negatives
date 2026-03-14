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

"""Plot cached paraphrase metrics from MongoDB without recomputing scores.

This runner reads rows from the ``paraphrase_scores`` collection, enriches them
with source metadata (model/dataset/prompt/approach) from paraphrase
collections, and saves one boxplot per metric for each dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from genai_detection.config import CONFIG
from genai_detection.experiments.paraphrases.paraphrase_data_loader import (
    ParaphraseDataLoader,
)
from genai_detection.experiments.paraphrases.paraphrase_metrics import (
    ParaphraseMetricCalculator,
)
from genai_detection.experiments.paraphrases.paraphrase_plots import ParaphrasePlotter
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)


class MongoParaphraseScorePlotter:
    """Load cached paraphrase scores from MongoDB and render boxplots."""

    def __init__(
        self,
        save_base_path: Optional[Path] = None,
        local_ray: Optional[bool] = None,
    ):
        if local_ray is None:
            local_ray = os.path.exists("/Users/klara")

        self.mongodb = ParaphraseMongoDB(local_ray=local_ray)
        self.data_loader = ParaphraseDataLoader(
            mongodb=self.mongodb,
            base_dirs={},
        )

        default_base = (
            Path(__file__).resolve().parents[3]
            / CONFIG.SAVE_PATH
            / "paraphrasing"
            / "paraphrase_evaluation"
            / "from_mongodb_scores"
        )
        self.save_base_path = save_base_path or default_base
        self.save_base_path.mkdir(parents=True, exist_ok=True)

        self.plotter = ParaphrasePlotter(save_base_path=self.save_base_path)

    def _load_score_rows(self) -> pd.DataFrame:
        rows = list(self.mongodb.paraphrase_score_collection.find({}))
        scores = pd.DataFrame(rows)
        if scores.empty:
            raise ValueError("No documents found in collection 'paraphrase_scores'.")
        logger.info("Loaded %d rows from paraphrase_scores.", len(scores))
        return scores

    def _load_source_rows(self) -> pd.DataFrame:
        source = self.data_loader.obtain_complete_paraphrase_df_from_mongodb()
        if source.empty:
            logger.warning("No source paraphrase rows available for enrichment.")
            return source

        source = source.copy()
        source = source.rename(columns={"_id": "paraphrase_id", "dataset": "dataset_name"})
        if "llm" in source.columns and "model" not in source.columns:
            source["model"] = source["llm"]

        keep_cols = [
            "paraphrase_id",
            "dataset_name",
            "model",
            "llm",
            "prompt",
            "temperature",
            "paraphrase_approach",
            "text",
            "paraphrase",
        ]
        keep_cols = [c for c in keep_cols if c in source.columns]
        source = source[keep_cols].drop_duplicates(subset="paraphrase_id", keep="first")
        return source

    @staticmethod
    def _to_str_id(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns:
            return df
        out = df.copy()
        out[col] = out[col].astype(str)
        return out

    def build_plot_dataframe(self) -> pd.DataFrame:
        scores = self._load_score_rows()
        source = self._load_source_rows()

        if "paraphrase_id" not in scores.columns:
            raise ValueError("'paraphrase_id' missing in paraphrase_scores rows.")

        scores = self._to_str_id(scores, "paraphrase_id")
        if not source.empty:
            source = self._to_str_id(source, "paraphrase_id")
            data = scores.merge(source, on="paraphrase_id", how="left", suffixes=("", "_src"))
        else:
            data = scores.copy()

        fields_to_fill = [
            "dataset_name",
            "model",
            "llm",
            "prompt",
            "temperature",
            "paraphrase_approach",
        ]
        for field in fields_to_fill:
            src_field = f"{field}_src"
            if src_field in data.columns:
                if field in data.columns:
                    data[field] = data[field].fillna(data[src_field])
                    data.drop(columns=[src_field], inplace=True)
                else:
                    data.rename(columns={src_field: field}, inplace=True)

        if "model" not in data.columns and "llm" in data.columns:
            data["model"] = data["llm"]
        elif "model" in data.columns and "llm" in data.columns:
            data["model"] = data["model"].fillna(data["llm"])

        if "dataset_name" not in data.columns:
            raise ValueError(
                "Could not resolve 'dataset_name'. Expected it in scores or source paraphrase rows."
            )
        if "model" not in data.columns:
            raise ValueError(
                "Could not resolve 'model'. Expected model/llm in scores or source paraphrase rows."
            )

        data = data.dropna(subset=["dataset_name", "model"]).copy()
        logger.info("Prepared plotting dataframe with %d rows.", len(data))
        return data

    @staticmethod
    def _resolve_metric_names(df: pd.DataFrame, metrics: Optional[list[str]]) -> list[str]:
        if metrics:
            resolved = [m for m in metrics if m in df.columns]
            if not resolved:
                raise ValueError(f"None of the requested metrics {metrics} are present in the dataframe: {df.columns}.")
            return resolved

        default_metrics = list(ParaphraseMetricCalculator._DEFAULT_METRICS)
        resolved = [m for m in default_metrics if m in df.columns]
        if resolved:
            return resolved

        reserved = {
            "_id",
            "paraphrase_id",
            "reference_id",
            "dataset_name",
            "model",
            "llm",
            "prompt",
            "temperature",
            "paraphrase_approach",
            "text",
            "paraphrase",
            "hashcode",
            "bertscore_hash",
        }
        numeric_metrics = [
            c
            for c in df.columns
            if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric_metrics:
            raise ValueError("No plottable numeric metrics found in score rows.")
        return numeric_metrics

    def run(
        self,
        metrics: Optional[list[str]] = None,
        datasets: Optional[list[str]] = None,
        display_plot: bool = False,
    ) -> pd.DataFrame:
        data = self.build_plot_dataframe()
        metric_names = self._resolve_metric_names(df=data, metrics=metrics)
        logger.info("Using %d metrics for plotting: %s", len(metric_names), metric_names)

        dataset_values = sorted(data["dataset_name"].dropna().astype(str).unique().tolist())
        if datasets:
            requested = set(datasets)
            dataset_values = [d for d in dataset_values if d in requested]
            if not dataset_values:
                raise ValueError(
                    f"None of the requested datasets were found. Requested={datasets}"
                )

        for dataset_name in dataset_values:
            df_subset = data[data["dataset_name"] == dataset_name].copy()
            if df_subset.empty:
                continue
            logger.info(
                "Plotting dataset=%s rows=%d grouped by model.",
                dataset_name,
                len(df_subset),
            )
            if len(df_subset['model'].unique()) <=1:
                logger.info(f"Skipping dataset={dataset_name} because it has only one model.")
                continue
            self.plotter.plot_metric_boxplots_per_metric(
                df=df_subset,
                metric_names=metric_names,
                data_category=dataset_name,
                group_by="model",
                display_plot=display_plot,
            )

        return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot existing paraphrase_scores from MongoDB per dataset and metric, "
            "grouped by model."
        )
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        default=["gohsen_delta"],
        help="Metric to plot. Repeat to pass multiple metrics.",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=None,
        help="Dataset to plot. Repeat to pass multiple datasets.",
    )
    parser.add_argument(
        "--display-plot",
        action="store_true",
        help="Display plots interactively in addition to saving.",
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
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    runner = MongoParaphraseScorePlotter(local_ray=not args.remote_ray)
    runner.run(
        metrics=args.metrics,
        datasets=args.datasets,
        display_plot=args.display_plot,
    )


if __name__ == "__main__":
    main()
