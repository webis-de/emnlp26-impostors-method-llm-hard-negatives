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

"""Plot paraphrase length percentages from MongoDB per dataset, grouped by model.

This runner reads naive and non-naive paraphrase collections, computes
``(length_paraphrased_text / length_original_text) * 100`` per row, and saves
one boxplot per dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from genai_detection.config import CONFIG
from genai_detection.experiments.paraphrases.paraphrase_plots import ParaphrasePlotter
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)


class MongoParaphraseLengthPercentagePlotter:
    """Load paraphrase rows from MongoDB and render length-percentage boxplots."""

    def __init__(
        self,
        save_base_path: Optional[Path] = None,
        local_ray: Optional[bool] = None,
    ):
        if local_ray is None:
            local_ray = os.path.exists("/Users/klara")

        self.mongodb = ParaphraseMongoDB(local_ray=local_ray)

        default_base = (
            Path(__file__).resolve().parents[3]
            / CONFIG.SAVE_PATH
            / "paraphrasing"
            / "paraphrase_evaluation"
            / "from_mongodb_length_percentage"
        )
        self.save_base_path = save_base_path or default_base
        self.save_base_path.mkdir(parents=True, exist_ok=True)

        self.plotter = ParaphrasePlotter(save_base_path=self.save_base_path)

    @staticmethod
    def _load_collection_rows(collection, paraphrase_approach: str) -> pd.DataFrame:
        rows = list(
            collection.find(
                {},
                {
                    "length_original_text": 1,
                    "length_paraphrased_text": 1,
                    "llm": 1,
                    "dataset_name": 1,
                },
            )
        )
        data = pd.DataFrame(rows)
        if data.empty:
            return data

        data["paraphrase_approach"] = paraphrase_approach
        data["model"] = data["llm"]
        data["length_original_text"] = pd.to_numeric(
            data["length_original_text"], errors="coerce"
        )
        data["length_paraphrased_text"] = pd.to_numeric(
            data["length_paraphrased_text"], errors="coerce"
        )

        # Manual computation from stored per-text word counts; original length is 100%.
        data["paraphrase_length_pct_words"] = np.where(
            data["length_original_text"] > 0,
            (data["length_paraphrased_text"] / data["length_original_text"]) * 100.0,
            np.nan,
        )
        return data

    def build_plot_dataframe(self) -> pd.DataFrame:
        non_naive = self._load_collection_rows(
            collection=self.mongodb.non_naive_paraphrase_collection,
            paraphrase_approach="non_naive",
        )
        naive = self._load_collection_rows(
            collection=self.mongodb.naive_paraphrase_collection,
            paraphrase_approach="naive",
        )

        data = pd.concat([non_naive, naive], axis=0, ignore_index=True)
        if data.empty:
            raise ValueError(
                "No documents found in naive or non-naive paraphrase collections."
            )

        required_cols = {"dataset_name", "model", "paraphrase_length_pct_words"}
        missing_cols = [c for c in required_cols if c not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns for plotting: {missing_cols}. "
                f"Available columns: {list(data.columns)}"
            )

        data = data.dropna(
            subset=["dataset_name", "model", "paraphrase_length_pct_words"]
        ).copy()
        logger.info("Prepared plotting dataframe with %d rows.", len(data))
        return data

    def run(
        self,
        datasets: Optional[list[str]] = None,
        display_plot: bool = False,
    ) -> pd.DataFrame:
        data = self.build_plot_dataframe()

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
                logger.warning(f"Skipping empty dataset={dataset_name!r}.")
                continue
            logger.info(
                "Plotting dataset=%s rows=%d grouped by model.",
                dataset_name,
                len(df_subset),
            )
            if len(df_subset["model"].unique()) <= 1:
                logger.info(
                    "Skipping dataset=%s because it has only one model.",
                    dataset_name,
                )
                continue
            logger.info("Found %d models for dataset %s.", len(df_subset["model"].unique()), dataset_name)
            self.plotter.plot_length_percentage_boxplot(
                df=df_subset,
                data_category=dataset_name,
                group_by="model",
                display_plot=display_plot,
                star_approach="non_naive",
                star_label="Two-step Approach",
            )

        return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paraphrase length percentages from MongoDB per dataset, "
            "grouped by model."
        )
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

    runner = MongoParaphraseLengthPercentagePlotter(local_ray=not args.remote_ray)
    runner.run(
        datasets=args.datasets,
        display_plot=args.display_plot,
    )


if __name__ == "__main__":
    main()
