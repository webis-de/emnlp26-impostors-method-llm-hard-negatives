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

"""Run end-to-end paraphrase evaluation and plotting experiments.

This module is the CLI/runner entrypoint for the paraphrase experiment pipeline:
1) load scored paraphrases (naive + non_naive),
2) compute missing metrics,
3) render plots globally and per dataset.

Technique label modes available for boxplots:
- ``model``: group only by the model/LLM name.
- ``prompt``: group only by prompt signature (prompt text summary).
- ``model_prompt``: combine model and prompt signature.
- ``model_prompt_approach``: combine approach (naive/non_naive), model, and prompt
  signature; this is the most descriptive mode for paper figures.

These modes only control labels when the plotter groups by the synthetic
``paraphrase_technique`` field. They are different from ``group_by``, which
chooses the actual grouping column.
"""

import argparse
import logging
from typing import Literal

from genai_detection.experiments.paraphrases.paraphraser_evaluation import (
    ParaphrasingEvaluator,
)

logger = logging.getLogger(__name__)
TechniqueLabelMode = Literal[
    "model",
    "prompt",
    "model_prompt",
    "model_prompt_approach",
]
TECHNIQUE_LABEL_MODES: tuple[str, ...] = (
    "model",
    "prompt",
    "model_prompt",
    "model_prompt_approach",
)


def run_paraphrase_experiments(
    display_plot: bool = False,
    technique_label_mode: TechniqueLabelMode = "model_prompt_approach",
    prompt_words: int = 6,
):
    """Run scoring + visualization for paraphrase experiments.

    Args:
        display_plot: If ``True``, show figures interactively in addition to saving.
        technique_label_mode: Label strategy used when plots group by the
            synthetic ``paraphrase_technique`` field. This does not choose the
            grouping column; the plotter's ``group_by`` parameter does that.
            Supported values:
            - ``model``: Only model/LLM name.
            - ``prompt``: Only prompt signature.
            - ``model_prompt``: Model plus prompt signature.
            - ``model_prompt_approach``: Approach (naive/non_naive) plus model and
              prompt signature.
        prompt_words: Max number of prompt words kept in prompt signatures when
            ``technique_label_mode`` includes prompt information.

    Returns:
        Tuple of (full evaluation dataframe, dataframe with min/max examples per
        metric).
    """
    evaluator = ParaphrasingEvaluator(metric_config={"include_meteor": False, "include_wmd": False,
                                                     "include_sbert_cos": False})
    logger.info("Starting evaluation of paraphrasers...")
    # Compute/load metric rows.
    df, extremest_paraphrases = evaluator.evaluate()
    logger.info("Finished computing scores. Columns: %s", list(df.columns))

    metrics_names = evaluator.get_metric_names()
    # Overall plots across all datasets.
    evaluator.plot_length_percentage_boxplot(
        df=df,
        technique_label_mode=technique_label_mode,
        prompt_words=prompt_words,
        display_plot=display_plot,
    )
    evaluator.plot_metric_boxplots_per_metric(
        df=df,
        technique_label_mode=technique_label_mode,
        prompt_words=prompt_words,
        display_plot=display_plot,
    )

    # Dataset-specific plots.
    for dataset_name in df["dataset_name"].unique():
        df_subset = df[df["dataset_name"] == dataset_name].copy()
        # evaluator.plot_metric_scatter(
        #     df=df_subset, data_category=dataset_name, display_plot=display_plot
        # )
        # evaluator.plot_models_metrics(
        #     df=df_subset, data_category=dataset_name, display_plot=display_plot
        # )
        evaluator.plot_length_percentage_boxplot(
            df=df_subset,
            data_category=dataset_name,
            technique_label_mode=technique_label_mode,
            prompt_words=prompt_words,
            display_plot=display_plot,
        )
        evaluator.plot_metric_boxplots_per_metric(
            df=df_subset,
            data_category=dataset_name,
            technique_label_mode=technique_label_mode,
            prompt_words=prompt_words,
            display_plot=display_plot,
        )

    logger.info("Evaluation complete.")
    return df, extremest_paraphrases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run paraphrase evaluation experiments.")
    parser.add_argument(
        "--label-mode",
        default="model",
        choices=TECHNIQUE_LABEL_MODES,
        help=(
            "How to build paraphrase-technique labels for boxplots: "
            "model | prompt | model_prompt | model_prompt_approach."
        ),
    )
    parser.add_argument(
        "--prompt-words",
        default=6,
        type=int,
        help="Maximum number of prompt words shown in technique labels.",
    )
    parser.add_argument(
        "--display-plot",
        action="store_true",
        help="Display plots interactively instead of only saving them.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_paraphrase_experiments(
        display_plot=args.display_plot,
        technique_label_mode=args.label_mode,
        prompt_words=args.prompt_words,
    )
