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

from __future__ import annotations

"""Calibrate alpha for directional hypothesis-test Type I error.

The script reads corrected directional p-values from MongoDB impostor outputs,
keeps only pairs marked as different-author pairs in ``all_pairs``, and plots
the empirical Type I error rate for alpha values from 0.001 to 0.1.
For on-the-fly impostors, ``retrieval_index`` is mapped back to the configured
method key so ``LABEL_TRANSLATIONS`` and ``LABEL_COLORS`` can be reused.

Usage:
    poetry run python scripts/calibrate_alpha_directional_hypothesis_test.py
"""

import argparse
import logging
import math
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(tempfile.gettempdir()) / "genai_detection_matplotlib"


def _ensure_writable_cache_env(env_var: str, fallback_name: str) -> None:
    current = os.environ.get(env_var)
    if current:
        current_path = Path(current)
        try:
            current_path.mkdir(parents=True, exist_ok=True)
            test_file = current_path / ".write_test"
            test_file.touch()
            test_file.unlink(missing_ok=True)
            return
        except OSError:
            pass

    fallback_path = _CACHE_DIR / fallback_name
    fallback_path.mkdir(parents=True, exist_ok=True)
    os.environ[env_var] = str(fallback_path)


_ensure_writable_cache_env("MPLCONFIGDIR", "matplotlib")
_ensure_writable_cache_env("XDG_CACHE_HOME", "xdg_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from pymongo.collection import Collection

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.experiments.reproduction.pan_metrics.run_pan_metrics import (
    N_IMPOSTORS,
)

logger = logging.getLogger(__name__)

LEFT_DISPUTED_RIGHT_P_VALUE_FIELD = (
    "left_disputed_right_candidate_corrected_p_value"
)
RIGHT_DISPUTED_LEFT_P_VALUE_FIELD = (
    "right_disputed_left_candidate_corrected_p_value"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / CONFIG.SAVE_PATH
    / "alpha_calibration"
)
RETRIEVAL_INDEX_TO_METHOD = {
    retrieval_index: method_key
    for method_key, retrieval_index in CONFIG.RETRIEVAL_INDEX_TRANSLATIONS.items()
}
PLOT_EXCLUDED_METHODS = {"on_the_fly_serpapi"}
PLOT_TITLE_FONT_SIZE = 15
PLOT_LABEL_FONT_SIZE = 13
PLOT_TICK_FONT_SIZE = 11
PLOT_LEGEND_FONT_SIZE = 11
PLOT_LEGEND_TITLE_FONT_SIZE = 12
DEFAULT_ALPHA_CRITICAL_N_TRIALS = (100, 50, 10)
DEFAULT_ALPHA_BONFERRONI_FACTOR = 2.0


def _safe_filename(value: str) -> str:
    safe_chars = [c if c.isalnum() or c in {"-", "_"} else "_" for c in value]
    return "_".join("".join(safe_chars).split("_")).strip("_") or "dataset"


def _display_dataset(dataset_name: str) -> str:
    return CONFIG.DATASET_TRANSLATIONS.get(
        dataset_name, dataset_name.replace("_", " ").title()
    )


def _display_method(method_name: str) -> str:
    if "__" in method_name:
        base_method, ablation = method_name.split("__", maxsplit=1)
        base_display = _display_method(base_method)
        ablation_display = ablation.removesuffix("ImpostorDetector").replace("_", " ")
        return f"{base_display} ({ablation_display})"
    return CONFIG.LABEL_TRANSLATIONS.get(
        method_name, method_name.replace("_", " ").title()
    )


def _method_order(methods: Iterable[str]) -> list[str]:
    method_set = set(methods)
    configured = [method for method in CONFIG.LABEL_TRANSLATIONS if method in method_set]
    remaining = sorted(method_set - set(configured))
    return configured + remaining


def _method_column(df: pd.DataFrame) -> str:
    return "method_key" if "method_key" in df.columns else "impostor_generation_technique"


def _plot_method_base(method_name: str) -> str:
    return method_name.split("__", maxsplit=1)[0]


def _technique_query_values(techniques: list[str] | None) -> list[str] | None:
    if not techniques:
        return None
    query_values = set(techniques)
    if any(technique in CONFIG.RETRIEVAL_INDEX_TRANSLATIONS for technique in techniques):
        query_values.add("on_the_fly")
    return sorted(query_values)


def _impostor_method_key(doc: dict[str, Any]) -> str:
    technique = doc["impostor_generation_technique"]
    if technique == "on_the_fly":
        retrieval_index = doc.get("retrieval_index")
        return RETRIEVAL_INDEX_TO_METHOD.get(retrieval_index, technique)
    return technique


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _alpha_grid(alpha_min: float, alpha_max: float, alpha_step: float) -> np.ndarray:
    if alpha_min <= 0:
        raise ValueError("--alpha-min must be greater than 0.")
    if alpha_max <= alpha_min:
        raise ValueError("--alpha-max must be greater than --alpha-min.")
    if alpha_step <= 0:
        raise ValueError("--alpha-step must be greater than 0.")

    n_steps = int(round((alpha_max - alpha_min) / alpha_step))
    alphas = alpha_min + np.arange(n_steps + 1) * alpha_step
    return alphas[alphas <= alpha_max + 1e-12]


def _critical_alpha_grid(
    *,
    n_trials_values: Iterable[int],
    n_impostors: int,
    bonferroni_factor: float = DEFAULT_ALPHA_BONFERRONI_FACTOR,
) -> np.ndarray:
    """Return alpha values where the binomial-test critical success count changes."""
    if n_impostors < 1:
        raise ValueError("--n-impostors must be at least 1.")
    if bonferroni_factor <= 0:
        raise ValueError("--alpha-bonferroni-factor must be greater than 0.")

    p0 = 1.0 / (1 + n_impostors)
    alphas: set[float] = set()
    for n_trials in n_trials_values:
        if n_trials < 1:
            raise ValueError("--alpha-critical-n-trials values must be positive.")
        for critical_k in range(1, n_trials + 1):
            tail_probability = sum(
                math.comb(n_trials, k) * (p0**k) * ((1 - p0) ** (n_trials - k))
                for k in range(critical_k, n_trials + 1)
            )
            alpha = min(1.0, bonferroni_factor * tail_probability)
            if 0.0 < alpha <= 1.0:
                alphas.add(float(alpha))
    if not alphas:
        raise ValueError(
            "No critical-k alpha values fall inside the requested alpha range."
        )
    return np.asarray(sorted(alphas), dtype=float)


def _load_author_pairs(
    collection: Collection,
    *,
    same_value: bool,
    dataset_names: list[str] | None,
    batch_size: int,
) -> set[tuple[Any, Any, str]]:
    query: dict[str, Any] = {"same": {"$eq": same_value, "$type": "bool"}}
    if dataset_names:
        query["dataset_name"] = {"$in": dataset_names}

    cursor = collection.find(
        query,
        {"_id": 0, "left_id": 1, "right_id": 1, "dataset_name": 1, "same": 1},
        batch_size=batch_size,
    )
    pairs = {
        (doc["left_id"], doc["right_id"], doc["dataset_name"])
        for doc in cursor
        if doc.get("same") is same_value
        and "left_id" in doc
        and "right_id" in doc
        and "dataset_name" in doc
    }
    pair_label = "same-author" if same_value else "different-author"
    logger.info("Loaded %d %s pairs from %s.", len(pairs), pair_label, collection.name)
    return pairs


def _load_impostor_outputs_for_pairs(
    collection: Collection,
    *,
    pair_keys: set[tuple[Any, Any, str]],
    pair_label: str,
    dataset_names: list[str] | None,
    techniques: list[str] | None,
    ablations: list[str] | None,
    n_impostors: int | None,
    left_p_value_field: str,
    right_p_value_field: str,
    batch_size: int,
) -> pd.DataFrame:
    query: dict[str, Any] = {
        left_p_value_field: {"$exists": True, "$ne": None},
        right_p_value_field: {"$exists": True, "$ne": None},
        "left_id": {"$exists": True, "$ne": None},
        "right_id": {"$exists": True, "$ne": None},
        "dataset_name": {"$exists": True, "$ne": None},
        "impostor_generation_technique": {"$exists": True, "$ne": None},
    }
    if dataset_names:
        query["dataset_name"] = {"$in": dataset_names}
    technique_query_values = _technique_query_values(techniques)
    if technique_query_values:
        query["impostor_generation_technique"] = {"$in": technique_query_values}
    if ablations:
        query["ablation"] = {"$in": ablations}
    if n_impostors is not None:
        query["n_impostors"] = n_impostors

    projection = {
        "_id": 0,
        "left_id": 1,
        "right_id": 1,
        "dataset_name": 1,
        "impostor_generation_technique": 1,
        "retrieval_index": 1,
        "ablation": 1,
        "n_impostors": 1,
        left_p_value_field: 1,
        right_p_value_field: 1,
    }

    rows: list[dict[str, Any]] = []
    skipped_invalid_p_values = 0
    skipped_missing_pair = 0
    cursor = collection.find(query, projection, batch_size=batch_size).sort("_id", 1)
    for doc in cursor:
        pair_key = (doc["left_id"], doc["right_id"], doc["dataset_name"])
        if pair_key not in pair_keys:
            skipped_missing_pair += 1
            continue

        method_key = _impostor_method_key(doc)
        ablation = doc.get("ablation")
        if ablation:
            method_key = f"{method_key}__{ablation}"
        if techniques and (
            doc["impostor_generation_technique"] not in techniques
            and _plot_method_base(method_key) not in techniques
        ):
            continue

        left_p_value = _float_or_none(doc.get(left_p_value_field))
        right_p_value = _float_or_none(doc.get(right_p_value_field))
        if left_p_value is None or right_p_value is None:
            skipped_invalid_p_values += 1
            continue

        rows.append(
            {
                "left_id": doc["left_id"],
                "right_id": doc["right_id"],
                "dataset_name": doc["dataset_name"],
                "impostor_generation_technique": doc["impostor_generation_technique"],
                "retrieval_index": doc.get("retrieval_index"),
                "ablation": ablation,
                "n_impostors": doc.get("n_impostors"),
                "method_key": method_key,
                "left_p_value": left_p_value,
                "right_p_value": right_p_value,
            }
        )

    if skipped_missing_pair:
        logger.info(
            "Skipped %d impostor-output records without a matching %s pair.",
            skipped_missing_pair,
            pair_label,
        )
    if skipped_invalid_p_values:
        logger.warning(
            "Skipped %d impostor-output records with invalid p-values.",
            skipped_invalid_p_values,
        )

    df = pd.DataFrame(rows)
    logger.info(
        "Loaded %d matched impostor-output records from %s.", len(df), collection.name
    )
    return df


def compute_type_one_error_curves(
    df: pd.DataFrame,
    alphas: np.ndarray,
    *,
    decision_rule: str = "pair_both",
) -> dict[str, dict[str, pd.DataFrame]]:
    return _compute_alpha_curves(
        df=df,
        alphas=alphas,
        rate_column="type_one_error",
        count_column="n_different_author_pairs",
        use_non_rejection_rate=False,
        record_label="different-author",
        decision_rule=decision_rule,
    )


def compute_same_author_non_rejection_curves(
    df: pd.DataFrame,
    alphas: np.ndarray,
    *,
    decision_rule: str = "pair_both",
) -> dict[str, dict[str, pd.DataFrame]]:
    return _compute_alpha_curves(
        df=df,
        alphas=alphas,
        rate_column="same_author_non_rejection_rate",
        count_column="n_same_author_pairs",
        use_non_rejection_rate=True,
        record_label="same-author",
        decision_rule=decision_rule,
    )


def _compute_alpha_curves(
    df: pd.DataFrame,
    alphas: np.ndarray,
    *,
    rate_column: str,
    count_column: str,
    use_non_rejection_rate: bool,
    record_label: str,
    decision_rule: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    curves: dict[str, dict[str, pd.DataFrame]] = {}
    if df.empty:
        return curves

    method_col = _method_column(df)
    for dataset_name, dataset_df in df.groupby("dataset_name", sort=False):
        dataset_curves: dict[str, pd.DataFrame] = {}
        for technique in _method_order(
            dataset_df[method_col].dropna().unique()
        ):
            technique_df = dataset_df[dataset_df[method_col] == technique]
            if technique_df.empty:
                continue
            technique_df = technique_df.drop_duplicates(
                subset=["left_id", "right_id", "dataset_name", method_col],
                keep="first",
            )

            left_p_values = technique_df["left_p_value"].to_numpy(dtype=float)
            right_p_values = technique_df["right_p_value"].to_numpy(dtype=float)
            if decision_rule == "pair_both":
                rejected = (
                    (left_p_values[None, :] <= alphas[:, None])
                    & (right_p_values[None, :] <= alphas[:, None])
                )
            elif decision_rule == "left_disputed_right_candidate":
                rejected = left_p_values[None, :] <= alphas[:, None]
            elif decision_rule == "right_disputed_left_candidate":
                rejected = right_p_values[None, :] <= alphas[:, None]
            else:
                raise ValueError(f"Unknown decision rule: {decision_rule}")
            if use_non_rejection_rate:
                rate = (~rejected).sum(axis=1) / len(technique_df)
            else:
                rate = rejected.sum(axis=1) / len(technique_df)
            dataset_curves[technique] = pd.DataFrame(
                {
                    "alpha": alphas,
                    rate_column: rate,
                    count_column: len(technique_df),
                }
            )
            logger.info(
                "%s / %s: %d %s records.",
                dataset_name,
                technique,
                len(technique_df),
                record_label,
            )
        if dataset_curves:
            curves[dataset_name] = dataset_curves

    return curves


def save_type_one_error_plots(
    curves: dict[str, dict[str, pd.DataFrame]],
    same_author_non_rejection_curves: dict[str, dict[str, pd.DataFrame]] | None = None,
    *,
    output_dir: Path,
    plot_suffix: str = "pair_both",
    plot_title_suffix: str = "",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    same_author_non_rejection_curves = same_author_non_rejection_curves or {}

    dataset_names = sorted(set(curves) | set(same_author_non_rejection_curves))
    for dataset_name in dataset_names:
        dataset_curves = curves.get(dataset_name, {})
        same_dataset_curves = same_author_non_rejection_curves.get(dataset_name, {})
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax_right = ax.twinx()
        max_error = 0.0
        max_same_non_rejection = 0.0
        alpha_min = math.inf
        alpha_max = 0.0

        technique_order = [
            technique
            for technique in _method_order(set(dataset_curves) | set(same_dataset_curves))
            if _plot_method_base(technique) not in PLOT_EXCLUDED_METHODS
        ]
        if not technique_order:
            logger.warning(
                "Skipping %s because no plottable methods remain after exclusions.",
                dataset_name,
            )
            plt.close(fig)
            continue

        for technique in technique_order:
            color = CONFIG.LABEL_COLORS.get(technique, "#4c4c4c")
            color = CONFIG.LABEL_COLORS.get(_plot_method_base(technique), color)
            if technique in dataset_curves:
                curve = dataset_curves[technique]
                max_error = max(max_error, float(curve["type_one_error"].max()))
                alpha_min = min(alpha_min, float(curve["alpha"].min()))
                alpha_max = max(alpha_max, float(curve["alpha"].max()))
                ax.plot(
                    curve["alpha"],
                    curve["type_one_error"],
                    color=color,
                    linewidth=2,
                    linestyle="-",
                    label=_display_method(technique),
                )

            if technique in same_dataset_curves:
                same_curve = same_dataset_curves[technique]
                max_same_non_rejection = max(
                    max_same_non_rejection,
                    float(same_curve["same_author_non_rejection_rate"].max()),
                )
                alpha_min = min(alpha_min, float(same_curve["alpha"].min()))
                alpha_max = max(alpha_max, float(same_curve["alpha"].max()))
                ax_right.plot(
                    same_curve["alpha"],
                    same_curve["same_author_non_rejection_rate"],
                    color=color,
                    linewidth=2,
                    linestyle=":",
                )

        display_dataset = _display_dataset(dataset_name)
        ax.set_title(
            r"$\alpha$" + f" Calibration on the {display_dataset} Dataset{plot_title_suffix}",
            fontsize=PLOT_TITLE_FONT_SIZE,
        )
        ax.set_xlabel(r"Significance level $\alpha$", fontsize=PLOT_LABEL_FONT_SIZE)
        ax.set_ylabel(
            "Type I Error Rate ",
            # r"$\frac{\#\ \mathrm{rejected}\ H_0\ \mathrm{among\ different-author\ pairs}}"
            # r"{\#\ \mathrm{different-author\ pairs}}$"
        fontsize=PLOT_LABEL_FONT_SIZE,
        )
        ax_right.set_ylabel(
            "Type II Error Rate ",
            # r"$\frac{\#\ \neg\mathrm{reject}\ H_0\ \mathrm{among\ same-author\ pairs}}"
            # r"{\#\ \mathrm{same-author\ pairs}}$"
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        ax.set_xlim(alpha_min, alpha_max)
        ax.set_ylim(0, min(1.0, max(0.105, max_error * 1.1)))
        ax_right.set_ylim(0, min(1.0, max(0.105, max_same_non_rejection * 1.1)))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_right.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.tick_params(axis="both", labelsize=PLOT_TICK_FONT_SIZE)
        ax_right.tick_params(axis="y", labelsize=PLOT_TICK_FONT_SIZE)
        ax.grid(True, linestyle="--", alpha=0.5)

        method_handles = [
            Line2D(
                [0],
                [0],
                color=CONFIG.LABEL_COLORS.get(
                    _plot_method_base(technique), "#4c4c4c"
                ),
                linewidth=2,
                label=_display_method(technique),
            )
            for technique in technique_order
        ]
        style_handles = [
            Line2D(
                [0],
                [0],
                color="#333333",
                linewidth=2,
                linestyle="-",
                label=r"Type I",
            ),
            Line2D(
                [0],
                [0],
                color="#333333",
                linewidth=2,
                linestyle=":",
                label=r"Type II",
            ),
        ]
        fig.legend(
            handles=method_handles,
            title="Impostor Generation",
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.14),
            borderaxespad=0,
            ncol=min(3, len(method_handles)),
        )
        fig.legend(
            handles=style_handles,
            title="Error Rate",
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.04),
            borderaxespad=0,
            ncol=len(style_handles),
        )

        fig.tight_layout(rect=(0, 0.24, 1, 1))

        filename = (
            "alpha_calibration_directional_hypothesis_tests_"
            f"{_safe_filename(dataset_name)}_{_safe_filename(plot_suffix)}"
        )
        for file_format in ("pdf", "svg"):
            output_path = output_dir / f"{filename}.{file_format}"
            fig.savefig(output_path, bbox_inches="tight")
            saved_paths.append(output_path)
            logger.info("Saved %s", output_path)
        plt.close(fig)

    return saved_paths


def write_attainable_fpr_table(
    *,
    output_dir: Path,
    n_trials_values: Iterable[int],
    n_impostors: int,
    bonferroni_factor: float,
) -> Path:
    """Write theoretical attainable FPRs for every critical count c."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if n_impostors < 1:
        raise ValueError("n_impostors must be at least 1.")
    if bonferroni_factor <= 0:
        raise ValueError("bonferroni_factor must be greater than 0.")

    p0 = 1.0 / (1 + n_impostors)
    rows: list[dict[str, float | int]] = []
    for n_trials in n_trials_values:
        if n_trials < 1:
            raise ValueError("n_trials values must be positive.")
        for critical_c in range(1, n_trials + 1):
            directional_fpr = sum(
                math.comb(n_trials, k) * (p0**k) * ((1 - p0) ** (n_trials - k))
                for k in range(critical_c, n_trials + 1)
            )
            rows.append(
                {
                    "n_trials": int(n_trials),
                    "n_impostors": int(n_impostors),
                    "null_probability": p0,
                    "critical_c": int(critical_c),
                    "directional_fpr": directional_fpr,
                    "bonferroni_corrected_alpha": min(
                        1.0, bonferroni_factor * directional_fpr
                    ),
                    "pair_level_fpr_both_directions_independent": directional_fpr
                    * directional_fpr,
                }
            )

    out_path = output_dir / "attainable_fpr_by_critical_count.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Saved attainable FPR table to %s", out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot alpha against empirical Type I error for corrected directional "
            "hypothesis tests."
        )
    )
    parser.add_argument(
        "--impostor-output-collection",
        default=CONFIG.MONGO_IMPOSTOR_OUTPUT_COLLECTION,
        help=(
            "MongoDB impostor output collection. Defaults to the repo config "
            f"({CONFIG.MONGO_IMPOSTOR_OUTPUT_COLLECTION!r})."
        ),
    )
    parser.add_argument(
        "--use-ablation-collection",
        action="store_true",
        help=(
            "Read from the configured ablation output collection "
            f"({CONFIG.MONGO_IMPOSTOR_ABLATION_OUTPUT_COLLECTION!r})."
        ),
    )
    parser.add_argument(
        "--all-pairs-collection",
        default=CONFIG.MONGO_ALL_PAIRS_COLLECTION,
        help=(
            "MongoDB collection containing pair ground truth. Defaults to the "
            f"repo config ({CONFIG.MONGO_ALL_PAIRS_COLLECTION!r})."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        action="append",
        dest="dataset_names",
        help="Restrict to one dataset. Repeat the option for multiple datasets.",
    )
    parser.add_argument(
        "--technique",
        action="append",
        dest="techniques",
        help=(
            "Restrict to one impostor_generation_technique. Repeat the option "
            "for multiple techniques."
        ),
    )
    parser.add_argument(
        "--ablation",
        action="append",
        dest="ablations",
        help=(
            "Restrict to one ablation class in an ablation output collection. "
            "Repeat the option for multiple ablations, e.g. "
            "PermutationCalibratedImpostorDetector."
        ),
    )
    parser.add_argument(
        "--left-p-value-field",
        default=LEFT_DISPUTED_RIGHT_P_VALUE_FIELD,
        help="Field containing left_disputed_right corrected p-values.",
    )
    parser.add_argument(
        "--right-p-value-field",
        default=RIGHT_DISPUTED_LEFT_P_VALUE_FIELD,
        help="Field containing right_disputed_left corrected p-values.",
    )
    parser.add_argument("--alpha-min", type=float, default=0.001)
    parser.add_argument("--alpha-max", type=float, default=0.075)
    parser.add_argument("--alpha-step", type=float, default=0.0001)
    parser.add_argument(
        "--alpha-grid",
        choices=("critical-k", "linear"),
        default="critical-k",
        help=(
            "Alpha values to evaluate. 'critical-k' evaluates only "
            "Bonferroni-corrected binomial tail probabilities where the critical "
            "success count changes; 'linear' keeps the previous fixed-step grid."
        ),
    )
    parser.add_argument(
        "--alpha-critical-n-trials",
        type=int,
        nargs="+",
        default=list(DEFAULT_ALPHA_CRITICAL_N_TRIALS),
        help=(
            "Trial counts used to build the critical-k alpha grid. Defaults to "
            "100 50 10."
        ),
    )
    parser.add_argument(
        "--n-impostors",
        type=int,
        default=N_IMPOSTORS,
        help=(
            "Number of impostors used to derive the null probability "
            "1 / (1 + n_impostors) for the critical-k alpha grid."
        ),
    )
    parser.add_argument(
        "--alpha-bonferroni-factor",
        type=float,
        default=DEFAULT_ALPHA_BONFERRONI_FACTOR,
        help=(
            "Correction factor applied to attainable one-direction binomial "
            "p-values when constructing the critical-k alpha grid."
        ),
    )
    parser.add_argument(
        "--attainable-fpr-output",
        type=Path,
        default=None,
        help=(
            "CSV path for theoretical attainable directional and pair-level FPRs "
            "for every critical count. Defaults to output-dir/"
            "attainable_fpr_by_critical_count.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PDF/SVG plots. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="MongoDB cursor batch size.",
    )
    parser.add_argument(
        "--remote-mongo",
        action="store_true",
        help="Use the in-cluster MongoDB connection instead of the local tunnel.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    if args.use_ablation_collection:
        args.impostor_output_collection = CONFIG.MONGO_IMPOSTOR_ABLATION_OUTPUT_COLLECTION
    if args.alpha_grid == "critical-k":
        alphas = _critical_alpha_grid(
            n_trials_values=args.alpha_critical_n_trials,
            n_impostors=args.n_impostors,
            bonferroni_factor=args.alpha_bonferroni_factor,
        )
    else:
        alphas = _alpha_grid(args.alpha_min, args.alpha_max, args.alpha_step)
    logger.info(
        "Using %s alpha grid with %d values from %.6g to %.6g.",
        args.alpha_grid,
        len(alphas),
        float(alphas[0]),
        float(alphas[-1]),
    )

    mongo = ParaphraseMongoDB(local_ray=not args.remote_mongo)
    impostor_output_collection = mongo.db[args.impostor_output_collection]
    all_pairs_collection = mongo.db[args.all_pairs_collection]

    different_author_pairs = _load_author_pairs(
        all_pairs_collection,
        same_value=False,
        dataset_names=args.dataset_names,
        batch_size=args.batch_size,
    )
    same_author_pairs = _load_author_pairs(
        all_pairs_collection,
        same_value=True,
        dataset_names=args.dataset_names,
        batch_size=args.batch_size,
    )
    logger.info(f"Using techniques: {args.techniques or 'all'}")
    df = _load_impostor_outputs_for_pairs(
        impostor_output_collection,
        pair_keys=different_author_pairs,
        pair_label="different-author",
        dataset_names=args.dataset_names,
        techniques=args.techniques,
        ablations=args.ablations,
        n_impostors=args.n_impostors,
        left_p_value_field=args.left_p_value_field,
        right_p_value_field=args.right_p_value_field,
        batch_size=args.batch_size,
    )
    same_author_df = _load_impostor_outputs_for_pairs(
        impostor_output_collection,
        pair_keys=same_author_pairs,
        pair_label="same-author",
        dataset_names=args.dataset_names,
        techniques=args.techniques,
        ablations=args.ablations,
        n_impostors=args.n_impostors,
        left_p_value_field=args.left_p_value_field,
        right_p_value_field=args.right_p_value_field,
        batch_size=args.batch_size,
    )
    if df.empty:
        raise ValueError(
            "No matched different-author impostor-output records found. Check "
            "the MongoDB collection names, p-value field names, and filters."
        )
    if same_author_df.empty:
        logger.warning(
            "No matched same-author impostor-output records found; right-axis "
            "same-author non-rejection curves will be omitted."
        )

    saved_paths: list[Path] = []
    plot_specs = [
        ("pair_both", "Pair-Level Both Directions", ""),
        (
            "left_disputed_right_candidate",
            "Left-Disputed Right-Candidate Direction",
            r" (Left $\rightarrow$ Right)",
        ),
        (
            "right_disputed_left_candidate",
            "Right-Disputed Left-Candidate Direction",
            r" (Right $\rightarrow$ Left)",
        ),
    ]
    for decision_rule, plot_suffix, plot_title_suffix in plot_specs:
        curves = compute_type_one_error_curves(
            df=df,
            alphas=alphas,
            decision_rule=decision_rule,
        )
        same_author_non_rejection_curves = compute_same_author_non_rejection_curves(
            df=same_author_df,
            alphas=alphas,
            decision_rule=decision_rule,
        )
        saved_paths.extend(
            save_type_one_error_plots(
                curves,
                same_author_non_rejection_curves=same_author_non_rejection_curves,
                output_dir=args.output_dir,
                plot_suffix=plot_suffix,
                plot_title_suffix=plot_title_suffix,
            )
        )
    attainable_fpr_output = args.attainable_fpr_output
    if attainable_fpr_output is None:
        attainable_fpr_output = write_attainable_fpr_table(
            output_dir=args.output_dir,
            n_trials_values=args.alpha_critical_n_trials,
            n_impostors=args.n_impostors,
            bonferroni_factor=args.alpha_bonferroni_factor,
        )
    else:
        attainable_fpr_output.parent.mkdir(parents=True, exist_ok=True)
        generated_path = write_attainable_fpr_table(
            output_dir=attainable_fpr_output.parent,
            n_trials_values=args.alpha_critical_n_trials,
            n_impostors=args.n_impostors,
            bonferroni_factor=args.alpha_bonferroni_factor,
        )
        if generated_path != attainable_fpr_output:
            generated_path.replace(attainable_fpr_output)
            logger.info("Moved attainable FPR table to %s", attainable_fpr_output)
    print("Saved plots:")
    for path in saved_paths:
        print(path)
    print("Saved attainable FPR table:")
    print(attainable_fpr_output)


if __name__ == "__main__":
    main()
