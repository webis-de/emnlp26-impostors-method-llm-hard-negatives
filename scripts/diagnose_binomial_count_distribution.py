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

"""Diagnose whether null-pair impostor counts follow the claimed binomial law.

For each different-author pair (``same == False``), this script loads the two
stored directional uncorrected binomial p-values from MongoDB and infers the
underlying directional success counts:

    X_i = sum_j Y_ij.

For a directional null test with ``n_trials`` feature-subsampling rounds and
``n_impostors`` impostors, the claimed null model is:

    X_i ~ Binomial(n_trials, 1 / (1 + n_impostors)).

The script compares the empirical count distribution to that theoretical
distribution and writes CSV summaries plus histogram/PMF overlay plots. It
does not write to MongoDB.

Usage:
    poetry run python scripts/diagnose_binomial_count_distribution.py \
        --dataset-name blog --dataset-name student_essays \
        --technique in_domain --technique two_step_llm \
        --n-trials 50 --n-impostors 50

Key outputs:
    counts_long.csv
        One row per null pair and direction, with inferred count X_i.
    distribution.csv
        Empirical vs theoretical probability for every count k.
    summary.csv
        Mean, variance, dispersion ratio, and distribution-distance diagnostics.
    plots/*.pdf and plots/*.svg
        Empirical count bars against the theoretical binomial probabilities.
    README.md
        Short guide to interpreting the generated files.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
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
from scipy.stats import binom, chisquare

from genai_detection.config import CONFIG
from genai_detection.experiments.reproduction.pan_metrics.run_pan_metrics import (
    IMPOSTOR_METHODS,
    N_IMPOSTORS,
    N_POTENTIAL_IMPOSTORS,
    ROUNDS,
)
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

LEFT_UNCORRECTED_P_VALUE_FIELD = "left_disputed_right_candidate_uncorrected_p_value"
RIGHT_UNCORRECTED_P_VALUE_FIELD = "right_disputed_left_candidate_uncorrected_p_value"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / CONFIG.SAVE_PATH
    / "binomial_count_diagnostics"
)
RETRIEVAL_INDEX_TO_METHOD = {
    retrieval_index: method_key
    for method_key, retrieval_index in CONFIG.RETRIEVAL_INDEX_TRANSLATIONS.items()
}
DIRECTION_LABELS = {
    "left_disputed_right_candidate": r"Left $\rightarrow$ Right",
    "right_disputed_left_candidate": r"Right $\rightarrow$ Left",
    "both_directions_pooled": "Both Directions Pooled",
}
DEFAULT_EXACT_COUNT_MAX = 10
DEFAULT_TAIL_BIN_WIDTH = 5
PLOT_TITLE_FONT_SIZE = 18
PLOT_LABEL_FONT_SIZE = 16
PLOT_TICK_FONT_SIZE = 13
PLOT_LEGEND_FONT_SIZE = 13
PLOT_LEGEND_TITLE_FONT_SIZE = 14
PLOT_ANNOTATION_FONT_SIZE = 12


def _safe_filename(value: str) -> str:
    safe_chars = [c if c.isalnum() or c in {"-", "_"} else "_" for c in value]
    return "_".join("".join(safe_chars).split("_")).strip("_") or "value"


def _display_dataset(dataset_name: str) -> str:
    return CONFIG.DATASET_TRANSLATIONS.get(
        dataset_name, dataset_name.replace("_", " ").title()
    )


def _display_method(method_name: str) -> str:
    return CONFIG.LABEL_TRANSLATIONS.get(
        method_name, method_name.replace("_", " ").title()
    )


def _method_color(method_name: str, fallback_index: int = 0) -> str:
    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#4c4c4c"])
    return CONFIG.LABEL_COLORS.get(
        method_name, fallback_colors[fallback_index % len(fallback_colors)]
    )


def _method_order(methods: Iterable[str]) -> list[str]:
    method_set = set(methods)
    configured = [method for method in CONFIG.LABEL_TRANSLATIONS if method in method_set]
    remaining = sorted(method_set - set(configured))
    return configured + remaining


def _plot_title(
    *,
    dataset_name: str,
    direction: str,
    method_key: str | None = None,
) -> str:
    method_part = "" if method_key is None else f" - {_display_method(method_key)}"
    title = (
        r"Candidate-Win Count under $H_0$"
        f" (Different-Author Pairs Only){method_part} on "
        f"{_display_dataset(dataset_name)} Dataset"
    )
    if direction != "both_directions_pooled":
        title = f"{title}\n{DIRECTION_LABELS.get(direction, direction)}"
    return title


def _x_axis_label(direction: str) -> str:
    if direction == "both_directions_pooled":
        return r"Pooled directional counts ($\neq$ final decision)"
    return r"Directional success count $X_i$"


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


def _load_null_pairs(
    mongo: ParaphraseMongoDB,
    *,
    dataset_names: list[str] | None,
    batch_size: int,
) -> set[tuple[Any, Any, str]]:
    query: dict[str, Any] = {"same": {"$eq": False, "$type": "bool"}}
    if dataset_names:
        query["dataset_name"] = {"$in": dataset_names}

    cursor = mongo.all_pairs_collection.find(
        query,
        {"_id": 0, "left_id": 1, "right_id": 1, "dataset_name": 1, "same": 1},
        batch_size=batch_size,
    )
    null_pairs = {
        (doc["left_id"], doc["right_id"], doc["dataset_name"])
        for doc in cursor
        if doc.get("same") is False
        and "left_id" in doc
        and "right_id" in doc
        and "dataset_name" in doc
    }
    logger.info("Loaded %d different-author null pairs.", len(null_pairs))
    return null_pairs


def _binomial_tail_values(n_trials: int, p0: float) -> np.ndarray:
    return np.asarray([binom.sf(k - 1, n_trials, p0) for k in range(n_trials + 1)])


def _infer_count_from_p_value(
    p_value: float,
    *,
    tail_values: np.ndarray,
    tolerance: float,
) -> tuple[int, float]:
    distances = np.abs(tail_values - p_value)
    count = int(np.argmin(distances))
    distance = float(distances[count])
    if distance > tolerance:
        logger.debug(
            "Closest inferred count has distance %.3g from p-value %.12g.",
            distance,
            p_value,
        )
    return count, distance


def load_inferred_counts(
    mongo: ParaphraseMongoDB,
    *,
    null_pairs: set[tuple[Any, Any, str]],
    dataset_names: list[str] | None,
    techniques: list[str] | None,
    n_trials: int,
    n_impostors: int,
    n_potential_impostors: int | None,
    left_p_value_field: str,
    right_p_value_field: str,
    batch_size: int,
    p_value_tolerance: float,
) -> pd.DataFrame:
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if n_impostors < 1:
        raise ValueError("n_impostors must be at least 1.")

    p0 = 1.0 / (1 + n_impostors)
    tail_values = _binomial_tail_values(n_trials=n_trials, p0=p0)
    query: dict[str, Any] = {
        left_p_value_field: {"$exists": True, "$ne": None},
        right_p_value_field: {"$exists": True, "$ne": None},
        "left_id": {"$exists": True, "$ne": None},
        "right_id": {"$exists": True, "$ne": None},
        "dataset_name": {"$exists": True, "$ne": None},
        "impostor_generation_technique": {"$exists": True, "$ne": None},
        "n_impostors": n_impostors,
    }
    if dataset_names:
        query["dataset_name"] = {"$in": dataset_names}
    technique_query_values = _technique_query_values(techniques)
    if technique_query_values:
        query["impostor_generation_technique"] = {"$in": technique_query_values}
    if n_potential_impostors is not None:
        query["n_potential_impostors"] = n_potential_impostors

    projection = {
        "_id": 0,
        "left_id": 1,
        "right_id": 1,
        "dataset_name": 1,
        "impostor_generation_technique": 1,
        "retrieval_index": 1,
        "n_impostors": 1,
        "n_potential_impostors": 1,
        left_p_value_field: 1,
        right_p_value_field: 1,
    }

    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, str, str]] = set()
    skipped_non_null = 0
    skipped_invalid = 0
    skipped_duplicate = 0
    cursor = mongo.impostor_output_collection.find(
        query, projection, batch_size=batch_size
    ).sort("_id", 1)
    for doc in cursor:
        pair_key = (doc["left_id"], doc["right_id"], doc["dataset_name"])
        if pair_key not in null_pairs:
            skipped_non_null += 1
            continue

        method_key = _impostor_method_key(doc)
        if techniques and (
            doc["impostor_generation_technique"] not in techniques
            and method_key not in techniques
        ):
            continue

        dedupe_key = (*pair_key, method_key)
        if dedupe_key in seen:
            skipped_duplicate += 1
            continue
        seen.add(dedupe_key)

        left_p_value = _float_or_none(doc.get(left_p_value_field))
        right_p_value = _float_or_none(doc.get(right_p_value_field))
        if left_p_value is None or right_p_value is None:
            skipped_invalid += 1
            continue

        for direction, p_value in (
            ("left_disputed_right_candidate", left_p_value),
            ("right_disputed_left_candidate", right_p_value),
        ):
            inferred_count, p_value_distance = _infer_count_from_p_value(
                p_value,
                tail_values=tail_values,
                tolerance=p_value_tolerance,
            )
            rows.append(
                {
                    "dataset_name": doc["dataset_name"],
                    "dataset_label": _display_dataset(doc["dataset_name"]),
                    "method_key": method_key,
                    "method_label": _display_method(method_key),
                    "left_id": str(doc["left_id"]),
                    "right_id": str(doc["right_id"]),
                    "direction": direction,
                    "n_trials": n_trials,
                    "n_impostors": n_impostors,
                    "n_potential_impostors": doc.get("n_potential_impostors"),
                    "null_probability": p0,
                    "inferred_count": inferred_count,
                    "p_value_uncorrected": p_value,
                    "p_value_inference_distance": p_value_distance,
                    "p_value_inference_within_tolerance": (
                        p_value_distance <= p_value_tolerance
                    ),
                }
            )

    if skipped_non_null:
        logger.info("Skipped %d impostor outputs not matching null pairs.", skipped_non_null)
    if skipped_duplicate:
        logger.info("Skipped %d duplicate method/pair records.", skipped_duplicate)
    if skipped_invalid:
        logger.warning("Skipped %d records with invalid p-values.", skipped_invalid)

    df = pd.DataFrame(rows)
    logger.info("Loaded %d directional null-count rows.", len(df))
    return df


def _summarize_group(group: pd.DataFrame, *, n_trials: int, p0: float) -> dict[str, Any]:
    counts = group["inferred_count"].to_numpy(dtype=int)
    n_null_directions = int(len(counts))
    empirical_mean = float(np.mean(counts)) if n_null_directions else float("nan")
    empirical_variance = (
        float(np.var(counts, ddof=1)) if n_null_directions > 1 else float("nan")
    )
    theoretical_mean = n_trials * p0
    theoretical_variance = n_trials * p0 * (1 - p0)
    dispersion_ratio = (
        empirical_variance / theoretical_variance
        if theoretical_variance > 0 and math.isfinite(empirical_variance)
        else float("nan")
    )

    support = np.arange(n_trials + 1)
    observed_counts = np.bincount(counts, minlength=n_trials + 1).astype(float)
    empirical_probs = observed_counts / n_null_directions
    theoretical_probs = binom.pmf(support, n_trials, p0)
    expected_counts = theoretical_probs * n_null_directions

    nonzero_expected = expected_counts > 0
    if nonzero_expected.any():
        chi = chisquare(
            f_obs=observed_counts[nonzero_expected],
            f_exp=expected_counts[nonzero_expected],
        )
        chi_square_statistic = float(chi.statistic)
        chi_square_p_value = float(chi.pvalue)
    else:
        chi_square_statistic = float("nan")
        chi_square_p_value = float("nan")

    cdf_empirical = np.cumsum(empirical_probs)
    cdf_theoretical = binom.cdf(support, n_trials, p0)
    ks_statistic = float(np.max(np.abs(cdf_empirical - cdf_theoretical)))
    total_variation_distance = float(
        0.5 * np.sum(np.abs(empirical_probs - theoretical_probs))
    )

    return {
        "n_null_directions": n_null_directions,
        "n_trials": n_trials,
        "n_impostors": int(group["n_impostors"].iloc[0]),
        "null_probability": p0,
        "theoretical_mean": theoretical_mean,
        "empirical_mean": empirical_mean,
        "theoretical_variance": theoretical_variance,
        "empirical_variance": empirical_variance,
        "dispersion_ratio": dispersion_ratio,
        "ks_statistic": ks_statistic,
        "total_variation_distance": total_variation_distance,
        "chi_square_statistic": chi_square_statistic,
        "chi_square_p_value": chi_square_p_value,
        "max_p_value_inference_distance": float(
            group["p_value_inference_distance"].max()
        ),
        "share_p_values_within_tolerance": float(
            group["p_value_inference_within_tolerance"].mean()
        ),
    }


def compute_summary_and_distribution(
    counts_df: pd.DataFrame,
    *,
    n_trials: int,
    n_impostors: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    p0 = 1.0 / (1 + n_impostors)
    summary_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    support = np.arange(n_trials + 1)
    theoretical_probs = binom.pmf(support, n_trials, p0)

    group_specs: list[tuple[str, pd.DataFrame]] = []
    for key, group in counts_df.groupby(["dataset_name", "method_key", "direction"]):
        dataset_name, method_key, direction = key
        group_specs.append((f"{dataset_name}\t{method_key}\t{direction}", group))
    for key, group in counts_df.groupby(["dataset_name", "method_key"]):
        dataset_name, method_key = key
        pooled = group.copy()
        pooled["direction"] = "both_directions_pooled"
        group_specs.append(
            (f"{dataset_name}\t{method_key}\tboth_directions_pooled", pooled)
        )

    for _, group in group_specs:
        dataset_name = str(group["dataset_name"].iloc[0])
        method_key = str(group["method_key"].iloc[0])
        direction = str(group["direction"].iloc[0])
        summary = _summarize_group(group, n_trials=n_trials, p0=p0)
        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "dataset_label": _display_dataset(dataset_name),
                "method_key": method_key,
                "method_label": _display_method(method_key),
                "direction": direction,
                "direction_label": DIRECTION_LABELS.get(direction, direction),
                **summary,
            }
        )

        counts = group["inferred_count"].to_numpy(dtype=int)
        observed_counts = np.bincount(counts, minlength=n_trials + 1).astype(int)
        n_obs = int(len(counts))
        for count_value, observed_count, theoretical_probability in zip(
            support, observed_counts, theoretical_probs
        ):
            distribution_rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_label": _display_dataset(dataset_name),
                    "method_key": method_key,
                    "method_label": _display_method(method_key),
                    "direction": direction,
                    "direction_label": DIRECTION_LABELS.get(direction, direction),
                    "n_trials": n_trials,
                    "n_impostors": n_impostors,
                    "null_probability": p0,
                    "count": int(count_value),
                    "observed_count": int(observed_count),
                    "observed_probability": (
                        float(observed_count / n_obs) if n_obs else float("nan")
                    ),
                    "theoretical_probability": float(theoretical_probability),
                    "expected_count": float(theoretical_probability * n_obs),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(distribution_rows)


def save_plots(
    *,
    distribution_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for key, group in distribution_df.groupby(["dataset_name", "method_key", "direction"]):
        dataset_name, method_key, direction = key
        summary_match = summary_df[
            (summary_df["dataset_name"] == dataset_name)
            & (summary_df["method_key"] == method_key)
            & (summary_df["direction"] == direction)
        ]
        if summary_match.empty:
            continue
        summary = summary_match.iloc[0]

        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        ax.bar(
            group["count"],
            group["observed_probability"],
            color="#7aa6c2",
            edgecolor="#31586f",
            linewidth=0.8,
            label="Empirical",
        )
        ax.plot(
            group["count"],
            group["theoretical_probability"],
            color="#b23b3b",
            linewidth=2,
            marker="o",
            markersize=3,
            label="Binomial PMF",
        )
        ax.set_title(
            _plot_title(
                dataset_name=dataset_name,
                method_key=method_key,
                direction=direction,
            ),
            fontsize=PLOT_TITLE_FONT_SIZE,
        )
        ax.set_xlabel(_x_axis_label(direction), fontsize=PLOT_LABEL_FONT_SIZE)
        ax.set_ylabel("Probability", fontsize=PLOT_LABEL_FONT_SIZE)
        ax.tick_params(axis="both", labelsize=PLOT_TICK_FONT_SIZE)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(
            frameon=False,
            fontsize=PLOT_LEGEND_FONT_SIZE,
        )

        annotation = (
            f"n={int(summary['n_null_directions'])}\n"
            f"mean: {summary['empirical_mean']:.3f} vs {summary['theoretical_mean']:.3f}\n"
            f"var: {summary['empirical_variance']:.3f} vs {summary['theoretical_variance']:.3f}\n"
            f"D={summary['dispersion_ratio']:.3f}"
        )
        ax.text(
            0.98,
            0.96,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
            fontsize=PLOT_ANNOTATION_FONT_SIZE,
        )
        fig.tight_layout()

        filename = (
            f"binomial_count_distribution_{_safe_filename(dataset_name)}_"
            f"{_safe_filename(method_key)}_{_safe_filename(direction)}"
        )
        for file_format in ("pdf", "svg"):
            out_path = plot_dir / f"{filename}.{file_format}"
            fig.savefig(out_path, bbox_inches="tight")
            saved_paths.append(out_path)
        plt.close(fig)

    logger.info("Saved %d plot files in %s.", len(saved_paths), plot_dir)
    return saved_paths


def _count_bins(
    *,
    n_trials: int,
    exact_count_max: int,
    tail_bin_width: int,
) -> list[tuple[int, int, str]]:
    if exact_count_max < 0:
        raise ValueError("exact_count_max must be non-negative.")
    if tail_bin_width < 1:
        raise ValueError("tail_bin_width must be positive.")

    bins: list[tuple[int, int, str]] = []
    exact_end = min(exact_count_max, n_trials)
    for count in range(exact_end + 1):
        bins.append((count, count, str(count)))
    start = exact_end + 1
    while start <= n_trials:
        end = min(start + tail_bin_width - 1, n_trials)
        bins.append((start, end, f"{start}-{end}" if start != end else str(start)))
        start = end + 1
    return bins


def _binned_probabilities(
    distribution_group: pd.DataFrame,
    *,
    bins: Sequence[tuple[int, int, str]],
) -> tuple[np.ndarray, np.ndarray]:
    observed: list[float] = []
    theoretical: list[float] = []
    for start, end, _ in bins:
        mask = (distribution_group["count"] >= start) & (distribution_group["count"] <= end)
        observed.append(float(distribution_group.loc[mask, "observed_probability"].sum()))
        theoretical.append(
            float(distribution_group.loc[mask, "theoretical_probability"].sum())
        )
    return np.asarray(observed), np.asarray(theoretical)


def save_combined_approach_plots(
    *,
    distribution_df: pd.DataFrame,
    output_dir: Path,
    exact_count_max: int = DEFAULT_EXACT_COUNT_MAX,
    tail_bin_width: int = DEFAULT_TAIL_BIN_WIDTH,
    alpha: float = 0.65,
) -> list[Path]:
    """Save combined empirical/theoretical count plots across all approaches."""
    plot_dir = output_dir / "plots" / "combined_approaches"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for key, group in distribution_df.groupby(["dataset_name", "direction"]):
        dataset_name, direction = key
        n_trials = int(group["n_trials"].iloc[0])
        bins = _count_bins(
            n_trials=n_trials,
            exact_count_max=exact_count_max,
            tail_bin_width=tail_bin_width,
        )
        x = np.arange(len(bins), dtype=float)
        labels = [label for _, _, label in bins]
        method_order = _method_order(group["method_key"].dropna().unique())
        if not method_order:
            continue

        fig_width = max(9.5, 0.42 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_width, 5.4))
        bar_width = min(0.8 / max(len(method_order), 1), 0.16)

        theoretical_reference: np.ndarray | None = None
        for method_idx, method_key in enumerate(method_order):
            method_group = group[group["method_key"] == method_key]
            if method_group.empty:
                continue
            observed, theoretical = _binned_probabilities(method_group, bins=bins)
            if theoretical_reference is None:
                theoretical_reference = theoretical
            offset = (method_idx - (len(method_order) - 1) / 2) * bar_width
            color = _method_color(method_key, fallback_index=method_idx)
            ax.bar(
                x + offset,
                observed,
                width=bar_width,
                color=color,
                alpha=alpha,
                edgecolor=color,
                linewidth=0.7,
                label=_display_method(method_key),
            )

        if theoretical_reference is not None:
            ax.plot(
                x,
                theoretical_reference,
                color="#111111",
                linestyle="-",
                linewidth=2.4,
                marker="o",
                markersize=3,
                label="Theoretical binomial",
            )

        ax.set_title(
            _plot_title(dataset_name=dataset_name, direction=direction),
            fontsize=PLOT_TITLE_FONT_SIZE,
        )
        ax.set_xlabel(
            _x_axis_label(direction),
            fontsize=PLOT_LABEL_FONT_SIZE,
        )
        ax.set_ylabel("Probability mass", fontsize=PLOT_LABEL_FONT_SIZE)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.tick_params(axis="both", labelsize=PLOT_TICK_FONT_SIZE)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(
            title="Empirical Bars / Reference Line",
            frameon=False,
            fontsize=PLOT_LEGEND_FONT_SIZE,
            title_fontsize=PLOT_LEGEND_TITLE_FONT_SIZE,
            ncol=2,
        )
        fig.tight_layout()

        filename = (
            f"combined_binomial_count_distribution_{_safe_filename(dataset_name)}_"
            f"{_safe_filename(direction)}"
        )
        for file_format in ("pdf", "svg"):
            out_path = plot_dir / f"{filename}.{file_format}"
            fig.savefig(out_path, bbox_inches="tight")
            saved_paths.append(out_path)
        plt.close(fig)

    logger.info("Saved %d combined approach plot files in %s.", len(saved_paths), plot_dir)
    return saved_paths


def write_readme(output_dir: Path) -> Path:
    text = """# Binomial Count Diagnostics

This directory contains diagnostics for the directional binomial-count assumption
of the impostor hypothesis test, restricted to null pairs (`same == False`).

## Files

- `counts_long.csv`: one row per null pair and direction. This is the raw audit
  table. Check `inferred_count`, `p_value_uncorrected`,
  `p_value_inference_distance`, and `p_value_inference_within_tolerance` here.
- `summary.csv`: one row per dataset, method, and direction, plus
  `both_directions_pooled`. This is the first file to inspect for binomial
  validity. The key column is `dispersion_ratio`.
- `distribution.csv`: empirical and theoretical probabilities for every count
  value. Use this to inspect which counts drive deviations from the binomial
  model.
- `plots/`: bar plots of the empirical count distribution overlaid with the
  theoretical binomial PMF. These are the main diagnostic figures.
- `plots/combined_approaches/`: combined figures comparing all impostor
  generation approaches for the same dataset and direction. These use exact
  bins at small counts and wider bins in the tail.

## Interpretation

The binomial model predicts:

`X_i ~ Binomial(n_trials, 1 / (1 + n_impostors))`

In `summary.csv`, inspect:

- `empirical_mean` vs `theoretical_mean`
- `empirical_variance` vs `theoretical_variance`
- `dispersion_ratio = empirical_variance / theoretical_variance`
- `ks_statistic` and `total_variation_distance`

As a rule of thumb:

- `dispersion_ratio` near `1`: compatible with binomial dispersion.
- `dispersion_ratio` much larger than `1`: overdispersion, consistent with
  dependent or heterogeneous trials.
- `dispersion_ratio` below `1`: underdispersion.

The chi-square p-value is included as a rough descriptive check. It can be
unstable when many expected bin counts are small, so use it together with the
dispersion ratio and the plots rather than as the sole decision criterion.

For comparing impostor-generation approaches directly, use
`plots/combined_approaches/`. Solid transparent bars show empirical probability
mass per approach; the solid black line shows the shared theoretical binomial
mass for the same count bins.
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare empirical directional null-pair impostor counts to the "
            "claimed binomial count distribution."
        )
    )
    parser.add_argument(
        "--dataset-name",
        action="append",
        dest="dataset_names",
        help="Restrict to one dataset. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--technique",
        action="append",
        dest="techniques",
        default=None,
        help=(
            "Restrict to one impostor-generation technique. Repeat for multiple "
            "techniques. Defaults to the configured impostors methods."
        ),
    )
    parser.add_argument("--n-trials", type=int, default=ROUNDS)
    parser.add_argument("--n-impostors", type=int, default=N_IMPOSTORS)
    parser.add_argument(
        "--n-potential-impostors",
        type=int,
        default=N_POTENTIAL_IMPOSTORS,
        help="Filter by n_potential_impostors. Use -1 to disable this filter.",
    )
    parser.add_argument(
        "--left-p-value-field",
        default=LEFT_UNCORRECTED_P_VALUE_FIELD,
        help="MongoDB field containing left-disputed/right-candidate uncorrected p-values.",
    )
    parser.add_argument(
        "--right-p-value-field",
        default=RIGHT_UNCORRECTED_P_VALUE_FIELD,
        help="MongoDB field containing right-disputed/left-candidate uncorrected p-values.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--p-value-tolerance",
        type=float,
        default=1e-9,
        help="Tolerance for matching stored p-values to attainable binomial tails.",
    )
    parser.add_argument(
        "--combined-exact-count-max",
        type=int,
        default=DEFAULT_EXACT_COUNT_MAX,
        help=(
            "For combined approach plots, keep exact one-count bins through this "
            "count before switching to wider tail bins."
        ),
    )
    parser.add_argument(
        "--combined-tail-bin-width",
        type=int,
        default=DEFAULT_TAIL_BIN_WIDTH,
        help="Width of tail bins in combined approach plots.",
    )
    parser.add_argument(
        "--combined-alpha",
        type=float,
        default=0.65,
        help="Transparency for empirical bars in combined approach plots.",
    )
    parser.add_argument(
        "--remote-mongo",
        action="store_true",
        help="Use the in-cluster MongoDB connection instead of the local tunnel.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    techniques = args.techniques or list(IMPOSTOR_METHODS)
    n_potential_impostors = (
        None if args.n_potential_impostors == -1 else args.n_potential_impostors
    )

    logger.info("Output directory: %s", args.output_dir)
    logger.info(
        "Running binomial-count diagnostics for datasets=%s, techniques=%s, "
        "n_trials=%d, n_impostors=%d, n_potential_impostors=%s.",
        args.dataset_names or "all",
        techniques,
        args.n_trials,
        args.n_impostors,
        n_potential_impostors,
    )

    mongo = ParaphraseMongoDB(local_ray=not args.remote_mongo)
    null_pairs = _load_null_pairs(
        mongo,
        dataset_names=args.dataset_names,
        batch_size=args.batch_size,
    )
    counts_df = load_inferred_counts(
        mongo,
        null_pairs=null_pairs,
        dataset_names=args.dataset_names,
        techniques=techniques,
        n_trials=args.n_trials,
        n_impostors=args.n_impostors,
        n_potential_impostors=n_potential_impostors,
        left_p_value_field=args.left_p_value_field,
        right_p_value_field=args.right_p_value_field,
        batch_size=args.batch_size,
        p_value_tolerance=args.p_value_tolerance,
    )
    if counts_df.empty:
        raise ValueError(
            "No directional null-count rows found. Check dataset/technique filters, "
            "n_trials/n_impostors, p-value fields, and MongoDB collection contents."
        )

    summary_df, distribution_df = compute_summary_and_distribution(
        counts_df,
        n_trials=args.n_trials,
        n_impostors=args.n_impostors,
    )

    counts_path = args.output_dir / "counts_long.csv"
    summary_path = args.output_dir / "summary.csv"
    distribution_path = args.output_dir / "distribution.csv"
    counts_df.to_csv(counts_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    distribution_df.to_csv(distribution_path, index=False)
    plot_paths = save_plots(
        distribution_df=distribution_df,
        summary_df=summary_df,
        output_dir=args.output_dir,
    )
    combined_plot_paths = save_combined_approach_plots(
        distribution_df=distribution_df,
        output_dir=args.output_dir,
        exact_count_max=args.combined_exact_count_max,
        tail_bin_width=args.combined_tail_bin_width,
        alpha=args.combined_alpha,
    )
    readme_path = write_readme(args.output_dir)

    logger.info("Saved counts_long.csv: raw inferred X_i count rows.")
    logger.info("Saved summary.csv: primary validity diagnostics; inspect dispersion_ratio.")
    logger.info("Saved distribution.csv: empirical vs theoretical probability per count.")
    logger.info("Saved plots/: empirical histograms overlaid with binomial PMF.")
    logger.info(
        "Saved plots/combined_approaches/: combined empirical/theoretical figures across approaches."
    )
    logger.info("Saved README.md: output-file guide and interpretation notes.")

    print("Saved binomial-count diagnostic outputs:")
    print(f"- counts_long: {counts_path}")
    print(f"- summary: {summary_path}")
    print(f"- distribution: {distribution_path}")
    print(f"- readme: {readme_path}")
    print(f"- plots: {args.output_dir / 'plots'} ({len(plot_paths)} files)")
    print(
        f"- combined plots: {args.output_dir / 'plots' / 'combined_approaches'} "
        f"({len(combined_plot_paths)} files)"
    )
    print("\nFor binomial validity, start with summary.csv dispersion_ratio and the matching plots.")


if __name__ == "__main__":
    main()
