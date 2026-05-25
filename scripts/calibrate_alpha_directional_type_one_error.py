#!/usr/bin/env python3
from __future__ import annotations

"""Calibrate alpha for directional hypothesis-test Type I error.

The script reads corrected directional p-values from MongoDB impostor outputs,
keeps only pairs marked as different-author pairs in ``all_pairs``, and plots
the empirical Type I error rate for alpha values from 0.001 to 0.1.

Usage:
    poetry run python scripts/calibrate_alpha_directional_type_one_error.py
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
from matplotlib.ticker import PercentFormatter
from pymongo.collection import Collection

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

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
    / "type_one_err_calibration"
)


def _safe_filename(value: str) -> str:
    safe_chars = [c if c.isalnum() or c in {"-", "_"} else "_" for c in value]
    return "_".join("".join(safe_chars).split("_")).strip("_") or "dataset"


def _display_dataset(dataset_name: str) -> str:
    return CONFIG.DATASET_TRANSLATIONS.get(
        dataset_name, dataset_name.replace("_", " ").title()
    )


def _display_method(method_name: str) -> str:
    return CONFIG.LABEL_TRANSLATIONS.get(
        method_name, method_name.replace("_", " ").title()
    )


def _method_order(methods: Iterable[str]) -> list[str]:
    method_set = set(methods)
    configured = [method for method in CONFIG.LABEL_TRANSLATIONS if method in method_set]
    remaining = sorted(method_set - set(configured))
    return configured + remaining


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


def _load_different_author_pairs(
    collection: Collection,
    *,
    dataset_names: list[str] | None,
    batch_size: int,
) -> set[tuple[Any, Any, str]]:
    # keep only different author pairs
    query: dict[str, Any] = {"same": {"$eq": False, "$type": "bool"}}
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
        if doc.get("same") is False
        and "left_id" in doc
        and "right_id" in doc
        and "dataset_name" in doc
    }
    logger.info("Loaded %d different-author pairs from %s.", len(pairs), collection.name)
    return pairs


def _load_impostor_outputs_for_pairs(
    collection: Collection,
    *,
    different_author_pairs: set[tuple[Any, Any, str]],
    dataset_names: list[str] | None,
    techniques: list[str] | None,
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
    if techniques:
        query["impostor_generation_technique"] = {"$in": techniques}

    projection = {
        "_id": 0,
        "left_id": 1,
        "right_id": 1,
        "dataset_name": 1,
        "impostor_generation_technique": 1,
        left_p_value_field: 1,
        right_p_value_field: 1,
    }

    rows: list[dict[str, Any]] = []
    skipped_invalid_p_values = 0
    skipped_missing_pair = 0
    cursor = collection.find(query, projection, batch_size=batch_size).sort("_id", 1)
    for doc in cursor:
        pair_key = (doc["left_id"], doc["right_id"], doc["dataset_name"])
        if pair_key not in different_author_pairs:
            skipped_missing_pair += 1
            continue

        left_p_value = _float_or_none(doc.get(left_p_value_field))
        right_p_value = _float_or_none(doc.get(right_p_value_field))
        if left_p_value is None or right_p_value is None:
            skipped_invalid_p_values += 1
            continue

        rows.append(
            {
                "dataset_name": doc["dataset_name"],
                "impostor_generation_technique": doc[
                    "impostor_generation_technique"
                ],
                "left_p_value": left_p_value,
                "right_p_value": right_p_value,
            }
        )

    if skipped_missing_pair:
        logger.info(
            "Skipped %d impostor-output records without a matching different-author pair.",
            skipped_missing_pair,
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
) -> dict[str, dict[str, pd.DataFrame]]:
    curves: dict[str, dict[str, pd.DataFrame]] = {}
    if df.empty:
        return curves

    for dataset_name, dataset_df in df.groupby("dataset_name", sort=False):
        dataset_curves: dict[str, pd.DataFrame] = {}
        for technique in _method_order(
            dataset_df["impostor_generation_technique"].dropna().unique()
        ):
            technique_df = dataset_df[
                dataset_df["impostor_generation_technique"] == technique
            ]
            if technique_df.empty:
                continue

            left_p_values = technique_df["left_p_value"].to_numpy(dtype=float)
            right_p_values = technique_df["right_p_value"].to_numpy(dtype=float)
            rejected = (
                (left_p_values[None, :] < alphas[:, None])
                & (right_p_values[None, :] < alphas[:, None])
            )
            type_one_error = rejected.sum(axis=1) / len(technique_df)
            dataset_curves[technique] = pd.DataFrame(
                {
                    "alpha": alphas,
                    "type_one_error": type_one_error,
                    "n_different_author_pairs": len(technique_df),
                }
            )
            logger.info(
                "%s / %s: %d different-author records.",
                dataset_name,
                technique,
                len(technique_df),
            )
        if dataset_curves:
            curves[dataset_name] = dataset_curves

    return curves


def save_type_one_error_plots(
    curves: dict[str, dict[str, pd.DataFrame]],
    *,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for dataset_name, dataset_curves in curves.items():
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        max_error = 0.0
        alpha_min = math.inf
        alpha_max = 0.0

        for technique in _method_order(dataset_curves.keys()):
            curve = dataset_curves[technique]
            max_error = max(max_error, float(curve["type_one_error"].max()))
            alpha_min = min(alpha_min, float(curve["alpha"].min()))
            alpha_max = max(alpha_max, float(curve["alpha"].max()))
            ax.plot(
                curve["alpha"],
                curve["type_one_error"],
                color=CONFIG.LABEL_COLORS.get(technique, "#4c4c4c"),
                linewidth=2,
                label=_display_method(technique),
            )

        display_dataset = _display_dataset(dataset_name)
        ax.set_title(f"Alpha Calibration: {display_dataset}")
        ax.set_xlabel(r"Alpha threshold ($\alpha$)")
        ax.set_ylabel(
            "Type I error rate "
            r"$\frac{\#\ \mathrm{rejected}\ H_0\ \mathrm{among\ different-author\ pairs}}"
            r"{\#\ \mathrm{different-author\ pairs}}$"
        )
        ax.set_xlim(alpha_min, alpha_max)
        ax.set_ylim(0, min(1.0, max(0.105, max_error * 1.1)))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(title="Impostor generation", frameon=False)

        # caption = (
        #     "Caption: Empirical Type I error on different-author pairs "
        #     "(all_pairs.same = false). $H_0$ is rejected only when both corrected "
        #     "directional p-values are below alpha."
        # )
        # fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, 0.08, 1, 1))

        filename = (
            "type_one_error_alpha_calibration_directional_hypothesis_tests_"
            f"{_safe_filename(dataset_name)}"
        )
        for file_format in ("pdf", "svg"):
            output_path = output_dir / f"{filename}.{file_format}"
            fig.savefig(output_path, bbox_inches="tight")
            saved_paths.append(output_path)
            logger.info("Saved %s", output_path)
        plt.close(fig)

    return saved_paths


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
    alphas = _alpha_grid(args.alpha_min, args.alpha_max, args.alpha_step)

    mongo = ParaphraseMongoDB(local_ray=not args.remote_mongo)
    impostor_output_collection = mongo.db[args.impostor_output_collection]
    all_pairs_collection = mongo.db[args.all_pairs_collection]

    different_author_pairs = _load_different_author_pairs(
        all_pairs_collection,
        dataset_names=args.dataset_names,
        batch_size=args.batch_size,
    )
    df = _load_impostor_outputs_for_pairs(
        impostor_output_collection,
        different_author_pairs=different_author_pairs,
        dataset_names=args.dataset_names,
        techniques=args.techniques,
        left_p_value_field=args.left_p_value_field,
        right_p_value_field=args.right_p_value_field,
        batch_size=args.batch_size,
    )
    if df.empty:
        raise ValueError(
            "No matched different-author impostor-output records found. Check "
            "the MongoDB collection names, p-value field names, and filters."
        )

    curves = compute_type_one_error_curves(df=df, alphas=alphas)
    saved_paths = save_type_one_error_plots(curves, output_dir=args.output_dir)
    print("Saved plots:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
