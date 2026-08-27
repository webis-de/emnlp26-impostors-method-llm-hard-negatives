"""Simple portion_delete sweep for in-domain impostors.

This version keeps the experiment deliberately small:

1. Load dataset pairs from MongoDB.
2. Generate normal in-domain impostors for each text.
3. Use ImpostorDetector with impostor_technique="in_domain" for scoring.
4. Re-score the same pair with different Scorer(portion_delete=...) values.
5. Compute PAN metrics and save CSV/JSON/plots under
   results/in_domain_impostors/n_delete_sweep.

Important: ImpostorDetector's MongoDB cache key does not include portion_delete.
This experiment therefore swaps in a no-cache output collection before calling
get_score(), so each scorer configuration is recomputed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from bson import ObjectId

from genai_detection.config import CONFIG
from genai_detection.detectors.components.scorer import Scorer
from genai_detection.detectors.components.vector_similarity import minmax_similarity
from genai_detection.detectors.impostor import ImpostorDetector
from genai_detection.experiments.reproduction.pan_metrics.pan_cv import SplitManager
from genai_detection.experiments.reproduction.pan_metrics.pan_metric_computation import (
    PANMetricComputer,
)
from genai_detection.mongo_db.mongo_utils import PROJECT_ROOT, ParaphraseMongoDB

logger = logging.getLogger(__name__)

METRICS = (
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auroc",
    "c_at_1",
    "auroc_c_at_1",
)

DEFAULT_PORTION_DELETE_VALUES = [round(v, 2) for v in np.arange(0.1, 1.0, 0.1)]

METHOD_NAME = "in_domain"


class _NoCacheCollection:
    """Minimal Mongo collection stand-in to force fresh detector scores."""

    name = "portion_delete_sweep_no_cache"

    class _EmptyCursor:
        def sort(self, *_args, **_kwargs):
            return self

        def __iter__(self):
            return iter(())

    def find(self, *_args, **_kwargs):
        return self._EmptyCursor()

    def insert_one(self, *_args, **_kwargs):
        return None


def default_save_dir() -> Path:
    return PROJECT_ROOT / CONFIG.SAVE_PATH / "in_domain_impostors" / "n_delete_sweep"


def validate_portion_delete_values(values: Iterable[float]) -> list[float]:
    values = sorted({float(v) for v in values})
    if not values:
        raise ValueError("At least one portion_delete value is required.")
    for value in values:
        if not 0.0 <= value < 1.0:
            raise ValueError(f"portion_delete must be in [0.0, 1.0), got {value}.")
    return values


def load_dataset_pairs(
    mongo: ParaphraseMongoDB,
    dataset_name: str,
    limit_pairs: int | None = None,
) -> list[dict[str, Any]]:
    """Load pair ids and labels for one dataset."""
    cursor = mongo.all_pairs_collection.find(
        {"dataset_name": dataset_name},
        {"left_id": 1, "right_id": 1, "same": 1},
    ).sort("_id", 1)

    rows = []
    # if maximum number of samples is specified, ensure same number for both class (equal class distribution)
    targets = None if limit_pairs is None else {0: limit_pairs // 2, 1: limit_pairs - limit_pairs // 2}
    counts = {0: 0, 1: 0}

    for doc in cursor:
        label = int(doc["same"])

        if targets is not None:
            if counts[label] >= targets[label]:
                continue
            counts[label] += 1

        rows.append({
                "left_id": ObjectId(doc["left_id"]), "right_id": ObjectId(doc["right_id"]), "same": label,
                })

        if targets is not None and counts == targets:
            break
    return rows


def score_pair_for_sweep(
    *,
    detector: ImpostorDetector,
    left_id: ObjectId,
    right_id: ObjectId,
    portion_delete_values: list[float],
    rounds: int,
    random_state: int,
    pair_idx: int,
) -> dict[float, float]:
    """Only this function varies the detector scorer."""
    scores = {}
    for portion_delete in portion_delete_values:
        random.seed(random_state + pair_idx * 1009)
        detector.portion_delete = portion_delete
        detector.scorer = Scorer(
            rounds=rounds,
            portion_delete=portion_delete,
            similarity_fn=minmax_similarity,
        )
        pair_scores = detector.get_score(text=[str(left_id), str(right_id)])
        if pair_scores is None or len(pair_scores) == 0:
            raise ValueError("ImpostorDetector returned no score for pair.")
        scores[portion_delete] = float(pair_scores[0])
    return scores


def compute_metrics(
    *,
    y_true: list[int],
    scores_by_portion: dict[float, list[float]],
    n_splits: int,
    n_repeats: int,
    random_state: int,
    ci_level: float,
    n_boot: int,
) -> pd.DataFrame:
    """Compute one metric row per portion_delete."""
    y_true_array = np.asarray(y_true, dtype=int)
    splits = SplitManager(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    ).build_splits(y_true_array)
    metric_computer = PANMetricComputer()

    rows = []
    for portion_delete, scores in scores_by_portion.items():
        cv_result = metric_computer.tune_thresholds(
            y_true=y_true_array,
            scores=np.asarray(scores, dtype=float),
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
            ci_level=ci_level,
            n_boot=n_boot,
            splits=splits,
        )
        rows.append({"portion_delete": portion_delete, **cv_result.metrics_mean})

    return pd.DataFrame(rows).sort_values("portion_delete")


def plot_metrics(
    summary_df: pd.DataFrame,
    dataset_name: str,
    method_name: str,
    save_dir: Path,
) -> None:
    """One plot per dataset, all metrics in the same plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for metric in METRICS:
        if metric in summary_df.columns:
            ax.plot(
                summary_df["portion_delete"],
                summary_df[metric],
                marker="o",
                label=CONFIG.SCORE_TRANSLATIONS.get(metric, metric),
            )

    ax.set_xlabel("Portion of features deleted")
    ax.set_ylabel("Metric score")
    ax.set_ylim(-0.01, 1.01)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(ncol=2)
    ax.set_title(f"{CONFIG.LABEL_TRANSLATIONS.get(method_name, method_name)} Approach on the "
                 f"{CONFIG.DATASET_TRANSLATIONS.get(dataset_name, dataset_name)} Dataset")
    fig.tight_layout()

    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            save_dir / f"{dataset_name}_{method_name}_portion_delete_metrics.{ext}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def run_sweep(
    *,
    datasets: list[str],
    method_name: str,
    portion_delete_values: list[float],
    n_impostors: int,
    min_impostors: int,
    rounds: int,
    top_n: int,
    min_n_tokens: int,
    upsample: bool,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    ci_level: float,
    n_boot: int,
    limit_pairs: int | None,
    save_dir: Path,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    mongo = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))

    all_summary = []

    for dataset_name in datasets:
        logger.info("Running sweep for dataset=%s", dataset_name)
        pair_docs = load_dataset_pairs(mongo, dataset_name, limit_pairs=limit_pairs)
        logger.info("Loaded %d pairs", len(pair_docs))
        detector = ImpostorDetector(
            rounds=rounds,
            top_n=top_n,
            portion_delete=portion_delete_values[0],
            n_impostors=n_impostors,
            impostor_technique=method_name,
            dataset_name=dataset_name,
            min_n_tokens=min_n_tokens,
            upsample=upsample,
        )
        detector.impostor_output_collection = _NoCacheCollection()

        y_true: list[int] = []
        scores_by_portion = {value: [] for value in portion_delete_values}
        score_rows = []
        skipped_rows = []

        for pair_idx, pair_doc in enumerate(pair_docs):
            left_id = pair_doc["left_id"]
            right_id = pair_doc["right_id"]

            try:
                portion_scores = score_pair_for_sweep(
                    detector=detector,
                    left_id=left_id,
                    right_id=right_id,
                    portion_delete_values=portion_delete_values,
                    rounds=rounds,
                    random_state=random_state,
                    pair_idx=pair_idx,
                )
            except Exception as exc:
                logger.exception("Skipping pair %s/%s", left_id, right_id)
                skipped_rows.append(
                    {
                        "left_id": str(left_id),
                        "right_id": str(right_id),
                        "reason": f"failed:{type(exc).__name__}",
                    }
                )
                continue

            y_true.append(pair_doc["same"])
            score_rows.append(
                {
                    "left_id": str(left_id),
                    "right_id": str(right_id),
                    "same": pair_doc["same"],
                    **{f"score_portion_delete_{k:g}": v for k, v in portion_scores.items()},
                }
            )
            for portion_delete, score in portion_scores.items():
                scores_by_portion[portion_delete].append(score)

        if not y_true:
            logger.warning("No usable pairs for dataset=%s", dataset_name)
            continue

        summary_df = compute_metrics(
            y_true=y_true,
            scores_by_portion=scores_by_portion,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
            ci_level=ci_level,
            n_boot=n_boot,
        )
        summary_df.insert(0, "dataset_name", dataset_name)
        summary_df.insert(1, "method_name", method_name)
        summary_df.insert(2, "n_pairs", len(y_true))

        base = f"{dataset_name}_{method_name}_portion_delete"
        summary_df.to_csv(save_dir / f"{base}_metrics.csv", index=False)
        pd.DataFrame(score_rows).to_csv(save_dir / f"{base}_scores.csv", index=False)
        pd.DataFrame(skipped_rows).to_csv(save_dir / f"{base}_skipped_pairs.csv", index=False)

        with open(save_dir / f"{base}_config.json", "w") as f:
            json.dump(
                {
                    "dataset_name": dataset_name,
                    "method_name": method_name,
                    "portion_delete_values": portion_delete_values,
                    "metrics": METRICS,
                    "n_pairs": len(y_true),
                    "n_requested_pairs": len(pair_docs),
                    "n_skipped_pairs": len(skipped_rows),
                    "n_impostors": n_impostors,
                    "min_impostors": min_impostors,
                    "rounds": rounds,
                    "top_n": top_n,
                    "min_n_tokens": min_n_tokens,
                    "upsample": upsample,
                    "random_state": random_state,
                },
                f,
                indent=2,
            )

        plot_metrics(summary_df, dataset_name, method_name, save_dir)
        all_summary.append(summary_df)

    if all_summary:
        pd.concat(all_summary, ignore_index=True).to_csv(
            save_dir / f"{method_name}_portion_delete_metrics_all_datasets.csv",
            index=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple in-domain impostor portion_delete sweep.")
    parser.add_argument("--datasets", nargs="+", default=[CONFIG.BLOG, CONFIG.STUDENT_ESSAYS,])
    parser.add_argument("--method-name", choices=[METHOD_NAME], default=METHOD_NAME)
    parser.add_argument("--portion-delete-values", nargs="+", type=float, default=DEFAULT_PORTION_DELETE_VALUES)
    parser.add_argument("--n-impostors", type=int, default=50)
    parser.add_argument("--min-impostors", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=100000)
    parser.add_argument("--min-n-tokens", type=int, default=500)
    parser.add_argument("--no-upsampling", action="store_true")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--save-dir", type=Path, default=default_save_dir())
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    run_sweep(
        datasets=list(args.datasets),
        method_name=args.method_name,
        portion_delete_values=validate_portion_delete_values(args.portion_delete_values),
        n_impostors=args.n_impostors,
        min_impostors=args.min_impostors,
        rounds=args.rounds,
        top_n=args.top_n,
        min_n_tokens=args.min_n_tokens,
        upsample=not args.no_upsampling,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        ci_level=args.ci_level,
        n_boot=args.n_boot,
        limit_pairs=args.limit_pairs,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
