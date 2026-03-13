"""
Run pairwise significance tests on impostor ablation outputs stored in MongoDB.

Workflow:
1. Load ablation scores from `impostor_ablation_output_collection` (no `n_impostors` filtering).
2. Normalize scores by the known number of rounds for each ablation variant.
3. Build binary predictions (use stored prediction field when available, else threshold normalized scores).
4. Intersect pair ids per ablation pair so each test uses the same instances.
5. Run McNemar (paired binary predictions) and DeLong (paired AUROC) tests.
6. Apply multiple-testing correction and save CSV artifacts.
"""

import argparse
import json
import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bson import ObjectId

from genai_detection.config import CONFIG
from genai_detection.experiments.reproduction.pairwise_significance_from_mongodb import (
    TechniqueOutputs,
    _adjust_pvalues,
    delong_roc_test,
    load_ground_truth_for_pairs,
    mcnemar_test_from_predictions,
)
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

PairKey = Tuple[ObjectId, ObjectId]

# Keep labels aligned with genai_detection/experiments/prec_recall_curves_ablations.py
ABLATION_CLASS_NAMES = {
    "bdi": "BDIImpostorDetector",
    "homotopy": "HBCImpostorDetector",
    "potha2017": "Potha2017ImpostorDetector",
    "asgalf": "ASGALFImpostorDetector",
    "std_impostor": "StdImpostor",
}

# Matches ABLATION_ARGS[variant]["rounds"] from the ablation experiment file.
ABLATION_ROUNDS = {
    "bdi": 100,
    "homotopy": 50,
    "potha2017": 10,
    "asgalf": 50,
    "std_impostor": 50,
}

LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[2]
    / CONFIG.SAVE_PATH
    / "ablations"
    / "statistical_tests"
)
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)


def load_ablation_scores_and_predictions(
    mongodb: ParaphraseMongoDB,
    dataset_name: str,
    ablation_name: str,
    *,
    impostor_technique: Optional[str],
    prediction_field: str,
    decision_threshold: float,
    batch_size: int,
) -> TechniqueOutputs:
    """
    Load ablation outputs for one variant and derive comparable scores/predictions.

    Notes:
    - No `n_impostors` or `n_potential_impostors` constraints are used.
    - If `prediction_field` is missing, predictions are computed from normalized scores.
    """
    if ablation_name not in ABLATION_CLASS_NAMES:
        raise ValueError(
            f"Unsupported ablation '{ablation_name}'. Choices: {sorted(ABLATION_CLASS_NAMES)}"
        )

    ablation_class = ABLATION_CLASS_NAMES[ablation_name]
    rounds = ABLATION_ROUNDS[ablation_name]

    query = {
        "dataset_name": dataset_name,
        "ablation": ablation_class,
    }
    if impostor_technique is not None:
        query["impostor_generation_technique"] = impostor_technique

    projection = {
        "left_id": 1,
        "right_id": 1,
        "scores_over_different_rounds": 1,
        prediction_field: 1,
    }

    cursor = mongodb.impostor_ablation_output_collection.find(
        query,
        projection,
        batch_size=batch_size,
    ).sort("_id", 1)

    scores_by_pair: Dict[PairKey, float] = {}
    preds_by_pair: Dict[PairKey, int] = {}

    for doc in cursor:
        left_id = doc.get("left_id")
        right_id = doc.get("right_id")
        if not isinstance(left_id, ObjectId) or not isinstance(right_id, ObjectId):
            continue

        pair = (left_id, right_id)
        raw_score = doc.get("scores_over_different_rounds")
        if raw_score is None:
            continue

        # Normalize to [0,1]-like scale across ablations with different rounds.
        norm_score = float(raw_score) / float(rounds)

        if pair not in scores_by_pair:
            scores_by_pair[pair] = norm_score

        pred_val = doc.get(prediction_field)
        if (
            pred_val is not None
            and not (isinstance(pred_val, float) and np.isnan(pred_val))
            and pair not in preds_by_pair
        ):
            preds_by_pair[pair] = int(bool(pred_val))
        elif pair not in preds_by_pair:
            preds_by_pair[pair] = int(norm_score >= decision_threshold)

    logger.info(
        "Loaded %d normalized scores and %d predictions for ablation=%s (%s)",
        len(scores_by_pair),
        len(preds_by_pair),
        ablation_name,
        dataset_name,
    )

    return TechniqueOutputs(scores_by_pair=scores_by_pair, preds_by_pair=preds_by_pair)


def run_pairwise_significance_tests_ablations_from_mongodb(
    dataset_name: str,
    ablation_names: List[str],
    *,
    impostor_technique: Optional[str] = "in_domain",
    prediction_field: str = "corr_pred_over_different_rounds",
    decision_threshold: float = 0.5,
    alpha: float = 0.05,
    correction_method: str = "holm",
    batch_size: int = 500,
) -> Dict[str, pd.DataFrame]:
    """Run McNemar and DeLong pairwise tests across selected ablations."""
    if not ablation_names:
        raise ValueError("ablation_names must not be empty")

    unknown = [name for name in ablation_names if name not in ABLATION_CLASS_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown ablations {unknown}. Choices: {sorted(ABLATION_CLASS_NAMES)}"
        )

    mongodb = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))

    outputs_by_ablation: Dict[str, TechniqueOutputs] = {}
    all_pairs: set[PairKey] = set()

    for ablation_name in ablation_names:
        outputs = load_ablation_scores_and_predictions(
            mongodb=mongodb,
            dataset_name=dataset_name,
            ablation_name=ablation_name,
            impostor_technique=impostor_technique,
            prediction_field=prediction_field,
            decision_threshold=decision_threshold,
            batch_size=batch_size,
        )
        outputs_by_ablation[ablation_name] = outputs
        all_pairs.update(outputs.scores_by_pair.keys())

    gt_by_pair = load_ground_truth_for_pairs(
        mongodb=mongodb,
        dataset_name=dataset_name,
        pairs=all_pairs,
        batch_size=batch_size,
    )

    logger.info(
        "Loaded ground truth for %d/%d unique ablation pairs",
        len(gt_by_pair),
        len(all_pairs),
    )

    mcnemar_rows = []
    delong_rows = []

    for ablation_a, ablation_b in combinations(ablation_names, 2):
        out_a = outputs_by_ablation[ablation_a]
        out_b = outputs_by_ablation[ablation_b]

        common_pred_pairs = (
            set(out_a.preds_by_pair.keys())
            & set(out_b.preds_by_pair.keys())
            & set(gt_by_pair.keys())
        )
        if common_pred_pairs:
            ordered_pairs = list(common_pred_pairs)
            y_true = np.asarray([gt_by_pair[p] for p in ordered_pairs], dtype=int)
            pred_a = np.asarray([out_a.preds_by_pair[p] for p in ordered_pairs], dtype=int)
            pred_b = np.asarray([out_b.preds_by_pair[p] for p in ordered_pairs], dtype=int)

            mc = mcnemar_test_from_predictions(y_true=y_true, pred_a=pred_a, pred_b=pred_b)
            mcnemar_rows.append(
                {
                    "dataset_name": dataset_name,
                    "technique_a": ablation_a,
                    "technique_b": ablation_b,
                    "n_common_pairs": len(ordered_pairs),
                    "prediction_field": prediction_field,
                    "decision_threshold": decision_threshold,
                    **mc,
                }
            )
        else:
            logger.warning(
                "No common prediction pairs for %s vs %s (%s)",
                ablation_a,
                ablation_b,
                dataset_name,
            )

        common_score_pairs = (
            set(out_a.scores_by_pair.keys())
            & set(out_b.scores_by_pair.keys())
            & set(gt_by_pair.keys())
        )
        if common_score_pairs:
            ordered_pairs = list(common_score_pairs)
            y_true = np.asarray([gt_by_pair[p] for p in ordered_pairs], dtype=int)
            scores_a = np.asarray([out_a.scores_by_pair[p] for p in ordered_pairs], dtype=float)
            scores_b = np.asarray([out_b.scores_by_pair[p] for p in ordered_pairs], dtype=float)

            de = delong_roc_test(y_true=y_true, scores_a=scores_a, scores_b=scores_b)
            delong_rows.append(
                {
                    "dataset_name": dataset_name,
                    "technique_a": ablation_a,
                    "technique_b": ablation_b,
                    "n_common_pairs": len(ordered_pairs),
                    **de,
                }
            )
        else:
            logger.warning(
                "No common score pairs for %s vs %s (%s)",
                ablation_a,
                ablation_b,
                dataset_name,
            )

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    delong_df = pd.DataFrame(delong_rows)

    if not mcnemar_df.empty and "p_value" in mcnemar_df.columns:
        pvals = mcnemar_df["p_value"].to_numpy(dtype=float)
        p_adj = _adjust_pvalues(pvals, method=correction_method)
        mcnemar_df["p_value_adj"] = p_adj
        mcnemar_df["reject_h0"] = mcnemar_df["p_value_adj"] <= alpha
        mcnemar_df["alpha"] = alpha
        mcnemar_df["correction_method"] = correction_method

    if not delong_df.empty and "p_value" in delong_df.columns:
        pvals = delong_df["p_value"].to_numpy(dtype=float)
        valid_mask = ~np.isnan(pvals)
        p_adj = np.full_like(pvals, fill_value=np.nan, dtype=float)
        if np.any(valid_mask):
            p_adj[valid_mask] = _adjust_pvalues(pvals[valid_mask], method=correction_method)
        delong_df["p_value_adj"] = p_adj
        delong_df["reject_h0"] = delong_df["p_value_adj"] <= alpha
        delong_df["alpha"] = alpha
        delong_df["correction_method"] = correction_method

    return {
        "mcnemar": mcnemar_df,
        "delong": delong_df,
    }


def save_results(
    results: Dict[str, pd.DataFrame],
    save_dir: Path,
    dataset_name: str,
    metadata: Dict,
) -> Dict[str, Path]:
    """Save output tables and metadata."""
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    for name, df in results.items():
        out_path = save_dir / f"{name}_ablations_{dataset_name}.csv"
        df.to_csv(out_path, index=False)
        paths[name] = out_path

    meta_path = save_dir / f"metadata_ablations_{dataset_name}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    paths["metadata"] = meta_path

    return paths


def _default_ablations() -> List[str]:
    return ["bdi", "homotopy", "potha2017", "asgalf", "std_impostor"]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pairwise McNemar/DeLong significance tests for ablation outputs in MongoDB.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=[CONFIG.BLOG, CONFIG.STUDENT_ESSAYS, "both"],
        default="both",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=_default_ablations(),
        help=f"Ablations to compare. Choices: {sorted(ABLATION_CLASS_NAMES)}",
    )
    parser.add_argument(
        "--impostor_technique",
        type=str,
        default="in_domain",
        help="Optional filter on impostor_generation_technique in ablation outputs.",
    )
    parser.add_argument(
        "--prediction_field",
        type=str,
        default="corr_pred_over_different_rounds",
        help="If missing per row, predictions are derived via score threshold.",
    )
    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=0.5,
        help="Threshold used when prediction_field is unavailable.",
    )
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--correction_method",
        type=str,
        choices=["none", "bonferroni", "holm", "fdr_bh"],
        default="bonferroni",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(LOCAL_SAVE_PATH),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    datasets = [args.dataset_name]
    if args.dataset_name == "both":
        datasets = [CONFIG.BLOG, CONFIG.STUDENT_ESSAYS]

    save_dir = Path(args.save_dir)

    for dataset_name in datasets:
        logger.info("Running ablation significance tests for dataset=%s", dataset_name)
        results = run_pairwise_significance_tests_ablations_from_mongodb(
            dataset_name=dataset_name,
            ablation_names=args.ablations,
            impostor_technique=args.impostor_technique,
            prediction_field=args.prediction_field,
            decision_threshold=args.decision_threshold,
            alpha=args.alpha,
            correction_method=args.correction_method,
            batch_size=args.batch_size,
        )

        metadata = {
            "dataset_name": dataset_name,
            "ablations": args.ablations,
            "impostor_technique": args.impostor_technique,
            "prediction_field": args.prediction_field,
            "decision_threshold": args.decision_threshold,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "correction_method": args.correction_method,
            "note": "No n_impostors/n_potential_impostors filtering is applied.",
        }

        saved_paths = save_results(
            results=results,
            save_dir=save_dir,
            dataset_name=dataset_name,
            metadata=metadata,
        )
        logger.info("Saved ablation significance artifacts for %s: %s", dataset_name, saved_paths)


if __name__ == "__main__":
    main()
