"""
Pairwise significance testing on precomputed MongoDB impostor outputs.

Pipeline overview:
1. Load per-technique stored scores and (optionally corrected) binary predictions.
2. Resolve technique aliases (e.g., ``on_the_fly_*``) to the MongoDB representation.
3. Intersect pairs so each pairwise test uses only instances available for both techniques.
4. Run McNemar tests on binary correctness outcomes (same instances, paired setting).
5. Run DeLong tests for AUROC differences (same instances, paired ROC setting).
6. Apply multiple-testing correction and write reproducible CSV/JSON artifacts.

The implementation is intentionally self-contained so it can be executed as a script
or imported by other experiment runners.

Refer to:
- https://medium.com/statistics-in-machine-learning/comparing-roc-curves-in-machine-learning-model-with-delongs
    -test-a-practical-guide-using-python-e70b5d20abde (12.03.2026) for DeLong's Test
- Hemmerich, W. (2023). StatistikGuru: McNemar-Test. Retrieved from https://statistikguru.de/lexikon/mcnemar-test.html
    (12.03.2026) for McNemar's Test
- https://www.geeksforgeeks.org/python/how-to-perform-mcnemars-test-in-python/ (12.03.2026) for McNemar's Test in Python
- https://github.com/Brritany/MLstatkit?tab=readme-ov-file#bootstrapping-for-confidence-intervals (12.03.2026)
    for paired bootstrap confidence intervals for AUROC (area under the ROC curve),
    AUPRC (area under the precision-recall curve), and F1 score metrics.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from MLstatkit import Bootstrapping, Permutation_test
from MLstatkit.stats import Delong_test
from bson import ObjectId
from statsmodels.stats.contingency_tables import mcnemar

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[3]
    / CONFIG.SAVE_PATH
    / "reproduction"
    / "statistical_tests"
)
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)


PairKey = Tuple[ObjectId, ObjectId]


@dataclass
class TechniqueOutputs:
    """Container holding precomputed outputs for one impostor technique."""

    scores_by_pair: Dict[PairKey, float]
    preds_by_pair: Dict[PairKey, int]


def _iter_batches(items: List, batch_size: int) -> Iterable[List]:
    """Yield fixed-size batches from a list."""

    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _normalize_technique_name(technique: str) -> Tuple[str, Optional[str]]:
    """
    Returns (mongodb_technique, retrieval_index_if_on_the_fly_variant).
    """
    if technique in CONFIG.RETRIEVAL_INDEX_TRANSLATIONS:
        return "on_the_fly", CONFIG.RETRIEVAL_INDEX_TRANSLATIONS[technique]
    return technique, None


def _load_left_ids_for_index(
    mongodb: ParaphraseMongoDB,
    retrieval_index: str,
    batch_size: int,
) -> set[ObjectId]:
    """
    Load anchor text ids that belong to a specific retrieval index.

    This is used as a fallback when legacy `impostor_outputs` rows do not carry
    the `retrieval_index` field directly.
    """

    cursor = mongodb.on_the_fly_collection.find(
        {"index": retrieval_index},
        {"text_id": 1},
        batch_size=batch_size,
    )
    return {doc["text_id"] for doc in cursor if "text_id" in doc}


def load_scores_and_predictions_for_technique(
    mongodb: ParaphraseMongoDB,
    dataset_name: str,
    technique: str,
    n_impostors: int = 50,
    n_potential_impostors: Optional[int] = None,
    batch_size: int = 500,
    prediction_field: str = "corr_pred_over_different_rounds",
) -> TechniqueOutputs:
    """
    Load precomputed scores/predictions for one technique from MongoDB.

    For `on_the_fly_*` variants, rows are filtered down to the matching retrieval index..
    """

    base_technique, retrieval_index = _normalize_technique_name(technique)

    query = {
        "impostor_generation_technique": base_technique,
        "dataset_name": dataset_name,
        "n_impostors": n_impostors,
    }
    if n_potential_impostors is not None:
        query["n_potential_impostors"] = n_potential_impostors

    projection = {
        "left_id": 1,
        "right_id": 1,
        "scores_over_different_rounds": 1,
        prediction_field: 1,
        "retrieval_index": 1,
    }

    allowed_left_ids: Optional[set[ObjectId]] = None
    if retrieval_index is not None:
        allowed_left_ids = _load_left_ids_for_index(
            mongodb=mongodb,
            retrieval_index=retrieval_index,
            batch_size=batch_size,
        )

    cursor = mongodb.impostor_output_collection.find(
        query,
        projection,
        batch_size=batch_size,
    ).sort("_id", 1)

    scores_by_pair: Dict[PairKey, float] = {}
    preds_by_pair: Dict[PairKey, int] = {}

    for doc in cursor:
        # Every statistical test is keyed by pair id; skip malformed rows.
        left_id = doc.get("left_id")
        right_id = doc.get("right_id")
        if not isinstance(left_id, ObjectId) or not isinstance(right_id, ObjectId):
            continue

        if retrieval_index is not None:
            # Prefer direct retrieval_index filtering when available.
            doc_retrieval_index = doc.get("retrieval_index")
            if doc_retrieval_index is not None and doc_retrieval_index != retrieval_index:
                continue
            # Backward-compatible fallback using anchor text ids.
            if doc_retrieval_index is None and allowed_left_ids is not None and left_id not in allowed_left_ids:
                continue

        pair = (left_id, right_id)
        if pair not in scores_by_pair:
            score = doc.get("scores_over_different_rounds")
            if score is None:
                continue
            scores_by_pair[pair] = float(score)

        pred_val = doc.get(prediction_field)
        if (
            pred_val is not None
            and not (isinstance(pred_val, float) and np.isnan(pred_val))
            and pair not in preds_by_pair
        ):
            preds_by_pair[pair] = int(bool(pred_val))

    logger.info(
        "Loaded %d scores and %d predictions for %s (%s)",
        len(scores_by_pair),
        len(preds_by_pair),
        technique,
        dataset_name,
    )
    return TechniqueOutputs(scores_by_pair=scores_by_pair, preds_by_pair=preds_by_pair)


def load_ground_truth_for_pairs(
    mongodb: ParaphraseMongoDB,
    dataset_name: str,
    pairs: Iterable[PairKey],
    batch_size: int = 500,
) -> Dict[PairKey, int]:
    """
    Load binary ground truth for a set of pair ids from `all_pairs_collection`.
    """

    pair_list = list(pairs)
    if not pair_list:
        return {}

    ground_truth: Dict[PairKey, int] = {}
    for batch in _iter_batches(pair_list, batch_size=batch_size):
        or_conditions = [{"left_id": left, "right_id": right} for left, right in batch]
        cursor = mongodb.all_pairs_collection.find(
            {"dataset_name": dataset_name, "$or": or_conditions},
            {"left_id": 1, "right_id": 1, "same": 1},
            batch_size=batch_size,
        )
        for doc in cursor:
            pair = (doc["left_id"], doc["right_id"])
            ground_truth[pair] = int(doc["same"])

    return ground_truth


def _adjust_pvalues(pvals: np.ndarray, method: str) -> np.ndarray:
    """
    Apply a multiple-testing correction to an array of p-values.

    Supported methods: `none`, `bonferroni`, `holm`, `fdr_bh`.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return pvals

    if method == "none":
        return np.clip(pvals, 0.0, 1.0)

    if method == "bonferroni":
        return np.clip(pvals * m, 0.0, 1.0)

    # Work in sorted order and map back to original order at the end.
    order = np.argsort(pvals)
    sorted_p = pvals[order]

    if method == "holm":
        adjusted_sorted = np.empty(m, dtype=float)
        running_max = 0.0
        for i, p in enumerate(sorted_p):
            factor = m - i
            value = factor * p
            running_max = max(running_max, value)
            adjusted_sorted[i] = running_max
        adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    elif method == "fdr_bh":
        adjusted_sorted = np.empty(m, dtype=float)
        for i, p in enumerate(sorted_p, start=1):
            adjusted_sorted[i - 1] = p * m / i

        for i in range(m - 2, -1, -1):
            adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
        adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    else:
        raise ValueError(f"Unsupported correction method: {method}")

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def delong_roc_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Dict[str, float]:
    """
    Two-sided DeLong test for AUROC difference between two paired score vectors.
    If z > 0, model A has higher AUROC; if z < 0, model B has higher AUROC.
    p-value < 0.05 (or any alpha) rejects the null hypothesis (where H_0: no difference between AUCs)‚.
    """

    if len(y_true) != len(scores_a) or len(y_true) != len(scores_b):
        raise ValueError("y_true, scores_a and scores_b must have same length")

    y_true = np.asarray(y_true, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    # Perform DeLong's test
    z_score, p_value = Delong_test(y_true, scores_a, scores_b)
    return {"statistic": z_score, "p_value": p_value}

def mcnemar_test_from_predictions(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> Dict[str, float]:
    """
    McNemar test comparing two paired classifiers on the same instances.

    p-value < 0.05 (or any alpha) rejects the null hypothesis (where H_0: no difference between predictions).
    """

    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)

    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("y_true and predictions must have same length")

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    # Discordant cells in the 2x2 paired correctness table.
    a = int(np.sum(a_correct & b_correct))  # Both correct
    b = int(np.sum(a_correct & ~b_correct))  # A correct, B wrong
    c = int(np.sum(~a_correct & b_correct))  # A wrong, B correct
    d = int(np.sum(~a_correct & ~b_correct))  # Both wrong

    data = [[a, b],
            [c, d]]
    # McNemar's Test with the continuity correction
    statistic, p_value = mcnemar(data, exact=False, correction=False)


    return {
      "statistic": statistic,
        "p_value": p_value
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    """
    Paired bootstrap for AUROC, AUPRC, and F1 score
    """

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have same length")

    results = {}
    # Calculate confidence intervals for AUROC, AUPRC, and F1 score
    for metric_name in ["roc_auc", "pr_auc", "f1"]:
        original_score, confidence_lower, confidence_upper = Bootstrapping(y_true, scores, metric_name)
        print(
            f"{metric_name.upper()} original score: {original_score:.3f}, "
            f"confidence interval: [{confidence_lower:.3f} - {confidence_upper:.3f}]")
        results[f"{metric_name}_original_score"] = original_score
        results[f"{metric_name}_ci_low"] = confidence_lower
        results[f"{metric_name}_ci_high"] = confidence_upper

    return results


def run_permutation_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    The Permutation_test function evaluates whether the observed difference in performance between two models is
    statistically significant.
    It works by randomly shuffling the predictions between the models and recalculating the chosen metric many times
    to generate a null distribution of differences.
    This approach makes no assumptions about the underlying distribution of the data, making it a robust method for
    model comparison.

    Performance measures: 'f1', 'accuracy', 'recall', 'precision', 'roc_auc', 'pr_auc', 'average_precision'

    https://github.com/Brritany/MLstatkit?tab=readme-ov-file#permutation-test-for-statistical-significance
    (12.03.2026) for implementation details.
    """

    y_true = np.asarray(y_true, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if len(y_true) != len(scores_a) or len(y_true) != len(scores_b):
        raise ValueError("y_true, scores_a and scores_b must have same length")

    results = {}
    # Compare models using a permutation test on different score
    for metric_name in ['f1', 'accuracy', 'recall', 'precision', 'roc_auc', 'pr_auc', 'average_precision']:
        metric_a, metric_b, p_value, benchmark, samples_mean, samples_std = Permutation_test(
            y_true, scores_a, scores_b, metric_str=metric_name
        )
        print(
            f"{metric_name.upper()} - Model A: {metric_a:.3f}, Model B: {metric_b:.3f}, p-value: {p_value:.4f}, benchmark: {benchmark:.3f}, samples mean: {samples_mean:.3f}, samples std: {samples_std:.3f}"
        )
        results[f"{metric_name}_model_a"] = metric_a
        results[f"{metric_name}_model_b"] = metric_b
        results[f"{metric_name}_p_value"] = p_value
        results[f"{metric_name}_benchmark"] = benchmark
        results[f"{metric_name}_samples_mean"] = samples_mean
        results[f"{metric_name}_samples_std"] = samples_std
    return results


def run_pairwise_significance_tests_from_mongodb(
    dataset_name: str,
    imp_gen_techniques: List[str],
    *,
    n_impostors: int = 50,
    n_potential_impostors: Optional[int] = None,
    batch_size: int = 500,
    prediction_field: str = "corr_pred_over_different_rounds",
    alpha: float = 0.05,
    correction_method: str = "holm",
) -> Dict[str, pd.DataFrame]:
    """
    Run all pairwise significance tests for the given dataset and techniques.

    Returns three dataframes:
    - `mcnemar`
    - `delong`
    - `bootstrap_diff` (not pairwise)
    """

    if not imp_gen_techniques:
        raise ValueError("imp_gen_techniques must not be empty")

    mongodb = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))

    outputs_by_technique: Dict[str, TechniqueOutputs] = {}
    all_pairs = set()

    for technique in imp_gen_techniques:
        outputs = load_scores_and_predictions_for_technique(
            mongodb=mongodb,
            dataset_name=dataset_name,
            technique=technique,
            n_impostors=n_impostors,
            n_potential_impostors=n_potential_impostors,
            batch_size=batch_size,
            prediction_field=prediction_field,
        )
        outputs_by_technique[technique] = outputs
        all_pairs.update(outputs.scores_by_pair.keys())

    gt_by_pair = load_ground_truth_for_pairs(
        mongodb=mongodb,
        dataset_name=dataset_name,
        pairs=all_pairs,
        batch_size=batch_size,
    )

    logger.info(
        "Loaded ground truth for %d/%d unique scored pairs",
        len(gt_by_pair),
        len(all_pairs),
    )

    mcnemar_rows = []
    delong_rows = []
    bootstrap_rows = []
    permutation_rows = []

    # -----------------------------
    # Bootstrap confidence intervals per technique
    # -----------------------------
    for technique, outputs in outputs_by_technique.items():
        common_score_pairs = set(outputs.scores_by_pair.keys()) & set(gt_by_pair.keys())

        if not common_score_pairs:
            logger.warning(
                "No common pairs with precomputed scores for bootstrap CI: %s (%s)",
                technique,
                dataset_name,
            )
            continue

        ordered_pairs = list(common_score_pairs)
        y_true = np.asarray([gt_by_pair[p] for p in ordered_pairs], dtype=int)
        scores = np.asarray([outputs.scores_by_pair[p] for p in ordered_pairs], dtype=float)

        try:
            ci_results = bootstrap_confidence_intervals(y_true=y_true, scores=scores)
            bootstrap_rows.append(
                {
                    "dataset_name": dataset_name,
                    "technique": technique,
                    "n_common_pairs": len(ordered_pairs),
                    **ci_results,
                }
            )
        except Exception as e:
            logger.exception(
                "Bootstrap CI failed for %s (%s): %s",
                technique,
                dataset_name,
                e,
            )

    for tech_a, tech_b in combinations(imp_gen_techniques, 2):
        out_a = outputs_by_technique[tech_a]
        out_b = outputs_by_technique[tech_b]

        # -----------------------------
        # McNemar (precomputed predictions)
        # -----------------------------
        # Critical requirement: evaluate only pairs available for both techniques.
        common_pred_pairs = (
            set(out_a.preds_by_pair.keys())
            & set(out_b.preds_by_pair.keys())
            & set(gt_by_pair.keys())
        )

        if common_pred_pairs:
            ordered_pred_pairs = list(common_pred_pairs)
            y_true = np.asarray([gt_by_pair[p] for p in ordered_pred_pairs], dtype=int)
            pred_a = np.asarray([out_a.preds_by_pair[p] for p in ordered_pred_pairs], dtype=int)
            pred_b = np.asarray([out_b.preds_by_pair[p] for p in ordered_pred_pairs], dtype=int)

            mc = mcnemar_test_from_predictions(y_true=y_true, pred_a=pred_a, pred_b=pred_b)
            mcnemar_rows.append(
                {
                    "dataset_name": dataset_name,
                    "technique_a": tech_a,
                    "technique_b": tech_b,
                    "n_common_pairs": len(ordered_pred_pairs),
                    "prediction_field": prediction_field,
                    **mc,
                }
            )
        else:
            logger.warning(
                "No common pairs with precomputed predictions for %s vs %s (%s)",
                tech_a,
                tech_b,
                dataset_name,
            )

        # -----------------------------
        # DeLong (AUROC on precomputed scores)
        # -----------------------------
        # Same intersection principle for score-based AUROC testing.
        common_score_pairs = (
            set(out_a.scores_by_pair.keys())
            & set(out_b.scores_by_pair.keys())
            & set(gt_by_pair.keys())
        )

        if common_score_pairs:
            ordered_score_pairs = list(common_score_pairs)
            y_true = np.asarray([gt_by_pair[p] for p in ordered_score_pairs], dtype=int)
            scores_a = np.asarray([out_a.scores_by_pair[p] for p in ordered_score_pairs], dtype=float)
            scores_b = np.asarray([out_b.scores_by_pair[p] for p in ordered_score_pairs], dtype=float)

            de = delong_roc_test(y_true=y_true, scores_a=scores_a, scores_b=scores_b)
            delong_rows.append(
                {
                    "dataset_name": dataset_name,
                    "technique_a": tech_a,
                    "technique_b": tech_b,
                    "n_common_pairs": len(ordered_score_pairs),
                    **de,
                }
            )

            # -----------------------------
            # Permutation test (pairwise score comparison)
            # -----------------------------
            try:
                perm = run_permutation_test(
                    y_true=y_true,
                    scores_a=scores_a,
                    scores_b=scores_b
                )
                permutation_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "technique_a": tech_a,
                        "technique_b": tech_b,
                        "n_common_pairs": len(ordered_score_pairs),
                        **perm,
                    }
                )
            except Exception as e:
                logger.exception(
                    "Permutation test failed for %s vs %s (%s): %s",
                    tech_a,
                    tech_b,
                    dataset_name,
                    e,
                )
        else:
            logger.warning(
                "No common pairs with precomputed scores for %s vs %s (%s)",
                tech_a,
                tech_b,
                dataset_name,
            )

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    delong_df = pd.DataFrame(delong_rows)
    bootstrap_ci_df = pd.DataFrame(bootstrap_rows)
    permutation_df = pd.DataFrame(permutation_rows)

    # Correct p-values family-wise across all pairwise comparisons per test family.
    if not mcnemar_df.empty:
        pvals = mcnemar_df["p_exact"].to_numpy(dtype=float)
        adj = _adjust_pvalues(pvals, method=correction_method)
        mcnemar_df["p_exact_adj"] = adj
        mcnemar_df["reject_h0"] = mcnemar_df["p_exact_adj"] <= alpha
        mcnemar_df["alpha"] = alpha
        mcnemar_df["correction_method"] = correction_method

    if not delong_df.empty:
        pvals = delong_df["p_value"].to_numpy(dtype=float)
        valid_mask = ~np.isnan(pvals)
        adj = np.full_like(pvals, fill_value=np.nan, dtype=float)
        if np.any(valid_mask):
            adj[valid_mask] = _adjust_pvalues(pvals[valid_mask], method=correction_method)
        delong_df["p_value_adj"] = adj
        delong_df["reject_h0"] = delong_df["p_value_adj"] <= alpha
        delong_df["alpha"] = alpha
        delong_df["correction_method"] = correction_method

    if not permutation_df.empty:
        pval_cols = [c for c in permutation_df.columns if c.endswith("_p_value")]

        for col in pval_cols:
            pvals = permutation_df[col].to_numpy(dtype=float)
            valid_mask = ~np.isnan(pvals)
            adj = np.full_like(pvals, fill_value=np.nan, dtype=float)

            if np.any(valid_mask):
                adj[valid_mask] = _adjust_pvalues(pvals[valid_mask], method=correction_method)

            adj_col = f"{col}_adj"
            reject_col = f"{col}_reject_h0"

            permutation_df[adj_col] = adj
            permutation_df[reject_col] = permutation_df[adj_col] <= alpha

        permutation_df["alpha"] = alpha
        permutation_df["correction_method"] = correction_method

    return {
        "mcnemar": mcnemar_df,
        "delong": delong_df,
        "bootstrap_ci": bootstrap_ci_df,
        "permutation": permutation_df,
    }


def save_pairwise_significance_results(
    results: Dict[str, pd.DataFrame],
    save_dir: Path,
    dataset_name: str,
    metadata: Optional[Dict] = None,
) -> Dict[str, Path]:
    """Persist result tables and optional metadata to disk."""

    save_dir.mkdir(parents=True, exist_ok=True)
    out_paths: Dict[str, Path] = {}

    for key, df in results.items():
        out_path = save_dir / f"{key}_{dataset_name}.csv"
        df.to_csv(out_path, index=False)
        out_paths[key] = out_path

    if metadata is not None:
        meta_path = save_dir / f"metadata_{dataset_name}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        out_paths["metadata"] = meta_path

    return out_paths


def _build_default_technique_list() -> List[str]:
    """Default technique set matching the main reproduction comparison."""

    return [
        "on_the_fly_chatnoir",
        # "on_the_fly_serpapi",
        "on_the_fly_startpage",
        "in_domain",
        # "one_step_llm",
        "two_step_llm",
    ]


def main() -> None:
    """CLI entry point for dataset-wide pairwise significance testing."""

    parser = argparse.ArgumentParser(
        description="Pairwise significance tests on precomputed impostor outputs from MongoDB.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=[CONFIG.BLOG, CONFIG.STUDENT_ESSAYS, "both"],
        default="both",
        help="Dataset to analyze.",
    )
    parser.add_argument(
        "--imp_gen_techniques",
        nargs="+",
        default=_build_default_technique_list(),
        help="Impostor generation techniques to compare pairwise.",
    )
    parser.add_argument("--n_impostors", type=int, default=50)
    parser.add_argument("--n_potential_impostors", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument(
        "--prediction_field",
        type=str,
        default="corr_pred_over_different_rounds",
        help="Field in impostor_outputs used as precomputed binary decision for McNemar.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--correction_method",
        type=str,
        choices=["none", "bonferroni", "holm", "fdr_bh"],
        default="holm",
    )
    parser.add_argument(
        "--include_bootstrap_auc_diff",
        action="store_true",
        help="Compute paired bootstrap confidence intervals for AUC differences.",
    )
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(LOCAL_SAVE_PATH),
        help="Output directory for CSV/JSON artifacts.",
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

    for dataset in datasets:
        logger.info("Running pairwise significance tests for dataset=%s", dataset)

        results = run_pairwise_significance_tests_from_mongodb(
            dataset_name=dataset,
            imp_gen_techniques=args.imp_gen_techniques,
            n_impostors=args.n_impostors,
            n_potential_impostors=args.n_potential_impostors,
            batch_size=args.batch_size,
            prediction_field=args.prediction_field,
            alpha=args.alpha,
            correction_method=args.correction_method,
        )

        metadata = {
            "dataset_name": dataset,
            "imp_gen_techniques": args.imp_gen_techniques,
            "n_impostors": args.n_impostors,
            "n_potential_impostors": args.n_potential_impostors,
            "batch_size": args.batch_size,
            "prediction_field": args.prediction_field,
            "alpha": args.alpha,
            "correction_method": args.correction_method,
            "include_bootstrap_auc_diff": args.include_bootstrap_auc_diff,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
        }
        saved = save_pairwise_significance_results(
            results=results,
            save_dir=save_dir,
            dataset_name=dataset,
            metadata=metadata,
        )
        logger.info("Saved artifacts for %s: %s", dataset, saved)


if __name__ == "__main__":
    """
    McNemar: pairwise, binary predictions
    DeLong: pairwise, AUROC comparison
    Bootstrap CI: single-model uncertainty estimate
    Permutation test: pairwise, broader metric comparison
    """
    main()
