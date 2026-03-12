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
from bson import ObjectId
from scipy.stats import binomtest, chi2, norm
from sklearn.metrics import roc_auc_score

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


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """
    Compute midranks used by DeLong's AUROC covariance estimator.
    """

    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(
    predictions_sorted_transposed: np.ndarray,
    label_1_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast DeLong implementation for correlated ROC AUC estimates.

    Returns:
    - vector of AUCs (one per classifier)
    - covariance matrix of AUC estimates
    """

    # number of positive examples m, negative examples n, and classifiers k
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    # Initialize arrays for midrank computations
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    # Calculate AUCs
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    # Compute variance components
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    # Compute covariance matrices
    sx = np.cov(v01)
    sy = np.cov(v10)

    if np.ndim(sx) == 0:
        sx = np.asarray([[float(sx)]])
    if np.ndim(sy) == 0:
        sy = np.asarray([[float(sy)]])

    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Dict[str, float]:
    """
    Two-sided DeLong test for AUROC difference between two paired score vectors.
    """

    if len(y_true) != len(scores_a) or len(y_true) != len(scores_b):
        raise ValueError("y_true, scores_a and scores_b must have same length")

    y_true = np.asarray(y_true, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if len(np.unique(y_true)) < 2:
        return {
            "auc_a": np.nan,
            "auc_b": np.nan,
            "delta_auc": np.nan,
            "z": np.nan,
            "p_value": np.nan,
            "var_delta": np.nan,
        }

    # DeLong expects positives first.
    order = np.argsort(-y_true)
    label_1_count = int(np.sum(y_true))
    predictions = np.vstack([scores_a, scores_b])[:, order]

    aucs, cov = _fast_delong(predictions_sorted_transposed=predictions, label_1_count=label_1_count)

    if cov.shape != (2, 2):
        return {
            "auc_a": float(aucs[0]),
            "auc_b": float(aucs[1]),
            "delta_auc": float(aucs[0] - aucs[1]),
            "z": np.nan,
            "p_value": np.nan,
            "var_delta": np.nan,
        }

    # Calculating z-score and p-value
    var_delta = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    delta_auc = float(aucs[0] - aucs[1])

    if var_delta <= 0:
        z_value = 0.0 if abs(delta_auc) < 1e-15 else np.nan
        p_value = 1.0 if abs(delta_auc) < 1e-15 else np.nan
    else:
        z_value = float(abs(delta_auc) / np.sqrt(var_delta))
        p_value = float(2.0 * norm.sf(z_value))

    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta_auc": delta_auc,
        "z": z_value,
        "p_value": p_value,
        "var_delta": var_delta,
    }


def mcnemar_test_from_predictions(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> Dict[str, float]:
    """
    McNemar test comparing two paired classifiers on the same instances.

    Returns both exact (binomial) and chi-square approximations.
    """

    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)

    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("y_true and predictions must have same length")

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    # Discordant cells in the 2x2 paired correctness table.
    b = int(np.sum(a_correct & ~b_correct))  # A correct, B wrong
    c = int(np.sum(~a_correct & b_correct))  # A wrong, B correct
    discordant = b + c

    if discordant == 0:
        exact_p = 1.0
        chi2_cc = 0.0
        chi2_no_cc = 0.0
        p_chi2_cc = 1.0
        p_chi2_no_cc = 1.0
    else:
        exact_p = float(binomtest(k=min(b, c), n=discordant, p=0.5, alternative="two-sided").pvalue)
        chi2_cc = float((abs(b - c) - 1.0) ** 2 / discordant)
        chi2_no_cc = float((b - c) ** 2 / discordant)
        p_chi2_cc = float(chi2.sf(chi2_cc, df=1))
        p_chi2_no_cc = float(chi2.sf(chi2_no_cc, df=1))

    acc_a = float(np.mean(a_correct))
    acc_b = float(np.mean(b_correct))

    return {
        "n": int(len(y_true)),
        "discordant": discordant,
        "b_a_correct_b_wrong": b,
        "c_a_wrong_b_correct": c,
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "delta_accuracy": acc_a - acc_b,
        "chi2_cc": chi2_cc,
        "p_chi2_cc": p_chi2_cc,
        "chi2_no_cc": chi2_no_cc,
        "p_chi2_no_cc": p_chi2_no_cc,
        "p_exact": exact_p,
    }


def bootstrap_auc_difference(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap for AUROC difference (AUC_A - AUC_B).

    This complements DeLong by providing an effect-size confidence interval.
    """

    y_true = np.asarray(y_true, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    n = len(y_true)
    if n == 0:
        return {
            "delta_auc_mean": np.nan,
            "delta_auc_ci_low": np.nan,
            "delta_auc_ci_high": np.nan,
            "bootstrap_p_value": np.nan,
            "n_bootstrap_used": 0,
        }

    rng = np.random.default_rng(seed)
    deltas: List[float] = []

    for _ in range(n_bootstrap):
        # Paired resampling preserves correlation between both systems.
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_a = roc_auc_score(y_b, scores_a[idx])
        auc_b = roc_auc_score(y_b, scores_b[idx])
        deltas.append(float(auc_a - auc_b))

    if not deltas:
        return {
            "delta_auc_mean": np.nan,
            "delta_auc_ci_low": np.nan,
            "delta_auc_ci_high": np.nan,
            "bootstrap_p_value": np.nan,
            "n_bootstrap_used": 0,
        }

    deltas_np = np.asarray(deltas, dtype=float)
    ci_low, ci_high = np.percentile(deltas_np, [2.5, 97.5])

    p_left = float(np.mean(deltas_np <= 0.0))
    p_right = float(np.mean(deltas_np >= 0.0))
    p_two_sided = float(min(1.0, 2.0 * min(p_left, p_right)))

    return {
        "delta_auc_mean": float(np.mean(deltas_np)),
        "delta_auc_ci_low": float(ci_low),
        "delta_auc_ci_high": float(ci_high),
        "bootstrap_p_value": p_two_sided,
        "n_bootstrap_used": int(len(deltas_np)),
    }


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
    include_bootstrap_auc_diff: bool = True,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Run all pairwise significance tests for the given dataset and techniques.

    Returns three dataframes:
    - `mcnemar`
    - `delong`
    - `bootstrap_auc_diff`
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

            if include_bootstrap_auc_diff:
                bs = bootstrap_auc_difference(
                    y_true=y_true,
                    scores_a=scores_a,
                    scores_b=scores_b,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed,
                )
                bootstrap_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "technique_a": tech_a,
                        "technique_b": tech_b,
                        "n_common_pairs": len(ordered_score_pairs),
                        **bs,
                    }
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
    bootstrap_df = pd.DataFrame(bootstrap_rows)

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

    return {
        "mcnemar": mcnemar_df,
        "delong": delong_df,
        "bootstrap_auc_diff": bootstrap_df,
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
            include_bootstrap_auc_diff=args.include_bootstrap_auc_diff,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
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
    main()
