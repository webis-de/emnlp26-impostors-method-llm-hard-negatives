from __future__ import annotations

import logging

"""Pairwise significance testing for PAN metrics."""

from dataclasses import asdict
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon

from genai_detection.experiments.corrected_ttest import repkfold_ttest

logger = logging.getLogger(__name__)

from genai_detection.experiments.reproduction.pan_metrics.pan_cv import PANEvaluator, SplitManager
from genai_detection.experiments.reproduction.pan_metrics.pan_metric_computation import EvaluationCVResult
from genai_detection.experiments.reproduction.pan_metrics.pan_storage import PANMetricsRecord, PANMetricsStore
from genai_detection.experiments.reproduction.pan_metrics.pan_data_loader import PANDataLoader


def _filter_finite_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _pairwise_p_value(
    a: np.ndarray,
    b: np.ndarray,
    test: str,
    alternative: str,
) -> float:
    if test == "wilcoxon":
        stat = wilcoxon(a, b, alternative=alternative, zero_method="pratt")
        return float(stat.pvalue)
    if test == "ttest_rel":
        stat = ttest_rel(a, b, alternative=alternative)
        return float(stat.pvalue)
    if test == "mannwhitney":
        stat = mannwhitneyu(a, b, alternative=alternative)
        return float(stat.pvalue)
    if test == "ttest_ind":
        stat = ttest_ind(a, b, alternative=alternative)
        return float(stat.pvalue)
    raise ValueError(f"Unknown test '{test}'.")


def _pairwise_repkfold_p_value(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    n_samples: int,
) -> float:

    total = n_splits * n_repeats
    if len(a) != total or len(b) != total:
        raise ValueError(
            f"Expected {total} per-fold values (n_splits={n_splits}, n_repeats={n_repeats}), "
            f"got {len(a)} and {len(b)}."
        )

    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")

    train_set_size = int(round(n_samples / n_splits))
    test_set_size = int(n_samples - train_set_size)
    if test_set_size <= 0 or train_set_size <= 0:
        raise ValueError(
            f"Invalid test_set_size/train_set_size derived from n_samples={n_samples}, n_splits={n_splits}."
        )

    rows = []
    for idx in range(total):
        k = (idx % n_splits) + 1
        r = (idx // n_splits) + 1
        rows.append({"model": "A", "values": float(a[idx]), "k": k, "r": r})
        rows.append({"model": "B", "values": float(b[idx]), "k": k, "r": r})
    df = pd.DataFrame(rows)

    result = repkfold_ttest(data=df, n1=test_set_size, n2=train_set_size, k=n_splits, r=n_repeats)
    if "p_value" in result.columns:
        return float(result["p_value"].iloc[0])
    if "p.value" in result.columns:
        return float(result["p.value"].iloc[0])
    raise ValueError("correctipy.repkfold_ttest did not return a p-value column.")


def compare_pan_metrics_significance(
    pan_metrics: dict,
    metrics: Iterable[str] | None = None,
    alpha_levels: Sequence[float] = (0.05, 0.01, 0.005),
    test: str = "repkfold_ttest",
    alternative: str = "two-sided",
    min_samples: int = 2,
) -> dict[str, dict]:
    """
    Assess pairwise significance between approaches per metric using per-fold values.
    """
    method_names = sorted(pan_metrics.keys())
    if len(method_names) < 2:
        raise ValueError("Need at least two methods to compare.")

    alpha_sorted = sorted(set(float(a) for a in alpha_levels))
    results: dict[str, dict] = {
        "test": test,
        "alternative": alternative,
        "alpha_levels": alpha_sorted,
        "pairs": {},
    }

    repkfold_meta: dict[str, int] | None = None
    if test == "repkfold_ttest":
        for method_name in method_names:
            method_metrics = pan_metrics[method_name]
            split_config = method_metrics.get("split_config")
            n_samples = method_metrics.get("n_samples")
            if not split_config or n_samples is None:
                raise ValueError(
                    f"Missing 'split_config' or 'n_samples' for method '{method_name}'. "
                )
            current = {
                "n_splits": int(split_config.get("n_splits")),
                "n_repeats": int(split_config.get("n_repeats")),
                "n_samples": int(n_samples),
            }
            if repkfold_meta is None:
                repkfold_meta = current
            elif repkfold_meta != current:
                raise ValueError(
                    f"Split metadata mismatch for method '{method_name}': "
                    f"expected {repkfold_meta}, got {current}."
                )

    for method_a, method_b in combinations(method_names, 2):
        metrics_a = pan_metrics[method_a].get("metric_values", {})
        metrics_b = pan_metrics[method_b].get("metric_values", {})
        if not metrics_a or not metrics_b:
            raise ValueError(
                f"Missing metric_values for methods '{method_a}' or '{method_b}'."
            )

        if metrics is None:
            metric_names = sorted(set(metrics_a.keys()) & set(metrics_b.keys()))
        else:
            metric_names = list(metrics)

        pair_key = f"{method_a} vs {method_b}"
        pair_results: dict[str, dict] = {}

        for metric in metric_names:
            values_a = np.asarray(metrics_a.get(metric, []), dtype=float)
            values_b = np.asarray(metrics_b.get(metric, []), dtype=float)

            if (test != "repkfold_ttest") or (test in {"mannwhitney", "ttest_ind"}):
                values_a, values_b = _filter_finite_pairs(values_a, values_b)
            if len(values_a) != len(values_b) and test in {"wilcoxon", "ttest_rel"}:
                raise ValueError(
                    f"Paired test '{test}' requires equal-length samples for "
                    f"metric '{metric}' ({method_a} vs {method_b})."
                )
            if len(values_a) < min_samples or len(values_b) < min_samples:
                pair_results[metric] = {
                    "p_value": float("nan"),
                    "n": int(min(len(values_a), len(values_b))),
                    "significant": {str(a): False for a in alpha_sorted},
                }
                continue

            try:
                if test == "repkfold_ttest":
                    if alternative != "two-sided":
                        raise ValueError(
                            "repkfold_ttest only supports two-sided tests in this pipeline."
                        )
                    assert repkfold_meta is not None
                    p_value = _pairwise_repkfold_p_value(
                        values_a,
                        values_b,
                        n_splits=repkfold_meta["n_splits"],
                        n_repeats=repkfold_meta["n_repeats"],
                        n_samples=repkfold_meta["n_samples"],
                    )
                else:
                    p_value = _pairwise_p_value(values_a, values_b, test, alternative)
            except ValueError:
                p_value = float("nan")

            pair_results[metric] = {
                "p_value": float(p_value),
                "n": int(min(len(values_a), len(values_b))),
                "significant": {str(a): bool(p_value <= a) for a in alpha_sorted},
            }

        results["pairs"][pair_key] = pair_results

    return results


class PANPairwiseSignificance:
    """
    Compute pairwise significance on aligned method intersections.
    """

    def __init__(
        self,
        data_loader: PANDataLoader,
        evaluator: PANEvaluator,
        store: PANMetricsStore,
        split_manager: SplitManager,
    ) -> None:
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.store = store
        self.split_manager = split_manager

    def compute_pairwise_significance(
        self,
        dataset_name: str,
        methods: Sequence[str],
        *,
        impostor_technique: str | None = None,
        n_impostors: int = 50,
        n_potential_impostors: int | None = None,
        rounds: int = 100,
        ci_level: float = 0.95,
        n_boot: int = 10000,
        test: str = "repkfold_ttest",
        alpha_levels: Sequence[float] = (0.05, 0.01, 0.005),
    ) -> dict[str, dict]:
        results: dict[str, dict] = {}
        methods = list(dict.fromkeys(methods))

        # unordered unique pairs
        for method_a, method_b in combinations(methods, 2):
            aligned = self._align_pairwise(
                dataset_name=dataset_name,
                method_a=method_a,
                method_b=method_b,
                impostor_technique=impostor_technique,
                n_impostors=n_impostors,
                n_potential_impostors=n_potential_impostors,
                rounds=rounds,
            )
            y_true, scores_a, scores_b, n_samples = aligned
            logger.info(f"{dataset_name} dataset: Aligned {method_a} vs {method_b} with {n_samples} samples.")

            record_a = self._get_or_compute_record(
                dataset_name,
                method_a,
                y_true,
                scores_a,
                n_samples,
                ci_level,
                n_boot,
            )
            record_b = self._get_or_compute_record(
                dataset_name,
                method_b,
                y_true,
                scores_b,
                n_samples,
                ci_level,
                n_boot,
            )

            pan_metrics = {
                method_a: record_a,
                method_b: record_b,
            }
            logger.info(f"Loaded PAN metrics for {method_a} vs {method_b}.")
            results[f"{method_a} vs {method_b}"] = compare_pan_metrics_significance(
                pan_metrics,
                test=test,
                alpha_levels=alpha_levels,
            )
            logger.info(f"Computed pairwise {test} significance for {method_a} vs {method_b}.")

        return results

    def _align_pairwise(
        self,
        *,
        dataset_name: str,
        method_a: str,
        method_b: str,
        impostor_technique: str | None,
        n_impostors: int,
        n_potential_impostors: int | None,
        rounds: int,
    ) -> tuple[list[int], list[float], list[float], int]:
        scores_a_by_pair = self.data_loader.load_scores(
            method_a,
            dataset_name,
            impostor_technique=impostor_technique,
            n_impostors=n_impostors,
            n_potential_impostors=n_potential_impostors,
            rounds=rounds,
        )
        scores_b_by_pair = self.data_loader.load_scores(
            method_b,
            dataset_name,
            impostor_technique=impostor_technique,
            n_impostors=n_impostors,
            n_potential_impostors=n_potential_impostors,
            rounds=rounds,
        )

        logger.info(
            "Loaded %d scores for %s and %d for %s (dataset=%s).",
            len(scores_a_by_pair),
            method_a,
            len(scores_b_by_pair),
            method_b,
            dataset_name,
        )
        common_keys = set(scores_a_by_pair.keys()) & set(scores_b_by_pair.keys())
        if not common_keys:
            raise ValueError(f"No overlapping pairs for {method_a} vs {method_b}.")

        gt_by_pair = self.data_loader.load_ground_truth(list(common_keys), dataset_name)
        ordered_keys = sorted(key for key in common_keys if key in gt_by_pair)
        if not ordered_keys:
            raise ValueError(f"Missing ground truth for {method_a} vs {method_b}.")

        y_true = [gt_by_pair[key] for key in ordered_keys]
        scores_a = [scores_a_by_pair[key] for key in ordered_keys]
        scores_b = [scores_b_by_pair[key] for key in ordered_keys]

        logger.info(
            "Aligned %s vs %s with %d shared samples.",
            method_a,
            method_b,
            len(ordered_keys),
        )
        return y_true, scores_a, scores_b, len(ordered_keys)

    def _get_or_compute_record(
        self,
        dataset_name: str,
        method_name: str,
        y_true: list[int],
        scores: list[float],
        n_samples: int,
        ci_level: float,
        n_boot: int,
    ) -> dict:
        stored = self.store.get_record(dataset_name, method_name)
        decision = self.store.validate_n_samples(
            stored.get("n_samples") if stored else None,
            n_samples,
        )

        if stored and decision == "reuse":
            stored = self.store.ensure_metric_values(stored, self.evaluator.extract_metric_values)
            logger.info(
                f"Retrieved PAN metrics for {dataset_name} using {method_name} on {n_samples} samples."
            )
            return stored
        if decision == "error":
            raise ValueError(
                f"Stored n_samples ({stored.get('n_samples')}) is smaller than requested "
                f"({n_samples}) for {method_name}."
            )

        # guarantees identical repeated CV splits for any methods that share the same sample set and ordering.
        # Ordering is set before ("aligned") given that method have the same number of samples.
        # Hence, repeated CV scores are based on the same random split given the same number of samples.
        splits = self.split_manager.build_splits(np.asarray(y_true))
        logger.info(f"{dataset_name} dataset: About to start {self.split_manager.n_repeats} repetitions of"
                    f" {self.split_manager.n_splits}-fold CV splits for {method_name}.")
        cv_result = self.evaluator.compute_cv(
            y_true=np.asarray(y_true),
            scores=np.asarray(scores),
            ci_level=ci_level,
            n_boot=n_boot,
            splits=splits,
        )
        logger.info(
            "Computed PAN metrics for %s (dataset=%s): %s",
            method_name,
            dataset_name,
            cv_result.metrics_mean,
        )

        record = self._record_from_cv(
            dataset_name=dataset_name,
            method_name=method_name,
            n_samples=n_samples,
            cv_result=cv_result,
        )
        self.store.save_record(record)

        record_dict = record.to_mongo_dict()
        record_dict["metric_values"] = cv_result.metric_values
        return record_dict

    @staticmethod
    def _record_from_cv(
        *,
        dataset_name: str,
        method_name: str,
        n_samples: int,
        cv_result: EvaluationCVResult,
    ) -> PANMetricsRecord:
        per_split = [asdict(result) for result in cv_result.per_split]
        return PANMetricsRecord(
            dataset_name=dataset_name,
            method_name=method_name,
            n_samples=n_samples,
            metrics_mean=cv_result.metrics_mean,
            metrics_std=cv_result.metrics_std,
            metrics_ci=cv_result.metrics_ci,
            pan_metrics_per_split=per_split,
            split_config=cv_result.split_config,
        )
