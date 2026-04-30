from __future__ import annotations

import logging

"""Pairwise significance testing for PAN metrics."""

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence
# Add effect size computation via Cohen's d_z
# https://pingouin-stats.org/generated/pingouin.compute_effsize.html#pingouin.compute_effsize (13.04.2026)

import pingouin as pg

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon

from genai_detection.experiments.corrected_ttest import repkfold_ttest


logger = logging.getLogger(__name__)

from genai_detection.experiments.reproduction.pan_metrics.pan_cv import PANEvaluator, SplitManager
from genai_detection.experiments.reproduction.pan_metrics.pan_metric_computation import EvaluationCVResult
from genai_detection.experiments.reproduction.pan_metrics.pan_storage import PANMetricsRecord, PANMetricsStore
from genai_detection.experiments.reproduction.pan_metrics.pan_data_loader import PANDataLoader
from genai_detection.experiments.reproduction.pan_metrics.pan_metric_computation import PANMetricComputer

@dataclass(frozen=True)
class PANPairwiseMetricSignificanceResult:
    """
    Pairwise significance outcome for a single metric (A vs B).
    """

    p_value: float
    p_value_bonf: float
    n_samples: int
    significant: dict[str, bool]
    effect_size: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_value": float(self.p_value),
            "p_value_bonf": float(self.p_value_bonf),
            "n_samples": int(self.n_samples),
            "significant": dict(self.significant),
            "effect_size": float(self.effect_size),
        }

    @classmethod
    def nan_result(
        cls,
        *,
        n_samples: int,
        alpha_levels: Sequence[float],
    ) -> "PANPairwiseMetricSignificanceResult":
        alpha_sorted = sorted(set(float(a) for a in alpha_levels))
        return cls(
            p_value=float("nan"),
            p_value_bonf=float("nan"),
            n_samples=int(n_samples),
            significant={str(a): False for a in alpha_sorted},
            effect_size=float("nan"),
        )


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

    test_set_size = int(round(n_samples / n_splits))
    train_set_size = int(n_samples - test_set_size)
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

    result = repkfold_ttest(data=df, train_set_size=train_set_size, test_set_size=test_set_size, k=n_splits, r=n_repeats)

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
    bonf_correction_factor:float=1.0
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
        "metrics": {},
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

        pair_results: dict[str, dict] = {}

        for metric in metric_names:
            values_a = np.asarray(metrics_a.get(metric, []), dtype=float)
            values_b = np.asarray(metrics_b.get(metric, []), dtype=float)

            if test in {"wilcoxon", "ttest_rel"}:
                # paired tests which do not exactly k (# folds) x r (# repetitions) observations
                values_a, values_b = _filter_finite_pairs(values_a, values_b)
                if len(values_a) != len(values_b):
                    raise ValueError(
                        f"Paired test '{test}' requires equal-length samples for "
                        f"metric '{metric}' ({method_a} vs {method_b})."
                    )
            n_pair_samples = int(min(len(values_a), len(values_b)))
            if len(values_a) < min_samples or len(values_b) < min_samples:
                pair_results[metric] = PANPairwiseMetricSignificanceResult.nan_result(
                    n_samples=n_pair_samples,
                    alpha_levels=alpha_sorted,
                ).to_dict()
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

            # bonferroni correction
            p_value_bonf = min(p_value * bonf_correction_factor, 1.0)

            # Match effect size to design
            if test in {"repkfold_ttest", "wilcoxon", "ttest_rel"}:
                # Cohen's d_z for paired samples (also: one-sample)
                effect_size = float(
                    pg.compute_effsize(
                        values_a, values_b, paired=True, eftype="cohen_dz"
                    )
                )
            elif test in {"mannwhitney", "ttest_ind"}:
                # Cohen's d for independent samples
                effect_size = float(
                    pg.compute_effsize(values_a, values_b, paired=False, eftype="cohen")
                )
            else:
                effect_size = float("nan")

            pair_results[metric] = PANPairwiseMetricSignificanceResult(
                p_value=float(p_value),
                p_value_bonf=float(p_value_bonf),
                n_samples=n_pair_samples,   # one value per CV split
                significant={str(a): bool(p_value_bonf <= a) for a in alpha_sorted},
                effect_size=float(effect_size),
            ).to_dict()

        results["metrics"] = pair_results

    return results


class PANPairwiseSignificance:
    """
    Compute pairwise significance on aligned method intersections.
    """

    def __init__(
        self,
        data_loader: PANDataLoader,
        evaluator: PANEvaluator,
        split_manager: SplitManager,
    ) -> None:
        self.data_loader = data_loader
        self.evaluator = evaluator
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
        alpha_levels: Sequence[float] = (0.1, 0.05, 0.01, 0.005),
    ) -> dict[str, dict]:
        results: dict[str, dict] = {}
        methods = list(dict.fromkeys(methods))
        method_combinations = list(combinations(methods, 2))

        # multiple testing correction via Bonferroni correction
        # Bonferroni controls the family‑wise error rate across all pairwise comparisons for a single metric
        # within a dataset.
        m = len(method_combinations)
        logger.info(f"Computing Bonferroni correction with factor {m}.")

        # unordered unique pairs
        for method_a, method_b in method_combinations:
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
            if n_samples == 0:
                logger.warning(f"{dataset_name} dataset: Skipping {method_a} vs {method_b}, bc no aligned samples.")
                continue
            logger.info(f"{dataset_name} dataset: Aligned {method_a} vs {method_b} with {n_samples} samples.")

            record_a = self.evaluator.get_or_compute_record(
                dataset_name=dataset_name,
                method_name=method_a,
                y_true=y_true,
                scores=scores_a,
                n_samples=n_samples,
                n_splits=self.split_manager.n_splits,
                n_repeats=self.split_manager.n_repeats,
                ci_level=ci_level,
                n_boot=n_boot,
            )
            record_b = self.evaluator.get_or_compute_record(
                dataset_name=dataset_name,
                method_name=method_b,
                y_true=y_true,
                scores=scores_b,
                n_samples=n_samples,
                n_splits=self.split_manager.n_splits,
                n_repeats=self.split_manager.n_repeats,
                ci_level=ci_level,
                n_boot=n_boot,
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
                bonf_correction_factor=m
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
