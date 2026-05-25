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

"""Cross-validation helpers for PAN metrics."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from .pan_metric_computation import EvaluationCVResult, PANMetricComputer
from .pan_storage import PANMetricsRecord, PANMetricsStore

logger = logging.getLogger(__name__)


def compute_aligned_pair_keys_hash(pair_keys: Sequence[tuple[object, object]]) -> str:
    """
    Return a stable, order-sensitive hash for aligned PAN pair keys.

    Repeated CV splits are index based, so the same pair set in a different
    order is a different cache identity.
    """
    canonical_keys = [[str(left_id), str(right_id)] for left_id, right_id in pair_keys]
    payload = json.dumps(canonical_keys, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class SplitManager:
    """
    Deterministic repeated stratified k-fold split generation.

    Store split configuration so it can be reused across methods for paired
    significance tests.
    """

    n_splits: int = 10
    n_repeats: int = 5
    random_state: int = 42

    def build_splits(self, y_true: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        splitter = RepeatedStratifiedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )
        y_true = np.asarray(y_true)
        return list(splitter.split(np.zeros_like(y_true), y_true))

    def split_config(self) -> dict[str, float | int]:
        return {
            "n_splits": int(self.n_splits),
            "n_repeats": int(self.n_repeats),
            "test_size": float(1.0 / self.n_splits),
            "random_state": int(self.random_state),
        }


class PANEvaluator:
    """
    Orchestrates PAN metric computation with consistent splits.
    """

    def __init__(self, metric_computer: PANMetricComputer, split_manager: SplitManager, store: PANMetricsStore) -> None:
        self.metric_computer = metric_computer
        self.split_manager = split_manager
        self.store = store

    def compute_cv(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        *,
        ci_level: float = 0.95,
        n_boot: int = 10000,
        splits: Iterable[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> EvaluationCVResult:
        if splits is None:
            logger.info(f"Building splits for {self.split_manager.n_repeats} repeated "
                        f"{self.split_manager.n_splits}-fold stratified CV with random state {self.split_manager.random_state}")
            splits = self.split_manager.build_splits(y_true)
        return self.metric_computer.tune_thresholds(
            y_true=y_true,
            scores=scores,
            n_splits=self.split_manager.n_splits,
            n_repeats=self.split_manager.n_repeats,
            random_state=self.split_manager.random_state,
            ci_level=ci_level,
            n_boot=n_boot,
            splits=splits,
        )

    def extract_metric_values(
        self,
        per_split: list[dict] | list,
    ) -> dict[str, list[float]]:
        return self.metric_computer.extract_metric_values(per_split)

    def get_or_compute_record(
        self,
        dataset_name: str,
        method_name: str,
        y_true: list[int],
        scores: list[float],
        n_samples: int,
        n_splits: int,
        n_repeats: int,
        ci_level: float,
        n_boot: int,
        aligned_pair_keys_hash: str | None = None,
    ) -> dict:
        if not aligned_pair_keys_hash:
            raise ValueError("aligned_pair_keys_hash is required for PAN metric cache identity.")

        stored = self.store.get_record(
            dataset_name=dataset_name,
            method_name=method_name,
            n_samples=n_samples,
            aligned_pair_keys_hash=aligned_pair_keys_hash,
        )
        if stored:
            precomputed_n_samples = stored.get("n_samples")
            precomputed_n_splits = stored.get("split_config", {}).get("n_splits")
            precomputed_n_repeats = stored.get("split_config", {}).get("n_repeats")
            precomputed_random_state = stored.get("split_config", {}).get("random_state")
            precomputed_aligned_pair_keys_hash = stored.get("aligned_pair_keys_hash")

            if (
                (precomputed_n_samples == n_samples)
                and (precomputed_n_splits == n_splits)
                and (precomputed_n_repeats == n_repeats)
                and (precomputed_random_state == self.split_manager.random_state)
                and (precomputed_aligned_pair_keys_hash == aligned_pair_keys_hash)
            ):
                stored = self.store.ensure_metric_values(
                    stored, self.extract_metric_values
                )
                logger.info(
                    f"Retrieved pre-computed PAN metrics for {dataset_name} using {method_name} on {n_samples} "
                    f"samples ("
                    f"{n_repeats}x{n_splits}-fold CV)."
                )
                return stored

        # guarantees identical repeated CV splits for any methods that share the same sample set and ordering.
        # Ordering is set before ("aligned") given that method have the same number of samples.
        # Hence, repeated CV scores are based on the same random split given the same number of samples.
        logger.info(
            f"{dataset_name} dataset: About to start {self.split_manager.n_repeats} repetitions of"
            f" {self.split_manager.n_splits}-fold CV splits for {method_name}."
        )
        cv_result = self.compute_cv(
            y_true=np.asarray(y_true),
            scores=np.asarray(scores),
            ci_level=ci_level,
            n_boot=n_boot,
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
            aligned_pair_keys_hash=aligned_pair_keys_hash,
            cv_result=cv_result,
        )
        # only newly computed records are saved
        self.store.save_record(record)
        logger.info(
            "Saved PAN metrics for %s on %s (%d samples).", method_name, dataset_name, n_samples
        )

        record_dict = record.to_mongo_dict()
        record_dict["metric_values"] = cv_result.metric_values
        return record_dict

    @staticmethod
    def _record_from_cv(
        *,
        dataset_name: str,
        method_name: str,
        n_samples: int,
        aligned_pair_keys_hash: str,
        cv_result: EvaluationCVResult,
    ) -> PANMetricsRecord:
        per_split = [asdict(result) for result in cv_result.per_split]
        return PANMetricsRecord(
            dataset_name=dataset_name,
            method_name=method_name,
            n_samples=n_samples,
            aligned_pair_keys_hash=aligned_pair_keys_hash,
            metrics_mean=cv_result.metrics_mean,
            metrics_std=cv_result.metrics_std,
            metrics_ci=cv_result.metrics_ci,
            pan_metrics_per_split=per_split,
            split_config=cv_result.split_config,
        )
