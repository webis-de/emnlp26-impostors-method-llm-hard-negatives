from __future__ import annotations

"""Persistence utilities for PAN metrics (MongoDB + disk)."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB


@dataclass
class PANMetricsRecord:
    """
    Canonical PAN metrics payload stored in MongoDB and optionally on disk.

    Required fields reflect the evaluation summary and per-split metrics.
    """

    dataset_name: str
    method_name: str
    n_samples: int
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]
    metrics_ci: dict[str, dict[str, float | int | str]]
    pan_metrics_per_split: list[dict]
    split_config: dict[str, float | int] | None = None

    def to_mongo_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return _jsonify_value(payload)


def _jsonify_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonify_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonify_value(v) for k, v in value.items()}
    return value


class PANMetricsStore:
    """
    Handle MongoDB persistence and disk snapshots for PAN metrics.
    """

    def __init__(
        self,
        mongo: ParaphraseMongoDB | None = None,
        save_dir: Path | None = None,
    ) -> None:
        self.mongo = mongo or ParaphraseMongoDB(local_ray=Path.home().exists())
        self.save_dir = save_dir or self._default_save_dir()
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_save_dir() -> Path:
        repo_root = Path(__file__).resolve().parents[4]
        return repo_root / CONFIG.SAVE_PATH / "reproduction" / "pan_metrics"

    def get_record(
        self,
        dataset_name: str,
        method_name: str,
        n_samples: int | None = None,
    ) -> dict[str, Any] | None:
        query: dict[str, Any] = {"dataset_name": dataset_name, "method_name": method_name}
        if n_samples is not None:
            query["n_samples"] = n_samples
        doc = self.mongo.pan_metrics_collection.find_one(
            query,
            {"_id": 0},
            sort=[("_id", -1)],
        )
        if doc is None:
            return None
        return self._normalize_doc(doc)

    def save_record(self, record: PANMetricsRecord, persist_disk: bool = True) -> None:
        payload = record.to_mongo_dict()
        self.mongo.pan_metrics_collection.insert_one(payload)
        if persist_disk:
            self.save_record_to_disk(record)

    def save_record_to_disk(self, record: PANMetricsRecord, save_dir: Path | None = None) -> Path:
        save_dir = save_dir or self.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{record.dataset_name}_pan_metrics_{record.method_name}.json"
        with open(out_path, "w") as f:
            json.dump(record.to_mongo_dict(), f, indent=2)
        return out_path

    def ensure_metric_values(
        self,
        record: dict[str, Any],
        extractor: Callable[[list], dict[str, list[float]]],
    ) -> dict[str, Any]:
        if "metric_values" not in record or not record["metric_values"]:
            per_split = record.get("pan_metrics_per_split") or record.get("per_split")
            if per_split:
                record["metric_values"] = extractor(per_split)
        return record

    @staticmethod
    def _normalize_doc(doc: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize legacy and new MongoDB record formats into a common shape.
        """
        if "pan_metrics_per_split" in doc:
            return doc

        pan_payload = doc.get("pan_metrics", {}) if isinstance(doc, dict) else {}
        if not pan_payload:
            return doc

        normalized = {
            "dataset_name": doc.get("dataset_name"),
            "method_name": doc.get("method_name"),
            "n_samples": doc.get("n_samples"),
            "metrics_mean": doc.get("metrics_mean") or pan_payload.get("metrics_mean") or {},
            "metrics_std": doc.get("metrics_std") or pan_payload.get("metrics_std") or {},
            "metrics_ci": doc.get("metrics_ci") or pan_payload.get("metrics_ci") or {},
            "pan_metrics_per_split": pan_payload.get("per_split")
            or pan_payload.get("pan_metrics_per_split")
            or [],
            "split_config": pan_payload.get("split_config") or doc.get("split_config"),
            "metric_values": pan_payload.get("metric_values"),
        }
        return normalized
