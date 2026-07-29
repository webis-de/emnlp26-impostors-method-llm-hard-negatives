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
"""Batch paraphrasing via OpenAI's asynchronous Batch API.

The BatchParaphraser intentionally does not implement realtime paraphrasing.
It collects paraphrase requests, submits them as one JSONL batch, and later
collects finished results. Completed paraphrases are persisted in MongoDB's
``naive_paraphrases`` collection using the same schema as other one-step
paraphrasers.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from bson import ObjectId
from openai import OpenAI
from pymongo.collection import Collection

from genai_detection.config import CONFIG
from genai_detection.paraphrasing.batch_signatures import DSPyOneStepBatchSignature
from genai_detection.paraphrasing.one_step_paraphrasers import OneStepParaphraser

logger = logging.getLogger(__name__)


class BatchParaphraser(OneStepParaphraser):
    """Collect, submit, and finalize one-step paraphrases using OpenAI batches."""

    ENDPOINT = "/v1/chat/completions"
    COMPLETION_WINDOW = "24h"
    TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
    ACTIVE_STATUSES = {
        "validating",
        "in_progress",
        "finalizing",
        "cancelling",
    }

    def __init__(
        self,
        model_id: str = CONFIG.BATCH_OPENAI_MODEL,
        n_paraphrases: int = 50,
        job_collection_name: str = CONFIG.MONGO_JOB_COLLECTION_NAME,
        endpoint: str = ENDPOINT,
        completion_window: str = COMPLETION_WINDOW,
    ) -> None:
        super().__init__(n_paraphrases=n_paraphrases, model_id=model_id)
        self.endpoint = endpoint
        self.completion_window = completion_window
        self.job_collection_name = job_collection_name
        self.job_collection = (
            self.mongoDB.job_collection
            if job_collection_name == CONFIG.MONGO_JOB_COLLECTION_NAME
            else self._ensure_job_collection(job_collection_name)
        )
        self.pending_requests: List[Dict[str, Any]] = []
        self.client = OpenAI(
            base_url=(
                CONFIG.OPENAI_URL
                if os.path.exists("/Users/klara")
                else os.environ["OPENAI_URL"]
            ),
            api_key=(
                CONFIG.OPENAI_KEY
                if os.path.exists("/Users/klara")
                else os.environ["OPENAI_KEY"]
            ),
        )

    def paraphrase(
        self,
        text: str,
        prompt: str = CONFIG.PROMPT,
        max_length: int = CONFIG.MAX_LENGTH,
    ) -> str:
        raise NotImplementedError(
            "BatchParaphraser is asynchronous. Use add_text/add_texts, "
            "submit_batch, and collect_batch instead of paraphrase."
        )

    def add_text(
        self,
        text_id: ObjectId | str,
        text: Optional[str] = None,
        dataset_name: Optional[str] = None,
        temperature: float = CONFIG.TEMPERATURE,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Add one text to the queue and return the generated custom ids."""
        original_text, original_text_id = self.mongoDB.get_text_or_id_from_orginal_collection(
            text=text,
            text_id=ObjectId(text_id),
        )
        if not dataset_name:
            doc = self.mongoDB.find_document_by_id(collection=self.mongoDB.original_collection, document_id=text_id, )[0]
            dataset_name = doc.get("dataset_name")
        if isinstance(original_text, tuple):
            original_text = original_text[0]

        custom_ids = []
        for paraphrase_index in range(self.n_paraphrases):
            custom_id = self._build_custom_id(
                text_id=ObjectId(original_text_id),
                request_index=len(self.pending_requests),
                paraphrase_index=paraphrase_index,
            )
            body = self._build_request_body(
                text=original_text,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
            )
            self.pending_requests.append(
                {
                    "custom_id": custom_id,
                    "text_id": ObjectId(original_text_id),
                    "original_text": original_text,
                    "dataset_name": dataset_name,
                    "temperature": temperature,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "paraphrase_index": paraphrase_index,
                    "metadata": metadata or {},
                    "request": {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": self.endpoint,
                        "body": body,
                    },
                    "saved": False,
                    "saved_paraphrase_id": None,
                    "error": None,
                }
            )
            custom_ids.append(custom_id)
        return custom_ids

    def add_texts(
        self,
        text_ids: Iterable[ObjectId | str],
        dataset_name: Optional[str] = None,
        temperature: float = CONFIG.TEMPERATURE,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Add multiple MongoDB original text ids to the in-memory queue."""
        custom_ids = []
        for text_id in text_ids:
            custom_ids.extend(
                self.add_text(
                    text_id=text_id,
                    dataset_name=dataset_name,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    metadata=metadata,
                )
            )
        return custom_ids

    def submit_batch(
        self,
        description: Optional[str] = None,
        clear_pending: bool = True,
    ) -> Optional[str]:
        """Upload queued requests, create an OpenAI batch, and store job state."""
        if not self.pending_requests:
            raise ValueError("No pending paraphrase requests to submit.")
        if self._has_pending_requests_already_active():
            return None

        input_file_path = self._write_jsonl_input_file(self.pending_requests)
        try:
            with input_file_path.open("rb") as input_file:
                uploaded_file = self.client.files.create(
                    file=input_file,
                    purpose="batch",
                )

            batch = self.client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint=self.endpoint,
                completion_window=self.completion_window,
                metadata=self._string_metadata(
                    {
                        "description": description or "batch paraphrasing",
                        "model": self.model_id,
                        "request_count": str(len(self.pending_requests)),
                    }
                ),
            )
            batch_doc = self._build_batch_job_document(
                batch=batch,
                input_file_id=uploaded_file.id,
                description=description,
                requests=self.pending_requests,
            )
            self.job_collection.insert_one(batch_doc)
            if clear_pending:
                self.pending_requests = []
            logger.info("Submitted OpenAI paraphrase batch %s.", batch.id)
            return batch.id
        finally:
            input_file_path.unlink(missing_ok=True)

    def collect_batch(self, batch_id: str, save: bool = True) -> Dict[str, Any]:
        """Refresh batch status and save completed paraphrases when available."""
        batch = self.client.batches.retrieve(batch_id)
        self._update_job_from_batch(batch_id=batch_id, batch=batch)

        if batch.status not in self.TERMINAL_STATUSES:
            return self._batch_summary(
                batch=batch,
                message="Batch is not finished yet.",
            )

        if batch.status != "completed":
            self._collect_error_file_if_present(batch_id=batch_id, batch=batch)
            return self._batch_summary(
                batch=batch,
                message=f"Batch ended with status {batch.status}.",
            )

        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            error_rows = self._collect_error_file_if_present(
                batch_id=batch_id,
                batch=batch,
            )
            return self._batch_summary(
                batch=batch,
                message="Batch completed but no output_file_id is available.",
                failed_count=len(error_rows),
                errors=error_rows[:10],
            )

        result_rows = self._download_jsonl_file(output_file_id)
        saved_count = 0
        failed_count = 0
        errors = []

        for row in result_rows:
            custom_id = row.get("custom_id")
            request_state = self._request_state(batch_id=batch_id, custom_id=custom_id)
            if not request_state:
                failed_count += 1
                errors.append(
                    {"custom_id": custom_id, "error": "No matching request state."}
                )
                continue

            response_error = row.get("error")
            if response_error:
                failed_count += 1
                errors.append({"custom_id": custom_id, "error": response_error})
                self._mark_request_failed(batch_id, custom_id, response_error)
                continue

            paraphrase = self._extract_paraphrase(row)
            if not paraphrase:
                failed_count += 1
                error = "No paraphrase found in batch response."
                errors.append({"custom_id": custom_id, "error": error})
                self._mark_request_failed(batch_id, custom_id, error)
                continue

            if save and not request_state.get("saved"):
                self._save_finished_paraphrase(
                    batch_id=batch_id,
                    output_file_id=output_file_id,
                    custom_id=custom_id,
                    request_state=request_state,
                    response_row=row,
                    paraphrase=paraphrase,
                )
                saved_count += 1

        self._collect_error_file_if_present(batch_id=batch_id, batch=batch)
        self.job_collection.update_one(
            {"batch_id": batch_id},
            {
                "$set": {
                    "status": batch.status,
                    "output_file_id": output_file_id,
                    "last_collected_at": self._now(),
                    "saved_count": self._count_saved_requests(batch_id),
                    "failed_count": failed_count,
                    "collection_errors": errors,
                }
            },
        )
        return self._batch_summary(
            batch=batch,
            message="Batch completed and output was collected.",
            saved_count=saved_count,
            failed_count=failed_count,
            errors=errors,
        )

    def get_batch_job_state(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored MongoDB job-state document for a batch."""
        return self.job_collection.find_one({"batch_id": batch_id})

    def _ensure_job_collection(self, collection_name: str) -> Collection:
        if collection_name not in self.mongoDB.db.list_collection_names():
            self.mongoDB.db.create_collection(collection_name)
            logger.info("Created MongoDB collection: %s", collection_name)
        return self.mongoDB.db[collection_name]

    def _has_pending_requests_already_active(self) -> bool:
        pending_text_ids = {
            ObjectId(request_state["text_id"])
            for request_state in self.pending_requests
        }
        if not pending_text_ids:
            return False

        active_jobs = self.job_collection.find(
            {
                "status": {"$in": list(self.ACTIVE_STATUSES)},
                "model": self.model_id,
                "endpoint": self.endpoint,
                "requests.text_id": {"$in": list(pending_text_ids)},
            },
            {
                "batch_id": 1,
                "status": 1,
                "requests.custom_id": 1,
                "requests.text_id": 1,
            },
        )
        overlaps = []
        for job in active_jobs:
            matching_text_ids = [
                request["text_id"]
                for request in job.get("requests", [])
                if request.get("text_id") in pending_text_ids
            ]
            if matching_text_ids:
                overlaps.append(
                    {
                        "batch_id": job.get("batch_id"),
                        "status": job.get("status"),
                        "text_ids": matching_text_ids,
                    }
                )

        if overlaps:
            logger.warning(
                "Refusing to submit duplicate OpenAI batch paraphrase requests. "
                "Active overlapping jobs exist: %s",
                overlaps,
            )
            return True
        return False

    def _build_request_body(
        self,
        text: str,
        temperature: float,
        top_p: Optional[float],
        frequency_penalty: Optional[float],
    ) -> Dict[str, Any]:
        is_reasoning_model = any(name in self.model_id.lower() for name in ["gpt-5"])
        body: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": DSPyOneStepBatchSignature.system_instruction(),
                },
                {
                    "role": "user",
                    "content": DSPyOneStepBatchSignature.input_content(text),
                },
            ],
            "temperature": 1.0 if is_reasoning_model else temperature,
            "response_format": DSPyOneStepBatchSignature.response_format(),
            "reasoning_effort": "minimal",
        }
        if top_p is not None:
            body["top_p"] = top_p
        if frequency_penalty is not None:
            body["frequency_penalty"] = frequency_penalty
        return body

    def _build_custom_id(
        self,
        text_id: ObjectId,
        request_index: int,
        paraphrase_index: int,
    ) -> str:
        return f"naive-paraphrase-{text_id}-{paraphrase_index}-{request_index}"

    def _write_jsonl_input_file(self, requests: List[Dict[str, Any]]) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".jsonl",
            delete=False,
        ) as tmp_file:
            for request_state in requests:
                tmp_file.write(
                    json.dumps(
                        request_state["request"],
                        ensure_ascii=True,
                        default=str,
                    )
                    + "\n"
                )
            return Path(tmp_file.name)

    def _build_batch_job_document(
        self,
        batch: Any,
        input_file_id: str,
        description: Optional[str],
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "batch_id": batch.id,
            "input_file_id": input_file_id,
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
            "status": batch.status,
            "model": self.model_id,
            "endpoint": self.endpoint,
            "completion_window": self.completion_window,
            "description": description,
            "request_count": len(requests),
            "saved_count": 0,
            "failed_count": 0,
            "created_at": self._now(),
            "submitted_at": self._now(),
            "last_checked_at": self._now(),
            "last_collected_at": None,
            "requests": requests,
            "batch_metadata": self._json_safe(self._model_dump(batch)),
        }

    def _update_job_from_batch(self, batch_id: str, batch: Any) -> None:
        self.job_collection.update_one(
            {"batch_id": batch_id},
            {
                "$set": {
                    "status": batch.status,
                    "output_file_id": getattr(batch, "output_file_id", None),
                    "error_file_id": getattr(batch, "error_file_id", None),
                    "last_checked_at": self._now(),
                    "batch_metadata": self._json_safe(self._model_dump(batch)),
                }
            },
        )

    def _download_jsonl_file(self, file_id: str) -> List[Dict[str, Any]]:
        content_response = self.client.files.content(file_id)
        if hasattr(content_response, "text"):
            raw_content = content_response.text
        elif hasattr(content_response, "read"):
            raw_content = content_response.read()
        else:
            raw_content = content_response

        if isinstance(raw_content, bytes):
            raw_content = raw_content.decode("utf-8")

        rows = []
        for line in str(raw_content).splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def _request_state(
        self, batch_id: str, custom_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not custom_id:
            return None
        job = self.job_collection.find_one(
            {"batch_id": batch_id},
            {"requests": {"$elemMatch": {"custom_id": custom_id}}},
        )
        if not job or not job.get("requests"):
            return None
        return job["requests"][0]

    def _extract_paraphrase(self, response_row: Dict[str, Any]) -> str:
        response_body = response_row.get("response", {}).get("body", {})
        return DSPyOneStepBatchSignature.parse_chat_completion_body(response_body)

    def _save_finished_paraphrase(
        self,
        batch_id: str,
        output_file_id: str,
        custom_id: str,
        request_state: Dict[str, Any],
        response_row: Dict[str, Any],
        paraphrase: str,
    ) -> None:
        response_body = response_row.get("response", {}).get("body", {})
        usage = response_body.get("usage")
        job_state = self.get_batch_job_state(batch_id) or {}
        extracted_info = {
            "batch_id": batch_id,
            "input_file_id": job_state.get("input_file_id"),
            "output_file_id": output_file_id,
            "custom_id": custom_id,
            "endpoint": self.endpoint,
            "completion_window": self.completion_window,
            "request_metadata": request_state.get("metadata", {}),
            "signature": DSPyOneStepBatchSignature.metadata(),
            "openai_response_id": response_body.get("id"),
            "openai_usage": usage,
            "decoding": {
                "temperature": request_state.get("temperature"),
                "top_p": request_state.get("top_p"),
                "frequency_penalty": request_state.get("frequency_penalty"),
            },
        }
        self.save_paraphrase_in_mongodb(
            original_text=request_state["original_text"],
            original_text_id=request_state["text_id"],
            paraphrased_text=paraphrase,
            extracted_info=extracted_info,
            total_costs=None,
            temperature=request_state.get("temperature", CONFIG.TEMPERATURE),
            prompt="DSPyOneStepSignature",
            dataset_name=request_state.get("dataset_name"),
            intermediate_prompt="",
            collection=self.mongoDB.naive_paraphrase_collection,
        )
        saved_doc = self.mongoDB.naive_paraphrase_collection.find_one(
            {"extracted_info.custom_id": custom_id},
            {"_id": 1},
            sort=[("_id", -1)],
        )
        saved_id = saved_doc["_id"] if saved_doc else None
        self.job_collection.update_one(
            {"batch_id": batch_id, "requests.custom_id": custom_id},
            {
                "$set": {
                    "requests.$.saved": True,
                    "requests.$.saved_at": self._now(),
                    "requests.$.saved_paraphrase_id": saved_id,
                    "requests.$.response_metadata": self._json_safe(response_body),
                }
            },
        )

    def _mark_request_failed(
        self, batch_id: str, custom_id: str, error: Any
    ) -> None:
        self.job_collection.update_one(
            {"batch_id": batch_id, "requests.custom_id": custom_id},
            {
                "$set": {
                    "requests.$.error": self._json_safe(error),
                    "requests.$.failed_at": self._now(),
                }
            },
        )

    def _collect_error_file_if_present(
        self,
        batch_id: str,
        batch: Any,
    ) -> List[Dict[str, Any]]:
        error_file_id = getattr(batch, "error_file_id", None)
        if not error_file_id:
            return []
        try:
            error_rows = self._download_jsonl_file(error_file_id)
        except Exception as e:
            logger.warning("Failed to download batch error file %s: %s", error_file_id, e)
            error_rows = [{"error": str(e)}]
        self.job_collection.update_one(
            {"batch_id": batch_id},
            {
                "$set": {
                    "error_file_id": error_file_id,
                    "error_rows": self._json_safe(error_rows),
                }
            },
        )
        return error_rows

    def _count_saved_requests(self, batch_id: str) -> int:
        job = self.job_collection.find_one({"batch_id": batch_id}, {"requests": 1})
        if not job:
            return 0
        return sum(1 for request in job.get("requests", []) if request.get("saved"))

    def _batch_summary(
        self,
        batch: Any,
        message: str,
        saved_count: int = 0,
        failed_count: int = 0,
        errors: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "batch_id": batch.id,
            "status": batch.status,
            "message": message,
            "request_counts": self._json_safe(getattr(batch, "request_counts", None)),
            "input_file_id": getattr(batch, "input_file_id", None),
            "output_file_id": getattr(batch, "output_file_id", None),
            "error_file_id": getattr(batch, "error_file_id", None),
            "saved_count": saved_count,
            "failed_count": failed_count,
            "errors": errors or [],
        }

    def _model_dump(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return value

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, ObjectId):
            return value
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d_%H-%M-%S")
        if hasattr(value, "model_dump"):
            return self._json_safe(value.model_dump())
        if hasattr(value, "dict"):
            return self._json_safe(value.dict())
        if isinstance(value, dict):
            return {key: self._json_safe(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_safe(item) for item in value]
        return str(value)

    def _string_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        return {key: str(value) for key, value in metadata.items() if value is not None}

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
