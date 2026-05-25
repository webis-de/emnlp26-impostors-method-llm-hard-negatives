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


"""Export the English paraphrase with the lowest semantic similarity score.

Reads the ``paraphrase_scores`` MongoDB collection, finds the row with the
lowest ``sem_sim_avg`` value whose resolved paraphrase is classified as
English, resolves its reference and paraphrase texts, and writes a
human-readable summary to
``results/paraphrasing/examples/lowest_semantic_score_paraphrase.txt``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from bson import ObjectId
from langdetect import DetectorFactory, LangDetectException, detect_langs
from pymongo.collection import Collection

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / CONFIG.SAVE_PATH
    / "paraphrasing"
    / "examples"
    / "lowest_semantic_score_paraphrase.txt"
)
MAX_LANGUAGE_DETECTION_CHARACTERS = 20_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the paraphrase_scores row with the lowest semantic score "
            "whose resolved paraphrase is classified as English and write the "
            "corresponding original/paraphrase texts to a summary file."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output file path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--score-field",
        default="sem_sim_avg",
        help=(
            "Numeric score field to minimize while searching for an English "
            "paraphrase. Defaults to sem_sim_avg."
        ),
    )
    parser.add_argument(
        "--non-naive-model",
        default=CONFIG.OPENAI_MODEL,
        help=(
            "Model value that marks a paraphrase as coming from "
            "non_naive_paraphrases. Defaults to CONFIG.OPENAI_MODEL."
        ),
    )
    parser.add_argument(
        "--remote-ray",
        action="store_true",
        help="Use remote Ray Mongo connection instead of local forwarded Mongo.",
    )
    return parser.parse_args()


def _coerce_object_id(value: Any) -> ObjectId | None:
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def _find_by_id(collection: Collection, value: Any) -> dict[str, Any] | None:
    candidates: list[Any] = []
    object_id = _coerce_object_id(value)
    if object_id is not None:
        candidates.append(object_id)
    candidates.extend([value, str(value)])

    seen: set[str] = set()
    for candidate in candidates:
        key = f"{type(candidate).__name__}:{candidate}"
        if key in seen:
            continue
        seen.add(key)
        doc = collection.find_one({"_id": candidate})
        if doc is not None:
            return doc
    return None


def _validate_score_row(score_doc: dict[str, Any]) -> None:
    for required_field in ["dataset_name", "paraphrase_id", "reference_id"]:
        if required_field not in score_doc:
            raise ValueError(
                f"Score row is missing required field {required_field!r}: "
                f"{score_doc}"
            )


def _iter_score_rows_by_score(
    mongo: ParaphraseMongoDB, score_field: str
) -> Iterator[dict[str, Any]]:
    cursor = mongo.paraphrase_score_collection.find(
        {score_field: {"$exists": True, "$ne": None}},
        sort=[(score_field, 1), ("_id", 1)],
    )
    found_any = False
    for score_doc in cursor:
        found_any = True
        _validate_score_row(score_doc)
        yield score_doc

    if not found_any:
        raise ValueError(
            f"No rows with a non-null {score_field!r} value found in "
            f"{CONFIG.MONGO_PARAPHRASE_SCORE_COLLECTION!r}."
        )


def _resolve_original_doc(
    mongo: ParaphraseMongoDB, reference_id: Any
) -> dict[str, Any]:
    original_doc = _find_by_id(mongo.original_collection, reference_id)
    if original_doc is None:
        raise ValueError(
            f"No original_texts document found with _id matching reference_id={reference_id!r}."
        )
    return original_doc


def _resolve_paraphrase_doc(
    mongo: ParaphraseMongoDB,
    paraphrase_id: Any,
    score_doc: dict[str, Any],
    non_naive_model: str,
) -> tuple[dict[str, Any], str]:
    score_model = score_doc.get("model") or score_doc.get("llm")
    non_naive_doc = _find_by_id(mongo.non_naive_paraphrase_collection, paraphrase_id)
    naive_doc = _find_by_id(mongo.naive_paraphrase_collection, paraphrase_id)

    if score_model == non_naive_model:
        if non_naive_doc is None:
            raise ValueError(
                f"Score row model={score_model!r}, but no non_naive_paraphrases "
                f"document has _id={paraphrase_id!r}."
            )
        return non_naive_doc, CONFIG.MONGO_PARAPHRASE_COLLECTION

    if score_model is not None:
        if naive_doc is None:
            raise ValueError(
                f"Score row model={score_model!r}, but no naive_paraphrases "
                f"document has _id={paraphrase_id!r}."
            )
        return naive_doc, CONFIG.MONGO_NAIVE_PARAPHRASE_COLLECTION

    if non_naive_doc is not None and non_naive_doc.get("llm") == non_naive_model:
        return non_naive_doc, CONFIG.MONGO_PARAPHRASE_COLLECTION
    if naive_doc is not None:
        return naive_doc, CONFIG.MONGO_NAIVE_PARAPHRASE_COLLECTION
    if non_naive_doc is not None:
        return non_naive_doc, CONFIG.MONGO_PARAPHRASE_COLLECTION

    raise ValueError(
        "No paraphrase document found with _id matching "
        f"paraphrase_id={paraphrase_id!r} in either paraphrase collection."
    )


def _require_paraphrase_text(paraphrase_doc: dict[str, Any]) -> str:
    paraphrase_text = paraphrase_doc.get("paraphrase")
    if not isinstance(paraphrase_text, str) or not paraphrase_text.strip():
        raise ValueError(
            f"Paraphrase document has no non-empty paraphrase field: {paraphrase_doc}"
        )
    return paraphrase_text


def _classify_text_language(text: str) -> str:
    text = text.strip()
    if not text:
        return "unknown"

    try:
        candidates = detect_langs(text[:MAX_LANGUAGE_DETECTION_CHARACTERS])
    except LangDetectException:
        return "unknown"

    if not candidates:
        return "unknown"
    return candidates[0].lang


def _resolve_lowest_english_paraphrase(
    mongo: ParaphraseMongoDB,
    score_field: str,
    non_naive_model: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str, int]:
    skipped_non_english = 0
    for score_doc in _iter_score_rows_by_score(mongo=mongo, score_field=score_field):
        paraphrase_doc, paraphrase_collection_name = _resolve_paraphrase_doc(
            mongo=mongo,
            paraphrase_id=score_doc["paraphrase_id"],
            score_doc=score_doc,
            non_naive_model=non_naive_model,
        )
        paraphrase_text = _require_paraphrase_text(paraphrase_doc)
        paraphrase_language = _classify_text_language(paraphrase_text)
        if paraphrase_language == "en":
            return (
                score_doc,
                paraphrase_doc,
                paraphrase_collection_name,
                paraphrase_language,
                skipped_non_english,
            )

        skipped_non_english += 1
        logger.info(
            "Skipping paraphrase_id=%s with %s=%s because detected language is %s.",
            _format_scalar(score_doc.get("paraphrase_id")),
            score_field,
            _format_scalar(score_doc.get(score_field)),
            paraphrase_language,
        )

    raise ValueError(
        "No English paraphrase found among rows with a non-null "
        f"{score_field!r} value in {CONFIG.MONGO_PARAPHRASE_SCORE_COLLECTION!r}."
    )


def _format_scalar(value: Any) -> str:
    if isinstance(value, ObjectId):
        return str(value)
    return "" if value is None else str(value)


def _format_extracted_info(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, indent=2, sort_keys=True, default=str)
    return str(value)


def _score_metadata_lines(score_doc: dict[str, Any]) -> list[str]:
    primary_fields = {"_id", "dataset_name", "paraphrase_id", "reference_id"}
    skip_fields = primary_fields | {"hashcode", "bertscore_hash"}
    lines = [
        f"score_id: {_format_scalar(score_doc.get('_id'))}",
        f"dataset_name: {_format_scalar(score_doc.get('dataset_name'))}",
        f"paraphrase_id: {_format_scalar(score_doc.get('paraphrase_id'))}",
        f"reference_id: {_format_scalar(score_doc.get('reference_id'))}",
    ]

    for key in sorted(k for k in score_doc.keys() if k not in skip_fields):
        value = score_doc[key]
        if isinstance(value, (str, int, float, bool, ObjectId)) or value is None:
            lines.append(f"{key}: {_format_scalar(value)}")
    return lines


def _build_summary(
    score_doc: dict[str, Any],
    original_doc: dict[str, Any],
    paraphrase_doc: dict[str, Any],
    paraphrase_collection_name: str,
    paraphrase_language: str,
    skipped_non_english: int,
) -> str:
    original_text = original_doc.get("text")
    paraphrase_text = _require_paraphrase_text(paraphrase_doc)
    if not isinstance(original_text, str) or not original_text.strip():
        raise ValueError(f"Original document has no non-empty text field: {original_doc}")

    lines = [
        "Lowest English Semantic Similarity Paraphrase",
        "=============================================",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Score Entry",
        "-----------",
        *_score_metadata_lines(score_doc),
        "",
        "Resolved Documents",
        "------------------",
        f"original_collection: {CONFIG.MONGO_ORIGINAL_TEXT_COLLECTION}",
        f"paraphrase_collection: {paraphrase_collection_name}",
        f"original_id: {_format_scalar(original_doc.get('_id'))}",
        f"paraphrase_id: {_format_scalar(paraphrase_doc.get('_id'))}",
        f"paraphrase_language: {paraphrase_language}",
        f"non_english_lower_score_candidates_skipped: {skipped_non_english}",
    ]

    if paraphrase_collection_name == CONFIG.MONGO_NAIVE_PARAPHRASE_COLLECTION:
        lines.append(f"naive_llm: {_format_scalar(paraphrase_doc.get('llm'))}")
    else:
        lines.append(f"non_naive_model: {_format_scalar(paraphrase_doc.get('llm'))}")
        extracted_info = _format_extracted_info(paraphrase_doc.get("extracted_info"))
        if extracted_info:
            lines.extend(
                [
                    "",
                    "Extracted Info",
                    "--------------",
                    extracted_info,
                ]
            )

    lines.extend(
        [
            "",
            "Original Text",
            "-------------",
            original_text.strip(),
            "",
            "Paraphrase Text",
            "---------------",
            paraphrase_text.strip(),
            "",
        ]
    )
    return "\n".join(lines)


def export_lowest_semantic_score_summary(
    output_path: Path,
    score_field: str,
    non_naive_model: str,
    local_ray: bool,
) -> Path:
    mongo = ParaphraseMongoDB(local_ray=local_ray)
    (
        score_doc,
        paraphrase_doc,
        paraphrase_collection_name,
        paraphrase_language,
        skipped_non_english,
    ) = _resolve_lowest_english_paraphrase(
        mongo=mongo,
        score_field=score_field,
        non_naive_model=non_naive_model,
    )
    original_doc = _resolve_original_doc(
        mongo=mongo, reference_id=score_doc["reference_id"]
    )

    summary = _build_summary(
        score_doc=score_doc,
        original_doc=original_doc,
        paraphrase_doc=paraphrase_doc,
        paraphrase_collection_name=paraphrase_collection_name,
        paraphrase_language=paraphrase_language,
        skipped_non_english=skipped_non_english,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    logger.info("Wrote summary to %s", output_path)
    return output_path


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    export_lowest_semantic_score_summary(
        output_path=args.output,
        score_field=args.score_field,
        non_naive_model=args.non_naive_model,
        local_ray=not args.remote_ray,
    )


if __name__ == "__main__":
    main()
