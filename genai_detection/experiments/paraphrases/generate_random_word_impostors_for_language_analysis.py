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

"""Generate random-English-word impostors for paraphrase language analysis.

The script samples or loads one original candidate text from MongoDB, generates
two random-word impostors with the same token count, and writes one text file per
impostor to ``results/paraphrase_language_analysis``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bson import ObjectId

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.random_word_impostor_generator import (
    RandomWordImpostorGenerator,
)
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[3]
    / CONFIG.SAVE_PATH
    / "paraphrase_language_analysis"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate two random-English-word impostors for one MongoDB candidate "
            "text and save one result file per impostor."
        )
    )
    parser.add_argument(
        "--dataset-name",
        default=CONFIG.STUDENT_ESSAYS,
        choices=(CONFIG.STUDENT_ESSAYS, CONFIG.BLOG),
        help="Dataset to sample from in MongoDB original_texts.",
    )
    parser.add_argument(
        "--text-id",
        help=(
            "Optional MongoDB _id of the candidate text. If omitted, one text is "
            "sampled randomly from the selected dataset."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where result text files are written.",
    )
    parser.add_argument(
        "--n-impostors",
        type=int,
        default=3,
        help="Number of impostors to generate.",
    )
    parser.add_argument(
        "--top-n-freq-words",
        type=int,
        default=100_000,
        help="TF-IDF feature vocabulary parameter required by the generator base.",
    )
    parser.add_argument(
        "--vocabulary-size",
        type=int,
        default=50_000,
        help="Number of frequent English words requested from wordfreq.",
    )
    parser.add_argument(
        "--distribution-temperature",
        type=float,
        default=1.0,
        help="Temperature applied to the empirical unigram distribution.",
    )
    parser.add_argument(
        "--sentence-mean-tokens",
        type=int,
        default=22,
        help="Mean pseudo-sentence length used for formatting sampled words.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=13,
        help="Seed for reproducible impostor sampling.",
    )
    parser.add_argument(
        "--remote-ray",
        action="store_true",
        help="Use the remote Ray MongoDB connection instead of local forwarded MongoDB.",
    )
    return parser.parse_args()


def _load_candidate(
    mongodb: ParaphraseMongoDB, dataset_name: str, text_id: str | None
) -> dict[str, Any]:
    projection = {"_id": 1, "text": 1, "dataset_name": 1, "dataset": 1}

    if text_id:
        doc = mongodb.original_collection.find_one(
            {"_id": ObjectId(text_id)},
            projection,
        )
        if not doc:
            raise ValueError(f"No original text found for _id={text_id}.")
        actual_dataset = doc.get("dataset_name") or doc.get("dataset")
        if actual_dataset != dataset_name:
            raise ValueError(
                f"Text _id={text_id} belongs to dataset {actual_dataset!r}, "
                f"not {dataset_name!r}."
            )
        return doc

    query = {
        "$or": [
            {"dataset_name": dataset_name},
            {"dataset": dataset_name},
        ],
        "text": {"$type": "string", "$ne": ""},
    }
    docs = list(
        mongodb.get_random_matching_documents_from_collection(
            collection=mongodb.original_collection,
            num_samples=1,
            search_args=query,
        )
    )
    if not docs:
        raise ValueError(f"No original texts found for dataset_name={dataset_name!r}.")
    return docs[0]


def _safe_filename_part(value: Any) -> str:
    return "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(value)
    ).strip("_")


def _write_impostor_file(
    output_dir: Path,
    metadata: dict[str, Any],
    impostor_text: str,
    index: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    text_id = _safe_filename_part(metadata["original_text_id"])
    dataset_name = _safe_filename_part(metadata["dataset_name"])
    output_path = output_dir / f"{dataset_name}_{text_id}_random_words_{index:02d}.txt"

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, indent=2, sort_keys=True))
        handle.write("\n\n--- generated_text ---\n")
        handle.write(impostor_text.strip())
        handle.write("\n")

    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()

    if args.n_impostors < 1:
        raise ValueError("--n-impostors must be at least 1.")

    mongodb = ParaphraseMongoDB(local_ray=not args.remote_ray)
    candidate = _load_candidate(
        mongodb=mongodb,
        dataset_name=args.dataset_name,
        text_id=args.text_id,
    )
    candidate_text = candidate["text"]

    generator_config = {
        "impostor_generation_technique": "random_words",
        "n_impostors": args.n_impostors,
        "top_n_freq_words": args.top_n_freq_words,
        "vocabulary_size": args.vocabulary_size,
        "distribution_temperature": args.distribution_temperature,
        "sentence_mean_tokens": args.sentence_mean_tokens,
        "random_seed": args.random_seed,
    }
    generator = RandomWordImpostorGenerator(
        n_impostors=args.n_impostors,
        top_n_freq_words=args.top_n_freq_words,
        vocabulary_size=args.vocabulary_size,
        distribution_temperature=args.distribution_temperature,
        sentence_mean_tokens=args.sentence_mean_tokens,
        random_seed=args.random_seed,
    )
    impostors = generator.generate_impostors(text=candidate_text, text_id=candidate["_id"])

    base_metadata = {
        "original_text_id": str(candidate["_id"]),
        "dataset_name": args.dataset_name,
        "source_collection": mongodb.original_collection.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_word_count": len(candidate_text.split()),
        "config": generator_config,
    }

    output_paths = []
    for index, impostor_text in enumerate(impostors, start=1):
        metadata = {
            **base_metadata,
            "paraphrase_index": index,
            "generated_word_count": len(impostor_text.split()),
        }
        output_paths.append(
            _write_impostor_file(
                output_dir=args.output_dir,
                metadata=metadata,
                impostor_text=impostor_text,
                index=index,
            )
        )

    logger.info("Wrote %d impostor files:", len(output_paths))
    for path in output_paths:
        logger.info("%s", path)


if __name__ == "__main__":
    main()
