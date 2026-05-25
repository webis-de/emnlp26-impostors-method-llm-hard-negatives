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

"""Summarize detected paraphrase languages per LLM for MongoDB collections."""

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from langdetect import DetectorFactory, LangDetectException, detect_langs

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

DEFAULT_COLLECTIONS = (
    CONFIG.MONGO_NAIVE_PARAPHRASE_COLLECTION,
    CONFIG.MONGO_PARAPHRASE_COLLECTION,
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / CONFIG.SAVE_PATH / "paraphrase_language_analysis"
)

LANGUAGE_LABELS = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify the language of paraphrase texts in MongoDB collections and "
            "write CSV and LaTeX summaries per collection."
        )
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=list(DEFAULT_COLLECTIONS),
        help="MongoDB collection names to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the per-collection CSV and TeX files are written.",
    )
    parser.add_argument(
        "--text-field",
        default="paraphrase",
        help="MongoDB field that contains the paraphrased text.",
    )
    parser.add_argument(
        "--llm-field",
        default="llm",
        help="MongoDB field that contains the LLM/model name.",
    )
    parser.add_argument(
        "--mongo-batch-size",
        type=int,
        default=500,
        help="MongoDB cursor batch size.",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=20_000,
        help="Maximum number of characters per paraphrase sent to the classifier.",
    )
    parser.add_argument(
        "--percentage-denominator",
        choices=("llm", "collection"),
        default="llm",
        help=(
            "Use each LLM total or the full collection total as denominator for "
            "relative_percentage."
        ),
    )
    parser.add_argument(
        "--remote-ray",
        action="store_true",
        help="Use the remote Ray MongoDB connection instead of the local forwarded MongoDB.",
    )
    return parser.parse_args()


def _normalize_llm(value: Any) -> str:
    if value is None:
        return "unknown"
    value = str(value).strip()
    return value if value else "unknown"


def _normalize_language_label(label: Any) -> str:
    label = str(label).strip()
    if not label:
        return "unknown"

    label_key = label.lower()
    return LANGUAGE_LABELS.get(label_key, label)


def _detect_language(text: str) -> str:
    text = text.strip()
    if not text:
        return "unknown"

    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return "unknown"

    if not candidates:
        return "unknown"

    return _normalize_language_label(candidates[0].lang)


def _write_summary(
    output_path: Path,
    counts: Counter[tuple[str, str]],
    llm_totals: Counter[str],
    collection_total: int,
    percentage_denominator: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for (llm, language), count in sorted(
        counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
    ):
        denominator = (
            llm_totals[llm] if percentage_denominator == "llm" else collection_total
        )
        percentage = (count / denominator * 100) if denominator else 0.0
        rows.append(
            {
                "llm": llm,
                "language": language,
                "relative_percentage": f"{percentage:.4f}",
                "absolute_count": count,
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["llm", "language", "relative_percentage", "absolute_count"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _is_non_naive_collection(collection_name: str) -> bool:
    return collection_name == CONFIG.MONGO_PARAPHRASE_COLLECTION


def _latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _latex_label(collection_name: str) -> str:
    slug = collection_name.replace("_", "-")
    return f"tab:{slug}-language-distribution"


def _latex_caption(collection_name: str, include_llm: bool) -> str:
    collection_label = _latex_escape(collection_name)
    if include_llm:
        return f"Language distribution of paraphrases in {collection_label} by LLM."
    return f"Language distribution of paraphrases in {collection_label}."


def _write_latex_table(
    output_path: Path,
    collection_name: str,
    counts: Counter[tuple[str, str]],
    llm_totals: Counter[str],
    collection_total: int,
    percentage_denominator: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    include_llm = not _is_non_naive_collection(collection_name)

    rows: list[dict[str, Any]] = []
    if include_llm:
        for (llm, language), count in sorted(
            counts.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
        ):
            denominator = (
                llm_totals[llm]
                if percentage_denominator == "llm"
                else collection_total
            )
            percentage = (count / denominator * 100) if denominator else 0.0
            rows.append(
                {
                    "llm": llm,
                    "language": language,
                    "relative_percentage": f"{percentage:.4f}",
                    "absolute_count": count,
                }
            )
    else:
        language_counts: Counter[str] = Counter()
        for (_, language), count in counts.items():
            language_counts[language] += count

        for language, count in sorted(
            language_counts.items(), key=lambda item: (-item[1], item[0])
        ):
            percentage = (count / collection_total * 100) if collection_total else 0.0
            rows.append(
                {
                    "language": language,
                    "relative_percentage": f"{percentage:.4f}",
                    "absolute_count": count,
                }
            )

    column_format = "llrr" if include_llm else "lrr"
    header = (
        r"LLM & Language & Relative percentage (\%) & Count \\"
        if include_llm
        else r"Language & Relative percentage (\%) & Count \\"
    )

    body_lines = []
    for row in rows:
        if include_llm:
            body_lines.append(
                " & ".join(
                    [
                        _latex_escape(row["llm"]),
                        _latex_escape(row["language"]),
                        row["relative_percentage"],
                        str(row["absolute_count"]),
                    ]
                )
                + r" \\"
            )
        else:
            body_lines.append(
                " & ".join(
                    [
                        _latex_escape(row["language"]),
                        row["relative_percentage"],
                        str(row["absolute_count"]),
                    ]
                )
                + r" \\"
            )

    table_lines = [
        r"\begin{table}",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{" + _latex_caption(collection_name, include_llm) + "}",
        r"\label{" + _latex_label(collection_name) + "}",
        r"\end{table}",
        "",
    ]
    output_path.write_text("\n".join(table_lines), encoding="utf-8")


def _analyze_collection(
    mongo: ParaphraseMongoDB,
    collection_name: str,
    args: argparse.Namespace,
) -> Path:
    collection = mongo.db[collection_name]
    projection = {"_id": 0, args.llm_field: 1, args.text_field: 1}
    cursor = collection.find({}, projection).batch_size(args.mongo_batch_size)

    counts: Counter[tuple[str, str]] = Counter()
    llm_totals: Counter[str] = Counter()
    collection_total = 0

    for doc in cursor:
        text = doc.get(args.text_field)
        if not isinstance(text, str) or not text.strip():
            text = ""
        elif args.max_characters > 0:
            text = text[: args.max_characters]

        llm = _normalize_llm(doc.get(args.llm_field))
        language = _detect_language(text)
        counts[(llm, language)] += 1
        llm_totals[llm] += 1
        collection_total += 1

    output_path = args.output_dir / f"{collection_name}_language_distribution.csv"
    _write_summary(
        output_path=output_path,
        counts=counts,
        llm_totals=llm_totals,
        collection_total=collection_total,
        percentage_denominator=args.percentage_denominator,
    )
    latex_output_path = output_path.with_suffix(".tex")
    _write_latex_table(
        output_path=latex_output_path,
        collection_name=collection_name,
        counts=counts,
        llm_totals=llm_totals,
        collection_total=collection_total,
        percentage_denominator=args.percentage_denominator,
    )
    logger.info("Wrote %s with %d analyzed documents.", output_path, collection_total)
    logger.info(
        "Wrote %s with %d analyzed documents.", latex_output_path, collection_total
    )
    return output_path


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    DetectorFactory.seed = 0
    mongo = ParaphraseMongoDB(local_ray=not args.remote_ray)

    for collection_name in args.collections:
        _analyze_collection(
            mongo=mongo,
            collection_name=collection_name,
            args=args,
        )


if __name__ == "__main__":
    main()
