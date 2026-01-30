import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from bson import ObjectId
from pymongo.errors import PyMongoError

from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

ID_FIELDS = {"_id", "text_id", "original_text_id"}
INT_FIELDS = {"length_original_text", "length_paraphrased_text", "length_impostor_text"}
FLOAT_FIELDS = {"temperature", "openai_costs", "total_costs"}
JSON_FIELDS = {"extracted_info"}


def _default_downloads_dir() -> Path:
    repo_downloads = (Path.cwd().absolute() / "Downloads")
    if repo_downloads.is_dir():
        return repo_downloads
    return Path.home() / "Downloads"


def _is_object_id(value: str) -> bool:
    if len(value) != 24:
        return False
    for ch in value:
        if ch not in "0123456789abcdefABCDEF":
            return False
    return True


def _coerce_value(key: str, value: Optional[str]):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
    if value == "" or value == "nan" or value == "NaN":
        return None

    if key in JSON_FIELDS:
        if isinstance(value, str) and value and value[0] in "{[":
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    if key in INT_FIELDS:
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return value

    if key in FLOAT_FIELDS:
        try:
            return float(value)
        except (ValueError, TypeError):
            return value

    if key in ID_FIELDS or key.endswith("_id"):
        if isinstance(value, str) and value.startswith("ObjectId("):
            value = value.replace("ObjectId(", "").replace(")", "").strip("'\" ")
        if isinstance(value, str) and _is_object_id(value):
            try:
                return ObjectId(value)
            except Exception:
                return value
        return value

    return value


def _iter_csv_documents(csv_path: Path) -> Iterable[Dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = {k: _coerce_value(k, v) for k, v in row.items()}
            yield doc


def _insert_csv(
    collection,
    csv_path: Path,
    batch_size: int,
) -> int:
    inserted = 0
    batch: List[Dict] = []
    for doc in _iter_csv_documents(csv_path):
        batch.append(doc)
        if len(batch) >= batch_size:
            try:
                collection.insert_many(batch, ordered=False)
                inserted += len(batch)
            except PyMongoError:
                pass
            batch.clear()
    if batch:
        try:
            collection.insert_many(batch, ordered=False)
            inserted += len(batch)
        except PyMongoError:
            pass
    return inserted


def _select_csvs(downloads_dir: Path) -> Dict[str, Path]:
    csvs = list(downloads_dir.glob("*.csv"))
    selected: Dict[str, Path] = {}
    for csv_path in csvs:
        name = csv_path.name.lower()
        if "non_naive_paraphrases" in name:
            selected["non_naive_paraphrases"] = csv_path
        elif "on_the_fly_paraphrases" in name:
            selected["on_the_fly_paraphrases"] = csv_path
        elif "naive_paraphrases" in name:
            selected["naive_paraphrases"] = csv_path
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path(os.getenv("DOWNLOADS_DIR")) if os.getenv("DOWNLOADS_DIR") else _default_downloads_dir(),
        help="Directory containing the CSV backups (default: Downloads or ./downloads).",
    )
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)

    downloads_dir: Path = args.downloads_dir
    print("downloads_dir:", downloads_dir)
    csv_map = _select_csvs(downloads_dir)
    print("csv_map:", csv_map)
    if not csv_map:
        raise FileNotFoundError(f"No matching CSVs found in {downloads_dir}")

    mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))

    # if "non_naive_paraphrases" in csv_map:
    #     print("Non Naive Paraphrases:")
    #     _insert_csv(
    #         mongoDB.non_naive_paraphrase_collection,
    #         csv_map["non_naive_paraphrases"],
    #         args.batch_size,
    #     )
    #     print("finished non naive paraphrases.")

    if "naive_paraphrases" in csv_map:
        print("Naive Paraphrases:")
        _insert_csv(
            mongoDB.naive_paraphrase_collection,
            csv_map["naive_paraphrases"],
            args.batch_size,
        )
        print("Finished Naive Paraphrases.")

    if "on_the_fly_paraphrases" in csv_map:
        print("on_the_fly_paraphrases")
        _insert_csv(
            mongoDB.on_the_fly_collection,
            csv_map["on_the_fly_paraphrases"],
            args.batch_size,
        )
        print("finished on_the_fly paraphrases.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
