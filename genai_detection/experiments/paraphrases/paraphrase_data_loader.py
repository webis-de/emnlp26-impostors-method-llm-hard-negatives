# Copyright 2024 Klara M. Gutekunst, Webis
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

import os
import re
from itertools import chain, zip_longest
from pathlib import Path
import chardet
import pandas as pd
from bson import ObjectId

from genai_detection.util import preprocess_text as _preprocess_text


class ParaphraseDataLoader:
    def __init__(
        self,
        mongodb,
        base_dirs: dict[str, Path],
        min_words: int = 700,
    ):
        self.mongodb = mongodb
        self.base_dirs = base_dirs
        self.min_words = min_words

    @staticmethod
    def get_century(time_period) -> int:
        if isinstance(time_period, str):
            if "present" in time_period.lower():
                return 21
            try:
                time_period = int(re.sub(r"[^\d]", "", time_period))
            except ValueError:
                return 0
        if not isinstance(time_period, (int, float)) or time_period <= 0:
            return 0

        if time_period > 100:
            century = time_period // 100
            century += 1
            return int(century)
        return int(time_period)

    def create_student_essays_metadata(self, base_dir: Path) -> pd.DataFrame:
        task_description = {
            "Ass1": "Stream of consciousness, personal journal, diary, essay",
            "Ass2": "childhood, personal journal, diary, essay",
            "Ass3": "personality, essay, reflection, self-analysis",
            "Ass4": "Thematic Apperception Test, essay",
            "Ass5": "different theories, essay",
        }
        files_by_ass = []

        for assignment in list(task_description.keys()):
            files = [
                file for file in (base_dir / assignment).glob("*.txt") if file.is_file()
            ]
            files_by_ass.append(files)

        interleaved_files = []
        for group in zip_longest(*files_by_ass):
            for f in group:
                if f is not None:
                    interleaved_files.append(f)

        rows = []
        for file in interleaved_files:
            with open(file, "rb") as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected["encoding"]
            text = raw_data.decode(encoding)
            if len(text.split()) < self.min_words:
                continue
            filename = file.stem
            assignment = file.parent.name
            if assignment == "Ass5" or assignment not in task_description:
                continue
            author = (
                filename
                if assignment != "Ass1" or "2006_" not in filename
                else filename.split("_")[-1]
            )
            metadata = {
                "filename": filename,
                "author": author,
                "topic": task_description.get(assignment, "Unknown"),
                "text": _preprocess_text(text),
                "year": "2006",
                "genre": "Essay",
                "century": 21,
            }
            rows.append(metadata)
        metadata_df = pd.DataFrame(rows)
        metadata_df.to_excel(base_dir / "file_metadata.xlsx", index=False)
        return metadata_df

    def load_dataset(self, base_dir: Path, dataset_type: str) -> pd.DataFrame:
        if not base_dir.exists():
            raise FileNotFoundError(
                f"Base directory {base_dir} does not exist. Current path: {os.getcwd()}"
            )

        metadata = None
        data = []
        path2metadata = base_dir / "file_metadata.xlsx"
        if dataset_type == "student_essays":
            if not path2metadata.exists():
                self.create_student_essays_metadata(base_dir)
            return pd.read_excel(path2metadata)

        if dataset_type in ["gutenberg", "custom"]:
            if not path2metadata.exists():
                raise FileNotFoundError(
                    f"Metadata file {path2metadata} does not exist. Current path: {os.getcwd()}"
                )
            metadata = pd.read_excel(path2metadata)

        for file in chain(base_dir.glob("*.txt"), base_dir.glob("*.csv")):
            if (
                dataset_type == "gutenberg"
                and "Complete_Works_of_William_Shakespeare" in file.name
            ):
                continue

            if file.suffix == ".csv" and dataset_type == "blog":
                df = pd.read_csv(file)
                df["text"] = df["text"].apply(_preprocess_text)
                df = df[df["text"].apply(lambda x: len(x.split()) >= self.min_words)]

                df["year"] = (
                    pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
                    .dt.year.fillna(0)
                    .astype(int)
                )
                df["century"] = df["year"].apply(self.get_century)

                return df

            if file.suffix == ".txt":
                with open(file, "r", encoding="utf-8") as f:
                    author = " ".join(file.stem.split("_")[-2:])
                    content = _preprocess_text(f.read())
                    if len(content.split()) >= self.min_words:
                        data.append(
                            {"author": author, "text": content, "filename": file.stem}
                        )

        if metadata is not None:
            df = pd.DataFrame(data)
            df = df.join(
                metadata.set_index("filename"),
                on="filename",
                how="inner",
                rsuffix="_meta",
            )
        else:
            df = pd.DataFrame(data)

        df.dropna(how="all", inplace=True)
        return df

    def obtain_complete_paraphrase_df_from_mongodb(self) -> pd.DataFrame:
        non_naive_paraphrases_cursor = self.mongodb.non_naive_paraphrase_collection.find({})
        non_naive_paraphrases = pd.DataFrame(non_naive_paraphrases_cursor)
        naive_paraphrases_cursor = self.mongodb.naive_paraphrase_collection.find({})
        naive_paraphrases = pd.DataFrame(naive_paraphrases_cursor)

        paraphrases = pd.concat([non_naive_paraphrases, naive_paraphrases], axis=0, ignore_index=True)
        extracted_df = pd.json_normalize(paraphrases["extracted_info"])

        paraphrases = pd.concat(
            [paraphrases.drop(columns=["extracted_info"]), extracted_df.drop(columns=["prompt"], errors="ignore")],
            axis=1,
        )
        dup_cols = paraphrases.columns[paraphrases.columns.duplicated()].tolist()
        paraphrases["text_id"] = paraphrases["text_id"].map(ObjectId)
        original_texts_cursor = self.mongodb.original_collection.find(
            {"_id": {"$in": paraphrases["text_id"].tolist()}},
            {"_id": True, "text": True, "dataset": True},
        )
        original_texts = pd.DataFrame(original_texts_cursor)

        df = paraphrases.merge(
            original_texts,
            left_on="text_id",
            right_on="_id",
            how="left",
        )
        df.rename(columns={"_id_x": "_id"}, inplace=True)
        df.drop(columns=["_id_y"], inplace=True)
        return df
