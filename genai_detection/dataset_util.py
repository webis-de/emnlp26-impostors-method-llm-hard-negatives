import argparse
from itertools import combinations
import os
import json
from pathlib import Path
from collections import Counter
from abc import ABC, abstractmethod
import random
import re
import sys
import unicodedata

from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
import pandas as pd
from tqdm import tqdm

# === BASE CLASS ===


class BaseDatasetLoader(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load(self) -> DatasetDict:
        pass

    @staticmethod
    def _load_jsonl(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _load_ids(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
        

# === Blog Corpus LOADER ===
        
class BlogCorpusDatasetLoader(BaseDatasetLoader):
    def __init__(self, path: str, name: str = "blog-corpus"):
        """Loader for the Blog Corpus dataset.
        
        Originally dataset is available at: https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus?resource=download (07.06.2025)
        """
        super().__init__(name=name)
        self.path = Path(path)

    def load(self) -> Dataset:
        df = pd.read_csv(self.path)
        topic_groups = df.groupby('topic')
        n_pairs = 5

        features = Features({
            "pair": [Value("string")],
            "authors": [Value("string")],
            "same": Value("bool")
        })

        all_pairs = []

        for topic, group in topic_groups:
            data = group.to_dict(orient='records')

            # Group texts by author id
            author_groups = {}
            for item in data:
                author_groups.setdefault(item['id'], []).append(item)

            # One same-author pair per topic (if possible)
            same_pair = None
            for author, texts in author_groups.items():
                if len(texts) < 2:
                    continue
                text_pairs = random.sample(texts, min(n_pairs*2, len(texts)))
                random.shuffle(text_pairs)   # avoid systematic bias
                
                for i in range(0, len(text_pairs) - 1, 2):
                    a = text_pairs[i]
                    b = text_pairs[i + 1]
                    all_pairs.append({
                        "pair": [a['text'], b['text']], # FIXME: maybe directly text
                        "authors": [author, author],#[a['id'], b['id']],
                        "same": True
                    })
              

               

            # One different-author pair per topic (if possible)
            diff_pair = None
            authors = list(author_groups.keys())
            if len(authors) > 1:
                authors = random.sample(authors, min(n_pairs*2, len(authors)))
                for i in range(0, len(authors) - 1, 2):
                    a1 = authors[i]
                    a2 = authors[i + 1]
                    if a1 == a2 or len(author_groups[a1]) < 1 or len(author_groups[a2]) < 1:
                        continue
                    # Randomly select one text from each author
                    t1 = random.choice(author_groups[a1])
                    t2 = random.choice(author_groups[a2])

                    all_pairs.append({
                        "pair": [t1['text'], t2['text']], # FIXME: maybe directly text
                        "authors": [a1, a2],
                        "same": False
                    })
                   

        return DatasetDict({"train": Dataset.from_list(all_pairs, features=features)})



# === Koppel Webis LOADER ===

class KoppelWebisDatasetLoader(BaseDatasetLoader):
    def __init__(self, path: str, name: str = "koppel-webis"):
        super().__init__(name=name)
        self.path = Path(path)

    def load(self) -> DatasetDict:
        data = []
        for author in self.path.iterdir():
            if author.is_dir():
                for file in author.iterdir():
                    if file.is_file() and file.suffix == '.txt':
                        with open(file, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                            content = unicodedata.normalize("NFKC", content)
                            content = re.sub(r"[^\x00-\x7F]+", " ", content)
                            content = re.sub(r"\s+", " ", content)
                            content = content.strip().lower()
                            data.append({'author': author.name, 'text': content})
        pairs = []
        for a, b in combinations(data, 2):
            pairs.append({
                "pair": [a["text"], b["text"]],
                "authors": [a["author"], b["author"]],
                "same": a["author"] == b["author"]
            })

        # Define the structure for Hugging Face datasets
        features = Features({
            "pair": [Value("string")],
            "authors": [Value("string")],
            "same": Value("bool")
        })
        return DatasetDict({"train": Dataset.from_list(pairs, features=features)})


# === PAN23 LOADER ===


class Pan23DatasetLoader(BaseDatasetLoader):
    def __init__(self, train_dir: str, test_dir: str, name: str="pan23"):
        super().__init__(name=name)
        self.train_dir = train_dir
        self.test_dir = test_dir

    def _load_dataset_from_directory(self, directory_path: str) -> Dataset:
        pairs = self._load_jsonl(os.path.join(directory_path, "pairs.jsonl"))
        path2truth = os.path.join(directory_path, "truth.jsonl")
        if not os.path.exists(path2truth):
            path2truth = os.path.join(directory_path.strip('/') + "-truth/", "truth.jsonl")
        truth = self._load_jsonl(path2truth)
        truth_map = {item["id"]: item for item in truth}

        merged_data = []
        for item in tqdm(pairs, desc=f"Processing {directory_path}"):
            merged_data.append({**item, **truth_map.get(item["id"], {})})
        return Dataset.from_list(merged_data)

    def load(self) -> DatasetDict:
        train = self._load_dataset_from_directory(self.train_dir)
        test = self._load_dataset_from_directory(self.test_dir)
        return DatasetDict(train=train, test=test)


#  === PAN20 LOADER ===

class Pan20DatasetLoader(Pan23DatasetLoader):
    def __init__(self, train_dir: str, test_dir: str):
        super().__init__(name="pan20", train_dir=train_dir, test_dir=test_dir)

    def _load_dataset_from_directory(self, directory_path: str) -> Dataset:
        pair_name = "pan20-authorship-verification-test.jsonl"
        truth_name = "pan20-authorship-verification-test-truth.jsonl"
        if not os.path.exists(os.path.join(directory_path, pair_name)):
            pair_name = "pan20-authorship-verification-training-small.jsonl"
            truth_name = "pan20-authorship-verification-training-small-truth.jsonl"
        pairs = self._load_jsonl(os.path.join(directory_path, pair_name))
        truth = self._load_jsonl(os.path.join(directory_path, truth_name))
        truth_map = {item["id"]: item for item in truth}

        merged_data = []
        for item in tqdm(pairs, desc=f"Processing {directory_path}"):
            merged_data.append({**item, **truth_map.get(item["id"], {})})
        return Dataset.from_list(merged_data)


# === PAN25 LOADER ===


class Pan25DatasetLoader(BaseDatasetLoader):
    def __init__(
        self,
        human_dir: str,
        machine_dir: str,
        train_ids_path: str,
        test_ids_path: str,
        model_name_parent: int = 1,
    ):
        super().__init__("pan25")
        self.human_dir = human_dir
        self.machine_dir = machine_dir
        self.train_ids_path = train_ids_path
        self.test_ids_path = test_ids_path
        self.model_name_parent = model_name_parent

    def _read_txts_from_dir(
        self, directory: str, label: str, is_human: bool, recursive: bool = True
    ):
        data = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if not filename.endswith(".txt"):
                    continue
                path = Path(os.path.join(root, filename))
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                file_id = f"{path.parents[0].name}/{path.stem}"
                model = (
                    path.parents[self.model_name_parent].name
                    if not is_human
                    else "human"
                )
                data.append(
                    {"id": file_id, "text": text, "label": label, "model": model}
                )
            if not recursive:
                break
        return data

    def load(self) -> DatasetDict:
        train_ids = self._load_ids(self.train_ids_path)
        test_ids = self._load_ids(self.test_ids_path)

        assert train_ids, "Train IDs list is empty."
        assert test_ids, "Test IDs list is empty."
        assert not train_ids.intersection(test_ids), "Train/Test ID sets overlap."

        data = self._read_txts_from_dir(
            self.human_dir, "human", is_human=True
        ) + self._read_txts_from_dir(self.machine_dir, "machine", is_human=False)

        print("Label counts:", Counter(d["label"] for d in data))
        print("Model counts:", Counter(d["model"] for d in data))

        train_data = [d for d in data if d["id"] in train_ids]
        test_data = [d for d in data if d["id"] in test_ids]

        print(
            f"Split into {len(train_data)} training and {len(test_data)} test records."
        )
        features = Features(
            {
                "id": Value("string"),
                "text": Value("string"),
                "label": ClassLabel(names=["human", "machine"]),
                "model": Value("string"),
            }
        )

        return DatasetDict(
            {
                "train": Dataset.from_list(train_data, features=features),
                "test": Dataset.from_list(test_data, features=features),
            }
        )


# === SYSTEM SPECIFIC USAGE ===


def run_pan23(base_dir:str, save_path:str):
    train_dir = os.path.join(base_dir, "pan23-authorship-verification-training-dataset")
    test_dir = os.path.join(base_dir, "pan23-authorship-verification-test-dataset")
    output_dir = os.path.join(save_path, "pan23-dataset-converted")

    loader = Pan23DatasetLoader(train_dir=train_dir, test_dir=test_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)
    # print("PAN23 example:", dataset["train"][0])

def run_pan20():
    base_dir = "data/datasets/pan20-authorship-verification/"
    train_dir = os.path.join(base_dir, "pan20-authorship-verification-training-dataset")
    test_dir = os.path.join(base_dir, "pan20-authorship-verification-test-dataset")
    output_dir = os.path.join(base_dir, "pan20-dataset-converted")

    loader = Pan20DatasetLoader(train_dir=train_dir, test_dir=test_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)
    # print("PAN20 example:", dataset["train"][0])


def run_pan25():
    base_dir = "data/datasets/dataset-extended-2025-part/"
    human_dir = os.path.join(base_dir, "human")
    machine_dir = os.path.join(base_dir, "machines")
    train_ids_path = os.path.join(base_dir, "ids-train.txt")
    test_ids_path = os.path.join(base_dir, "ids-test.txt")
    output_dir = os.path.join(base_dir, "pan25-dataset-converted")

    loader = Pan25DatasetLoader(human_dir, machine_dir, train_ids_path, test_ids_path)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)
    # print("PAN25 example:", dataset["train"][0])

def run_koppel_webis():
    base_dir = "data/datasets/corpus-webis-authorship/koppel/"
    output_dir = os.path.join(base_dir, "koppel-webis-dataset-converted")

    loader = KoppelWebisDatasetLoader(path=base_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)
    # print("Koppel Webis example:", dataset["train"][0])

def run_blog_corpus():
    sys.path.append(os.path.abspath(".."))
    base_dir = Path("data/datasets/Blog_corpus/")
    output_dir = base_dir / "blog-dataset-converted"

    loader = BlogCorpusDatasetLoader(path=base_dir / "blogtext.csv")
    dataset = loader.load()
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate classifiers on Curlie dataset and save metrics to CSV/Parquet."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="data/datasets/pan23-authorship-verification/",
        help="Path to the input dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/datasets/pan23-authorship-verification/",
        help="Path where Huggingface dataset should be saved (default: %(default)s)",
    )
   
    args = parser.parse_args()

    # run_pan23(base_dir=args.path, save_path=args.out)
    # run_pan25()
    # run_pan20()
    # run_koppel_webis()
    run_blog_corpus()
