import argparse
from itertools import combinations, product
import os
import json
from pathlib import Path
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
import random
import re
import sys
import unicodedata
import typing as t

from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CONFIG
from genai_detection.util import preprocess_text as _preprocess_text

random.seed(42)

# === BASE CLASS ===


class BaseDatasetLoader(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load(self) -> DatasetDict:
        pass

    def preprocess(self, text: t.Union[str, t.Iterable[str]]) -> t.Union[str, t.List[str]]:
        return _preprocess_text(text=text)
        

    @staticmethod
    def _load_jsonl(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _load_ids(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
        
    def _generate_pairs(self, data):
        """
        Generate pairs of texts from the dataset irrespective of confounders (i.e. naive combinations).
        Dataset is expected to be a list of dictionaries with 'text' and 'author' keys.
        Each pair consists of two texts, their authors, and a boolean indicating if they are from the same author.
        """
        pairs = []
        for a, b in combinations(data, 2):
            pairs.append({
                "pair": [a["text"], b["text"]],
                "authors": [a["author"], b["author"]],
                "same": a["author"] == b["author"]
            })
            
        return pairs
        

# === Blog Corpus LOADER ===
        
class BlogCorpusDatasetLoader(BaseDatasetLoader):
    def __init__(self, path: str, name: str = CONFIG.BLOG):
        """Loader for the Blog Corpus dataset.
        Contains blog posts with dates from 01 January 1999 to 23 August 2006.
        When pairing texts, it is important to control confounders (i.e. pair similar external situations).
        Confounders can be:
        - topic 
        - time period (e.g. 1999 vs. 2006)
        - age (?!)
        - gender (?!)
        
        Originally dataset is available at: https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus?resource=download (07.06.2025)
        """
        super().__init__(name=name)
        self.path = Path(path)

    def load(self) -> Dataset:
        df = pd.read_csv(self.path)
        print("Initial number of entries:", len(df))
        df['text'] = df['text'].apply(lambda x: self.preprocess(x))
        df = df[df['text'].apply(lambda x: len(x.split()) >= 500)]  # filter out text with less than 500 words (not characters, bc there are 501 characters one-word entries)

        df['year'] = pd.to_datetime(df["date"], format='mixed', dayfirst=True, errors='coerce').dt.year
        print("number of entries after filtering:", len(df))
      
        topics = df['topic'].unique().tolist()
        random.shuffle(topics)

        split_idx = int(0.8 * len(topics))
        train_topics = set(topics[:split_idx])
        test_topics = set(topics[split_idx:])

        def generate_pairs(topic_subset, n_pairs=2, groupby_cols:list = ['topic']):
            """ 
            Generate pairs of texts from the dataset based on the specified topic subset.
            :param topic_subset: Set of topics to filter the dataset.
            :param n_pairs: Number of pairs to generate per group.
            :param groupby_cols: Columns to group by, should include 'topic'.
            :return: List of pairs with their authors and a boolean indicating if they are from the
            same author.
            """
            topic_df = df[df['topic'].isin(topic_subset)]
            assert 'topic' in groupby_cols, "The 'topic' should be one of the columns to group by."


            grouped = topic_df.groupby(groupby_cols)
            pairs = []

            for group_values, group in grouped:
                data = group.to_dict(orient='records')

                # Group texts by author
                author_groups = {}
                for item in data:
                    author_groups.setdefault(item['id'], []).append(item)

                # Same-author pairs
                same_author_pairs = []
                for author, texts in author_groups.items():
                    if len(texts) < 2:
                        continue
                    selected = random.sample(texts, min(n_pairs*2, len(texts)))
                    random.shuffle(selected)
                    for i in range(0, len(selected) - 1, 2):
                        a, b = selected[i], selected[i + 1]
                        same_author_pairs.append({
                            "pair": [a['text'], b['text']],
                            "authors": [author, author],
                            "same": True
                        })
                pairs.extend(same_author_pairs)

                # Different-author pairs, balanced to same author pairs count
                authors = list(author_groups.keys())
                if len(authors) > 1 and same_author_pairs:
                    n_diff_pairs_target = len(same_author_pairs)    # goal: match number of different-author pairs to same-author pairs
                    author_pairs = []
                    for i in range(len(authors)):
                        for j in range(i+1, len(authors)):
                            author_pairs.append((authors[i], authors[j]))
                    random.shuffle(author_pairs)

                    count = 0
                    for a1, a2 in author_pairs:
                        if count >= n_diff_pairs_target:    # generate enough different-author pairs
                            break
                        if not author_groups[a1] or not author_groups[a2]:  # ensured above that authors are different
                            continue
                        t1 = random.choice(author_groups[a1])
                        t2 = random.choice(author_groups[a2])
                        pairs.append({
                            "pair": [t1['text'], t2['text']],
                            "authors": [a1, a2],
                            "same": False
                        })
                        count += 1

            return pairs
        

        features = Features({
            "pair": [Value("string")],
            "authors": [Value("string")],
            "same": Value("bool")
        })

        # define the columns to group by
        groupby_cols=['topic', 'year', 'gender', 'age']
        train_pairs = generate_pairs(train_topics, groupby_cols=groupby_cols)
        test_pairs = generate_pairs(test_topics, groupby_cols=groupby_cols)

        return DatasetDict({
            "train": Dataset.from_list(train_pairs, features=features),
            "test": Dataset.from_list(test_pairs, features=features)
        })



# === Koppel Webis LOADER ===

class KoppelWebisDatasetLoader(BaseDatasetLoader):
    def __init__(self, path: str, name: str = CONFIG.KOPPEL):
        super().__init__(name=name)
        self.path = Path(path)
        assert self.path.exists(), f"Path {self.path} does not exist. Current path: {os.getcwd()}"

    def load(self) -> DatasetDict:
        data = []
        for author in self.path.iterdir():
            if author.is_dir():
                for file in author.iterdir():
                    if file.is_file() and file.suffix == '.txt':
                        with open(file, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                            content = self.preprocess(content)
                            data.append({'author': author.name, 'text': content})
        pairs = self._generate_pairs(data)

        # Define the structure for Hugging Face datasets
        features = Features({
            "pair": [Value("string")],
            "authors": [Value("string")],
            "same": Value("bool")
        })
        return DatasetDict({"train": Dataset.from_list(pairs, features=features)})


# === PAN23 LOADER ===

class Pan23DatasetLoader(BaseDatasetLoader):
    def __init__(self, train_dir: str, test_dir: str, name: str=CONFIG.PAN23):
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
        """
        Loader for the PAN 2020 Authorship Verification dataset about fanfiction.
        The dataset is available at: https://zenodo.org/records/5106099 (08.06.2025)
        
        References:
        =========
        Sebastian Bischoff, Niklas Deckers, Marcel Schliebs, Ben Thies, Matthias Hagen, Efstathios Stamatatos, Benno Stein, and Martin Potthast. The Importance of Suppressing Domain Style in Authorship Analysis. CoRR, abs/2005.14714, May 2020.
        """
        super().__init__(name=CONFIG.PAN20, train_dir=train_dir, test_dir=test_dir)

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
        
        merged_data = pd.DataFrame(merged_data)
        merged_data['pair'] = merged_data['pair'].apply(
            lambda pair: [self.preprocess(text) for text in pair]
        )
        # Keep only pairs with at least 500 words in each text, bc there are one-word entries
        merged_data = merged_data[
            merged_data['pair'].apply(
                lambda pair: all(len(text.split()) >= 500 for text in pair)
            )
        ]

        return Dataset.from_list(merged_data.to_dict(orient='records'))


# === PAN24 LOADER ===

class Pan24DatasetLoader(BaseDatasetLoader):
    def __init__(self, train_dir: str, test_dir: str, name: str = "pan24"):
        """
        Loader for the PAN 2024 Authorship Verification dataset.
        Dataset download at https://zenodo.org/records/10718757 (08.06.2025), restricted access.
        The dataset contains human and machine-generated texts on news articles.
        """
        super().__init__(name=name)
        self.train_dir = train_dir
        self.test_dir = test_dir

    def load(self) -> DatasetDict:
        # TODO: Implement the loading logic for PAN24 dataset
        pass



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
        """
        Loader for the PAN 2025 Authorship Verification dataset.
        More information found at https://pan.webis.de/clef25/pan25-web/generated-content-analysis.html (08.06.2025)
        The dataset from subtask 1 contains human and machine-generated texts, with IDs for training and testing.
        """
        super().__init__(CONFIG.PAN25)
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

# === Gutenberg LOADER ===

class GutenbergDatasetLoader(BaseDatasetLoader):
    def __init__(self, path: str, name: str = CONFIG.GUTENBERG):
        super().__init__(name=name)
        self.path = Path(path)
        assert self.path.exists(), f"Path {self.path} does not exist. Current path: {os.getcwd()}"
        self.path2metadata = self.path / "file_metadata.xlsx"
        assert self.path2metadata.exists(), f"Metadata file {self.path2metadata} does not exist. Current path: {os.getcwd()}"

    def generate_pairs(self, df, n_pairs=2, groupby_cols:list = ['genre']):
            """ 
            Generate pairs of texts from the dataset based on the specified groupby columns.
            :param df: DataFrame containing the dataset with at least 'text' and 'author' columns.
            :param n_pairs: Number of pairs to generate per group.
            :param groupby_cols: Columns to group by, should include 'genre'.
            :return: List of pairs with their authors and a boolean indicating if they are from the
            same author.
            """
            for col in groupby_cols:
                assert col in df.columns, f"Column '{col}' not found in DataFrame."

            for col in groupby_cols:
                num_nans = df[col].isna().sum()
                if num_nans > 0:
                    print(f"Column '{col}' has {num_nans} NaN values.")
                    print(f"Rows with NaN in '{col}':\n{df[df[col].isna()]}\n")


            grouped = df.groupby(groupby_cols)
            # print(f"\nTotal groups: {len(grouped)}")
            # print(f"Groups: {list(grouped.groups.keys())}\n\n")
            pairs = []

            for group_values, group in grouped:
                print(f"Processing group: {group_values}, size: {len(group)}")
                data = group.to_dict(orient='records')

                # Group texts by author
                author_groups = {}
                for item in data:
                    author_groups.setdefault(item['author'], []).append(item)

                # Same-author pairs
                same_author_pairs = []
                for author, texts in author_groups.items():
                    if len(texts) < 2:
                        continue
                    selected = random.sample(texts, min(n_pairs*2, len(texts)))
                    random.shuffle(selected)
                    for i in range(0, len(selected) - 1, 2):
                        a, b = selected[i], selected[i + 1]
                        same_author_pairs.append({
                            "pair": [a['text'], b['text']],
                            "authors": [author, author],
                            "same": True
                        })
                pairs.extend(same_author_pairs)

                # Different-author pairs, balanced to same author pairs count
                authors = list(author_groups.keys())
                if len(authors) > 1 and same_author_pairs:
                    n_diff_pairs_target = len(same_author_pairs)    # goal: match number of different-author pairs to same-author pairs
                    author_pairs = []
                    for i in range(len(authors)):
                        for j in range(i+1, len(authors)):
                            author_pairs.append((authors[i], authors[j]))
                    random.shuffle(author_pairs)

                    count = 0
                    # Calculate max number of pairs to sample per author pair (reduce class (i.e. different-author) imbalance introdoced prior when only one pair per author pair was sampled)
                    max_pairs_per_pair = max(n_diff_pairs_target // len(author_pairs), 1)
                    for a1, a2 in author_pairs:
                        texts_a1 = author_groups[a1]
                        texts_a2 = author_groups[a2]

                        if not texts_a1 or not texts_a2:
                            continue

                        # All possible combinations between texts from different authors
                        all_combinations = list(product(texts_a1, texts_a2))
                        random.shuffle(all_combinations)  # Shuffle to introduce randomness

                        num_to_sample = min(len(all_combinations), max_pairs_per_pair)
                        for t1, t2 in all_combinations[:num_to_sample]:
                            pairs.append({
                                "pair": [t1['text'], t2['text']],
                                "authors": [a1, a2],
                                "same": False
                            })
                            count += 1

                        if count >= n_diff_pairs_target:
                            break

            return pairs
    
    def load(self, train_split_portion:float=0.8) -> Dataset:
        """
        Loader for the Gutenberg dataset.
        The dataset is expected to be a directory with text files, where each file is named in the format "title_firstName_sirname.txt".
        Each file contains the text of a book, and the author is derived from the filename.

        Authors cannot appear in both training and test sets.
        """
        assert train_split_portion > 0 and train_split_portion < 1, "train_split_portion must be between 0 and 1."
        data = []
        # obtain texts
        for file in self.path.glob("*.txt"):
            if "Complete_Works_of_William_Shakespeare" in file.name:
                # Skip the complete works of Shakespeare as it is way longer than other texts
                continue
            with open(file, "r", encoding="utf-8") as f:
                author = ' '.join(file.stem.split("_")[-2:])  # filename format is "title_firstName_sirname.txt"
                content = f.read()
                content = self.preprocess(content)
                data.append({"author": author, "text": content, "filename":file.stem})

        df = pd.DataFrame(data)
        # obtain metadata: contains time_period, author, topic (incomplete), summary, genre, century etc.
        metadata = pd.read_excel(self.path2metadata)
        # join on data's filename column and metadata's index (always 'others' index)
        df = df.join(metadata.set_index('filename'), on='filename', how='left', rsuffix='_meta')

        groupyby_cols = ['genre', 'century']
        assert all(col in df.columns for col in groupyby_cols), "Missing required metadata columns."

        # Group authors by (genre, century)
        author_meta = df.groupby('author').first().reset_index()    # keep only first occurrence of each author
        group_map = defaultdict(list)  # {(genre, century): [author1, author2, ...]}
        for _, row in author_meta.iterrows():
            key = (row['genre'], row['century'])
            group_map[key].append(row['author'])

        # Shuffle and split groups
        all_groups = list(group_map.items())
        random.shuffle(all_groups)

        train_authors = set()
        test_authors = set()
        train_count = 0
        total_authors = sum(len(authors) for _, authors in all_groups)
        author_limit = int(train_split_portion * total_authors)

        for key, authors in all_groups:
            if train_count + len(authors) <= author_limit:  # adds multiple authors from the same group at once
                train_authors.update(authors)
                train_count += len(authors)
            else:
                test_authors.update(authors)

        # Final sanity check
        assert train_authors.isdisjoint(test_authors), "Author overlap between train and test."
        train_df = df[df['author'].isin(train_authors)].sample(frac=1, random_state=42).reset_index(drop=True)
        test_df = df[df['author'].isin(test_authors)].sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Train authors: {train_authors}, Test authors: {test_authors}\n\n")

        # TODO: add similarity on summary sbert?
        train_pairs = self.generate_pairs(df=train_df, groupby_cols=groupyby_cols)
        test_pairs = self.generate_pairs(df=test_df, groupby_cols=groupyby_cols)
        print(f"Generated {len(train_pairs)} training pairs and {len(test_pairs)} test pairs.")
        # print("training pairs:", [train_pairs[i]['authors'] for i in range(len(train_pairs))]if train_pairs else "No training pairs generated.")
        # print("test pairs:", [test_pairs[i]['authors'] for i in range(len(test_pairs))] if test_pairs else "No test pairs generated.")

        features = Features({
            "pair": [Value("string")],
            "authors": [Value("string")],
            "same": Value("bool")
        })

        return DatasetDict({
            "train": Dataset.from_list(train_pairs, features=features),
            "test": Dataset.from_list(test_pairs, features=features),
        })


# === SYSTEM SPECIFIC USAGE ===
def run_pan23(base_dir:str, save_path:str):
    base_dir = Path(__file__).resolve().parent / base_dir
    train_dir = os.path.join(base_dir, "pan23-authorship-verification-training-dataset")
    test_dir = os.path.join(base_dir, "pan23-authorship-verification-test-dataset")
    output_dir = os.path.join(save_path, "pan23-dataset-converted")

    loader = Pan23DatasetLoader(train_dir=train_dir, test_dir=test_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)
    # print("PAN23 example:", dataset["train"][0])


def run_pan20():
    base_dir = Path(__file__).resolve().parent.parent / "data/datasets/pan20-authorship-verification/"
    train_dir = os.path.join(base_dir, "pan20-authorship-verification-training-dataset")
    test_dir = os.path.join(base_dir, "pan20-authorship-verification-test-dataset")
    output_dir = os.path.join(base_dir, "pan20-dataset-converted")

    loader = Pan20DatasetLoader(train_dir=train_dir, test_dir=test_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)
    # print("PAN20 example:", dataset["train"][0])


def run_pan25():
    base_dir = Path(__file__).resolve().parent / "data/datasets/dataset-extended-2025-part/"
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
    base_dir = Path(__file__).resolve().parent.parent / "data/datasets/corpus-webis-authorship/koppel/"
    output_dir = os.path.join(base_dir, "koppel-webis-dataset-converted")

    loader = KoppelWebisDatasetLoader(path=base_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)
    # print("Koppel Webis example:", dataset["train"][0])


def run_blog_corpus():
    # sys.path.append(os.path.abspath(".."))
    base_dir = Path(__file__).resolve().parent.parent / "data/datasets/Blog_corpus/"
    assert base_dir.exists(), f"Base directory {base_dir} does not exist. Current path: {os.getcwd()}"
    output_dir = base_dir / "blog-dataset-converted"

    loader = BlogCorpusDatasetLoader(path=base_dir / "blogtext.csv")
    dataset = loader.load()
    dataset.save_to_disk(output_dir)

def run_gutenberg_corpus():
    # sys.path.append(os.path.abspath(".."))
    base_dir = Path(__file__).resolve().parent.parent / "data/datasets/gutenberg/"
    assert base_dir.exists(), f"Base directory {base_dir} does not exist. Current path: {os.getcwd()}"
    output_dir = base_dir / "gutenberg-dataset-converted"

    loader = GutenbergDatasetLoader(path=base_dir)
    dataset = loader.load()
    dataset.save_to_disk(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Dataset creation."
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
    # # run_pan25()
    # run_pan20()
    # run_koppel_webis()
    run_blog_corpus()
    # run_gutenberg_corpus()
