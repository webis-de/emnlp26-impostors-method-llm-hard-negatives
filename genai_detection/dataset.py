from collections import Counter
import csv
import json
from pathlib import Path
import os
import sys
import types
import typing as t
import uuid

import click
from datasets import ClassLabel, concatenate_datasets, Dataset, DatasetDict, Features, load_dataset, Value
from cli.dataset import _load_dataset_and_adapt_schema, _load_jsonl, _load_json_single, create_dataset_split
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import json
from datasets import Dataset, DatasetDict

### PAN 25

DATASET_FEATURES = Features({
    'id': Value('string'),
    'text': Value('string'),
    'label': ClassLabel(names=['human', 'machine']),
    'model': Value('string'),
})

def read_txts_from_dir(directory, label, is_human, recursive=True):
    data = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(".txt"):
                continue
            path = Path(os.path.join(root, filename))
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            file_id = path.stem
            # file_id = os.path.splitext(filename)[0]
            # model = os.path.basename(os.path.dirname(path))
            parent_index = min(model_name_parent, len(path.parents) - 1)
            model = path.parents[parent_index].name if not is_human else 'human'
            data.append({
                "id": f"{path.parents[0].name}/{path.stem}",
                "text": text,
                "label": label,
                "model": model#p.parents[min(model_name_parent, len(p.parents) - 1)].name if not is_human else 'human'
            })
        if not recursive:
            break
    return data

def load_ids(path):
    with open(path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def build_dataset(human_dir, machine_dir, train_ids_path, test_ids_path, output_path="dataset_output"):
    train_ids = load_ids(train_ids_path)
    test_ids = load_ids(test_ids_path)

    print(list(train_ids)[:5])


    assert len(train_ids) > 0, "No training IDs found."
    assert len(test_ids) > 0, "No test IDs found."
    assert not train_ids.intersection(test_ids), "Training and test IDs must not overlap."
    assert os.path.isdir(human_dir), f"Human directory does not exist: {human_dir}"
    assert os.path.isdir(machine_dir), f"Machine directory does not exist: {machine_dir}"

    data = read_txts_from_dir(human_dir, is_human=True, label="human") + read_txts_from_dir(machine_dir, is_human=False, label="machine")

    label_counts = Counter(d["label"] for d in data)
    print("Label counts:", label_counts)

    model_counts = Counter(d["model"] for d in data)
    print("model counts:", model_counts)

    print(f"Loaded {len(data)} total records from directories.")
    # print(data[:5])  # Print first 5 records for debugging
    train_data, test_data = [], []
    print(f"{data[0]["id"]}" )
    for row in data:
        if row["id"] in test_ids:
            test_data.append(row)
        elif row["id"] in train_ids:
            train_data.append(row)
        # You can warn on "orphan" IDs here if needed
    print(f"Split into {len(train_data)} training and {len(test_data)} test records.")

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data)
    })

    dataset_dict.save_to_disk(output_path)
    print(f"Dataset saved to: {output_path}")


##### PAN 23


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def merge_data_and_labels(pairs, truth):
    truth_map = {item["id"]: item for item in truth}
    merged = []
    for example in pairs:
        ex_id = example["id"]
        label_data = truth_map.get(ex_id, {})
        merged.append({**example, **label_data})
    return merged

def load_dataset_from_directory(directory_path):
    pairs_path = os.path.join(directory_path, "pairs.jsonl")
    truth_path = os.path.join(directory_path, "truth.jsonl")
    
    pairs = load_jsonl(pairs_path)
    truth = load_jsonl(truth_path)

    merged_data = merge_data_and_labels(pairs, truth)
    return Dataset.from_list(merged_data)




# Example usage
if __name__ == "__main__":

    # PAN 25
    # is_human = True  # Change to False for machine dataset
    # human_basedir = "data/datasets/dataset-extended-2025-part/human" 
    # machine_basedir =  "data/datasets/dataset-extended-2025-part/machines" 
    # basedir = human_basedir if is_human else machine_basedir
    # test_ids = "data/datasets/dataset-extended-2025-part/ids-test.txt" 
    # recursive = True
    # model_name_parent = 1
    # val_split_size = 2000
    # output = "data/datasets/dataset-extended-2025-part-converted"
    # train_ids = "data/datasets/dataset-extended-2025-part/ids-train.txt"
    # test_ids = "data/datasets/dataset-extended-2025-part/ids-test.txt"

    # build_dataset(
    #     human_dir=human_basedir,
    #     machine_dir=machine_basedir,
    #     train_ids_path=train_ids,
    #     test_ids_path=test_ids,
    #     output_path=output
    # )

    # PAN 23
    # Define the paths
    base_dir = "data/datasets/pan23-authorship-verification/"
    train_dir = os.path.join(base_dir, "pan23-authorship-verification-training-dataset")
    test_dir = os.path.join(base_dir, "pan23-authorship-verification-test-dataset")

    # Load datasets
    train_dataset = load_dataset_from_directory(train_dir)
    test_dataset = load_dataset_from_directory(test_dir)

    # Combine into a DatasetDict for use with Hugging Face Transformers
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # Save to disk or explore
    dataset.save_to_disk(os.path.join(base_dir, "processed_pan23_dataset"))
    print(dataset["train"][0])