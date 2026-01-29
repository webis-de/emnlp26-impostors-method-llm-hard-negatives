# scripts/check_identical_pairs.py (or run in a python shell)
import os
from pathlib import Path

from bson import ObjectId
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.config import CONFIG

mongo = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
dataset = CONFIG.BLOG  # or CONFIG.STUDENT_ESSAYS

# 1) Pairs where left_id == right_id
same_id = list(mongo.all_pairs_collection.find(
    {"dataset_name": dataset, "$expr": {"$eq": ["$left_id", "$right_id"]}},
    {"left_id": 1, "right_id": 1, "same": 1}
))
print("Pairs with left_id == right_id:", len(same_id))

# 2) Pairs where text content is identical
pairs = list(mongo.all_pairs_collection.find(
    {"dataset_name": dataset},
    {"left_id": 1, "right_id": 1, "dataset_name": dataset}
))
ids = {p["left_id"] for p in pairs} | {p["right_id"] for p in pairs}
texts = mongo.original_collection.find({"_id": {"$in": list(ids)}}, {"text": 1})
text_map = {d["_id"]: d["text"] for d in texts}

identical_text_pairs = []
for p in pairs:
    l_id = p["left_id"]
    r_id = p["right_id"]
    lt = text_map.get(l_id)
    rt = text_map.get(r_id)
    if lt is not None and rt is not None and lt == rt:
        identical_text_pairs.append([l_id, r_id, p["dataset_name"]])
if len(identical_text_pairs) > 0:
    print(f"{len(identical_text_pairs)} pairs with identical text:")
    print("Pairs with identical text:", identical_text_pairs)
    dataset_name = CONFIG.BLOG
    mongo.delete_identical_text_pairs(dataset_name=dataset_name, output_path=Path(CONFIG.SAVE_PATH) / "datasets" / dataset_name / "duplicate_texts.csv", batch_size=1000)
else:
    print(f"No pairs with identical text.")

