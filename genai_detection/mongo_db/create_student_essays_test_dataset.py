# requirements:
# - 50 % same and 50 % different author pairs
# - pair <X,Y>: X, Y different subgenres
# (- 2000 pairs)
# (- 500 words)
import logging
import os
import random
from itertools import product

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s", )


mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
ass3_cursor = mongoDB.find_document_by_non_id_field(collection=mongoDB.original_collection,
                                                    document_field_name="assignment", document_value="Ass3")
ass3_docs = list(ass3_cursor)
authors_of_ass3 = {doc.get("author") for doc in ass3_docs}
ass4_cursor = mongoDB.find_document_by_multiple_fields(collection=mongoDB.original_collection,
                                                    search_args={"assignment":"Ass4", "author": {"$in": list(authors_of_ass3)}})
logging.info(f"Found {len(authors_of_ass3)} unique authors.")

ass4_docs = list(ass4_cursor)

logging.info(f"Found {len(authors_of_ass3)} unique authors.")

# --- Drop old collection ---
mongoDB.reset_collection(collection_name=CONFIG.MONGO_TEST_PAIRS_COLLECTION)

# --- Organize by author ---
ass3_by_author = {}
ass4_by_author = {}

for d in ass3_docs:
    ass3_by_author.setdefault(d["author"], []).append(d)

for d in ass4_docs:
    ass4_by_author.setdefault(d["author"], []).append(d)

# --- Build SAME-AUTHOR pairs ---
same_author_pairs = []

for author in authors_of_ass3:
    left_docs = ass3_by_author.get(author, [])
    right_docs = ass4_by_author.get(author, [])

    # TODO: do not save assignments, bc not applicable to other datasets
    # Create all cross combinations for same author
    for left, right in product(left_docs, right_docs):
        same_author_pairs.append({
            "left_id": left["_id"],
            "right_id": right["_id"],
            "left_author": left["author"],
            "right_author": right["author"],
            "left_assignment": left["assignment"],
            "right_assignment": right["assignment"],
            "same": True,
            "dataset_name": CONFIG.STUDENT_ESSAYS
        })

# --- Build DIFFERENT-AUTHOR pairs ---
different_author_pairs = []

all_authors = list(authors_of_ass3)

for left in ass3_docs:
    # authors different from left
    negative_authors = [a for a in all_authors if a != left["author"]]

    # TODO: do not save assignments, bc not applicable to other datasets
    # choose a random negative author for the right side
    for neg_author in negative_authors:
        if neg_author in ass4_by_author:
            for right in ass4_by_author[neg_author]:
                different_author_pairs.append({
                    "left_id": left["_id"],
                    "right_id": right["_id"],
                    "left_author": left["author"],
                    "right_author": right["author"],
                    "left_assignment": left["assignment"],
                    "right_assignment": right["assignment"],
                    "same": False,
                    "dataset_name": CONFIG.STUDENT_ESSAYS
                })

# --- Balance: half same-author, half different-author ---
# TODO: try with more data
min_pairs = min(len(same_author_pairs), len(different_author_pairs), 2)
random.shuffle(same_author_pairs)
random.shuffle(different_author_pairs)

balanced_pairs = same_author_pairs[:min_pairs] + different_author_pairs[:min_pairs]
random.shuffle(balanced_pairs)

logging.info(f"Created {len(balanced_pairs)} balanced pairs ({min_pairs} same, {min_pairs} different).")

mongoDB.insert_documents(collection=mongoDB.test_pairs_collection, insert_data=balanced_pairs)
