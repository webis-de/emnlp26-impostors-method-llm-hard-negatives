import logging
import os
from pathlib import Path

import pandas as pd
from bson import ObjectId

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[2]
    / CONFIG.SAVE_PATH
    / "impostors-approach"
)
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

def _obtain_complete_paraphrase_df_from_mongodb():
    mongodb = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    # contains paraphrase, prompt, temperature, text_id, extracted_info, length_original_text, length_paraphrased_text
    paraphrases_cursor = mongodb.non_naive_paraphrase_collection.find({})
    paraphrases = pd.DataFrame(paraphrases_cursor)
    # flatten extracted_info into columns
    extracted_df = pd.json_normalize(paraphrases["extracted_info"])

    # drop nested column and concat
    paraphrases = pd.concat(
        [paraphrases.drop(columns=["extracted_info"]), extracted_df],
        axis=1
    )

    # obtain original text via text_id (which is _id in original_texts collection)
    paraphrases["text_id"] = paraphrases["text_id"].map(ObjectId)
    original_texts_cursor = mongodb.original_collection.find(
        {"_id": {"$in": paraphrases["text_id"].tolist()}},  # $in matches any ID in the list
        {"_id": True, "text": True, "dataset": True},
    )
    original_texts = pd.DataFrame(original_texts_cursor)
    logger.info(f"Obtained {len(original_texts)} original texts from df with columns {original_texts.columns}")
    print(original_texts["dataset"].value_counts(dropna=False))

    # merge data on paraphrases' text_id and original_texts_cursor's _id field
    df = paraphrases.merge(
        original_texts,
        left_on="text_id",
        right_on="_id",
        how="left"
    )
    return df

def save_to_disk(df):
    logger.info(f"Save to path {LOCAL_SAVE_PATH}")
    # ---------------------------------------------------------------------
    # Save original texts and paraphrases
    # ---------------------------------------------------------------------

    original_dir = LOCAL_SAVE_PATH / "original_texts"
    paraphrased_dir = LOCAL_SAVE_PATH / "paraphrased_texts"

    original_dir.mkdir(parents=True, exist_ok=True)
    paraphrased_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Save unique original texts
    # -----------------------------
    unique_originals = (
        df[["text_id", "text", "dataset"]]
        .drop_duplicates(subset=["text_id"])
    )

    for _, row in unique_originals.iterrows():
        text_id = str(row["text_id"])
        text = row["text"]
        dataset = row["dataset"]

        file_path = original_dir / f"{dataset}_{text_id}.txt"
        file_path.write_text(text, encoding="utf-8")

    logger.info(f"Saved {len(unique_originals)} unique original texts.")

    # -----------------------------
    # Save paraphrased texts
    # -----------------------------
    for idx, row in df.iterrows():
        text_id = str(row["text_id"])
        paraphrase = row["paraphrase"]
        dataset = row["dataset"]

        # use row index as guaranteed-unique suffix
        file_name = f"{dataset}_{text_id}_{idx}.txt"
        file_path = paraphrased_dir / file_name

        file_path.write_text(paraphrase, encoding="utf-8")

    logger.info(f"Saved {len(df)} paraphrased texts That corresponds to around {len(df)//len(unique_originals)} paraphrases per original text.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    df = _obtain_complete_paraphrase_df_from_mongodb()
    save_to_disk(df)