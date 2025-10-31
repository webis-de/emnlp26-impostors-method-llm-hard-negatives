import json
from datetime import datetime
from pathlib import Path

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.paraphrasing import two_step_paraphrasers
from genai_detection.paraphrasing.openai_utils import OpenaiCostEstimator

# MongoDB setup
mongoDB = ParaphraseMongoDB()
original_collection = mongoDB.original_collection
paraphrase_collection = mongoDB.paraphrase_collection

# 30.10.25, morning: $7,891.29 balance left on OpenAI account
# 30.10.25, evening: $7,891.28 balance left on OpenAI account
# 31.10.25, morning: $7,891.28 balance left on OpenAI account
# 31.10.25, evening: $7,891.27 balance left on OpenAI account

# Iterate over all documents in original_text
paraphraser_llm = two_step_paraphrasers.TwoStepParaphraser(model_id=CONFIG.OPENAI_MODEL)

# TODO: delete following line
max_docs = 5  # for testing, limit to first 5 documents
openai_cost_est = OpenaiCostEstimator()

save_dir = (
        Path(__file__).resolve().parent.parent.parent
        / CONFIG.SAVE_PATH
        / "openai_paraphrases"
)
save_dir.mkdir(parents=True, exist_ok=True)


def save_original_text(doc_n: int, text: str):
    """
    Save original text to file.
    :param doc_n: The number of the document. This is used to create unique filenames.
    :param text: The original text to save.
    """
    original_text_filename = (
        f"{doc_n}_original_text_{CONFIG.OPENAI_MODEL.replace('/', '_')}.txt"
    )
    with open(save_dir / original_text_filename, "w", encoding="utf-8") as f:
        f.write(text)


def save_extracted_info(doc_n: int, extracted_info_dict: dict):
    """
    Save extracted information to file.
    :param doc_n: Number of the document. This is used to create unique filenames.
    """
    extr_info_filename = (
        f"{doc_n}_extracted_information_{CONFIG.OPENAI_MODEL.replace('/', '_')}.jsonl"
    )
    with open(save_dir / extr_info_filename, "w", encoding="utf-8") as f:
        json.dump(extracted_info_dict, f, ensure_ascii=False, indent=4)


def save_paraphrase(doc_n: int, paraphrase: str):
    """
    Save paraphrase to file.
    :param doc_n: Number of the document. This is used to create unique filenames.
    :param paraphrase: The paraphrased text to save.
    """
    paraphrase_filename = (
        f"{doc_n}_paraphrase_{CONFIG.OPENAI_MODEL.replace('/', '_')}.txt"
    )
    with open(save_dir / paraphrase_filename, "w", encoding="utf-8") as f:
        f.write(paraphrase)


for i, doc in enumerate(original_collection.find(limit=max_docs)):
    text_id = doc["_id"]
    original_text = doc["text"]
    existing_paraphrase = mongoDB.find_document(
        collection=paraphrase_collection, document_id=text_id
    )
    if existing_paraphrase:
        extracted_info = existing_paraphrase.get("extracted_info", {})
        # skip the document if already has a paraphrase
        if existing_paraphrase.get("paraphrase", False):
            print(f"Skipping document ID: {text_id} as it already has a paraphrase.")
            continue
        elif extracted_info:  # != {}
            print(
                f"Generate paraphrase based on extracted data from document with document ID: {text_id}."
            )
            paraphrased_text, total_costs = paraphraser_llm.generate_paraphrase(
                extracted_info=extracted_info
            )
            update_fields = {
                "paraphrase": paraphrased_text,
                "openai_costs": total_costs,
                "updated_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            }

            # Update the document in place
            modified_count = mongoDB.update_document(
                text_id=text_id,
                collection=paraphrase_collection,
                update_data=update_fields,
            )

            if modified_count > 0:
                print(f"Updated document {text_id} with new paraphrase.")
                save_original_text(doc_n=i, text=original_text)
                save_paraphrase(doc_n=i, paraphrase=paraphrased_text)
                assert (
                        type(paraphrased_text) == str
                ), f"paraphrased_text must be a str, but is of type {type(paraphrased_text)}"
                assert len(paraphrased_text.split()) > 0, "paraphrased_text is empty"
            else:
                print(f"No update performed for {text_id} (may already be up-to-date).")
            continue

    # prompt is not used, bc we use dspy
    extracted_info, paraphrased_text, total_costs = paraphraser_llm.paraphrase(
        text=original_text
    )

    # save results to files for manual inspection
    save_original_text(doc_n=i, text=original_text)
    save_paraphrase(doc_n=i, paraphrase=paraphrased_text)
    save_extracted_info(doc_n=i, extracted_info_dict=extracted_info)

    assert (
            type(paraphrased_text) == str
    ), f"paraphrased_text must be a str, but is of type {type(paraphrased_text)}"
    assert len(paraphrased_text.split()) > 0, "paraphrased_text is empty"

    # Build the new document
    paraphrase_doc = {
        "_id": text_id,
        "length_original_text": len(original_text.split()),
        "length_paraphrased_text": len(paraphrased_text.split()),
        "intermediate_prompt": "bullet points dspy",
        "prompt": "bullet points dspy",  # CONFIG.PROMPT,
        "llm": CONFIG.OPENAI_MODEL,
        "temperature": 1.0,  # requirements for reasoning models like gpt-5-nano
        "paraphrase": paraphrased_text,
        "extracted_info": extracted_info,
        "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "openai_costs": total_costs,
    }

    # Insert into document into paraphrase collection
    paraphrase_collection.insert_one(paraphrase_doc)
    print(f"Inserted paraphrase for document ID: {text_id}")

print(f"Paraphrasing complete. Inserted/Updated {max_docs} documents.")
