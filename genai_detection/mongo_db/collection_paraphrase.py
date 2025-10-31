import json
from datetime import datetime
from pathlib import Path

from pymongo import MongoClient
from genai_detection.config import CONFIG
from genai_detection.paraphrasing import paraphraser
from genai_detection.paraphrasing.openai_utils import OpenaiCostEstimator
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

# MongoDB setup
mongoDB = ParaphraseMongoDB()
original_collection = mongoDB.original_collection
paraphrase_collection = mongoDB.paraphrase_collection

# 30.10.25, morning: $7,891.29 balance left on OpenAI account
# 30.10.25, evening: $7,891.28 balance left on OpenAI account
# 31.10.25, morning: $7,891.28 balance left on OpenAI account

# Iterate over all documents in original_text
paraphraser_llm = paraphraser.OpenAIParaphraser_dspy(model_id=CONFIG.OPENAI_MODEL)

# TODO: delete following line
max_docs = 2  # for testing, limit to first 5 documents
openai_cost_est = OpenaiCostEstimator()
i = 0

save_dir = Path(CONFIG.SAVE_PATH) / "openai_paraphrases"
save_dir.mkdir(parents=True, exist_ok=True)

def save_original_text(i:int, original_text:str):
    original_text_filename = f"{i}_original_text_{CONFIG.OPENAI_MODEL.replace('/', '_')}.txt"
    with open(save_dir / original_text_filename, "w", encoding="utf-8") as f:
        f.write(original_text)

def save_extracted_info(i:int, extracted_info:dict):
    extr_info_filename = f"{i}_extacted_information_{CONFIG.OPENAI_MODEL.replace('/','_')}.jsonl"
    with open(save_dir / extr_info_filename, "w", encoding="utf-8") as f:
        json.dump(extracted_info, f, ensure_ascii=False, indent=4)

def save_paraphrase(i: int, paraphrased_text: str):
    paraphrase_filename = f"{i}_paraphrase_{CONFIG.OPENAI_MODEL.replace('/', '_')}.txt"
    with open(save_dir / paraphrase_filename, "w", encoding="utf-8") as f:
        f.write(paraphrased_text)

for doc in original_collection.find(limit=max_docs):
    text_id = doc["_id"]
    original_text = doc["text"]
    existing_paraphrase = mongoDB.find_document(collection=paraphrase_collection, id=text_id)
    if existing_paraphrase:
        extracted_info = existing_paraphrase.get("extracted_info", False)
        # skip the document if already has a paraphrase
        if existing_paraphrase.get("paraphrase", False):
            print(f"Skipping document ID: {text_id} as it already has a paraphrase.")
            continue
        elif extracted_info:
            print(f"Generate paraphrase based on extracted data from document with document ID: {text_id}.")
            assert type(extracted_info) == dict, f"extracted_info must be a dict, but is of type {type(extracted_info)}"
            paraphrased_text, total_costs = paraphraser_llm.generate_paraphrase(extracted_info=extracted_info)
            update_fields = {
                "paraphrase": paraphrased_text,
                "openai_costs": total_costs,
                "updated_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            }

            # Update the document in place
            modified_count = mongoDB.update_document(text_id=text_id, collection=paraphrase_collection,
                                                     update_data=update_fields)

            if modified_count > 0:
                print(f"Updated document {text_id} with new paraphrase.")
                save_original_text(i=i, original_text=original_text)
                save_paraphrase(i=i, paraphrased_text=paraphrased_text)
                assert type(
                    paraphrased_text) == str, f"paraphrased_text must be a str, but is of type {type(paraphrased_text)}"
                assert len(paraphrased_text.split()) > 0, "paraphrased_text is empty"
            else:
                print(f"No update performed for {text_id} (may already be up-to-date).")


    # prompt is not used, bc we use dspy
    extracted_info, paraphrased_text, total_costs = paraphraser_llm.paraphrase(text=original_text, prompt=CONFIG.PROMPT)

    # save results to files for manual inspection
    save_original_text(i=i, original_text=original_text)
    save_paraphrase(i=i, paraphrased_text=paraphrased_text)
    save_extracted_info(i=i, extracted_info=extracted_info)

    assert type(paraphrased_text) == str, f"paraphrased_text must be a str, but is of type {type(paraphrased_text)}"
    assert len(paraphrased_text.split()) > 0, "paraphrased_text is empty"

    # Build the new document
    paraphrase_doc = {
        "text_id": text_id,
        "length_original_text": len(original_text.split()),
        "length_paraphrased_text": len(paraphrased_text.split()),
        "intermediate_prompt": "bulletpoints dspy",
        "prompt": "bulletpoints dspy",#CONFIG.PROMPT,
        "llm": CONFIG.OPENAI_MODEL,
        "temperature": 1.0, # requirements for reasoning models like gpt-5-nano
        "paraphrase": paraphrased_text,
        "extracted_info": extracted_info,
        "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "openai_costs": total_costs,
    }

    # Insert into document into paraphrase collection
    paraphrase_collection.insert_one(paraphrase_doc)
    print(f"Inserted paraphrase for document ID: {text_id}")

print(f"Paraphrasing complete. Inserted/Updated {max_docs} documents.")
