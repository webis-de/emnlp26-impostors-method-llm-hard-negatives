import json
from datetime import datetime

from pymongo import MongoClient
from genai_detection.config import CONFIG
from genai_detection.paraphrasing import paraphraser
from genai_detection.paraphrasing.openai_utils import OpenaiCostEstimator

# MongoDB setup
uri = f"mongodb://{CONFIG.MONGO_USER}:{CONFIG.MONGO_PASSWORD}@{CONFIG.MONGO_HOST}/"
client = MongoClient(uri)
db = client[CONFIG.MONGO_DATABASE or "impostors"]
original_collection = db[CONFIG.MONGO_ORIGINAL_TEXT_COLLECTION or "original_text"]
paraphrase_collection_name = CONFIG.MONGO_PARAPHRASE_COLLECTION or "paraphrase"
# 30.10.25, morning: $7,891.29 balance left on OpenAI account
# 30.10.25, evening: $7,891.28 balance left on OpenAI account
try:
    client.admin.command("ping")
    print("Successfully connected as MongoDB root user!")
except Exception as e:
    print("Connection failed:", e, uri)
    raise e

# Create paraphrase collection if it doesn't exist
if paraphrase_collection_name not in db.list_collection_names():
    db.create_collection(paraphrase_collection_name)
    print(f"Created collection: {paraphrase_collection_name}")
else:
    print(
        f"Collection '{paraphrase_collection_name}' already exists. Skipping creation."
    )

paraphrase_collection = db[paraphrase_collection_name]

# Iterate over all documents in original_text
paraphraser_llm = paraphraser.OpenAIParaphraser_dspy(model_id=CONFIG.OPENAI_MODEL)

# TODO: delete following line
max_docs = 2  # for testing, limit to first 5 documents
openai_cost_est = OpenaiCostEstimator()
i = 0
# FIXME: based on extracted info, create paraphrase prompt (has not work yet) works!!! make clean and fit together again.
# original_collection.find(limit=max_docs)
for doc in paraphrase_collection.find({"llm":"openai/gpt-5-nano-2025-08-07"}, limit=max_docs):
    # if i == 0:
    #     # skip first document for testing
    #     i += 1
    #     continue
    i += 1
    text_id = doc["_id"]
    # original_text = doc["text"]
    extracted_info = doc["extracted_info"]
    # print(openai_cost_est.compute_cost(prompt=CONFIG.PROMPT, text=original_text))

    # Call the paraphrase function
    # extracted_info, paraphrased_text, total_costs = paraphraser_llm.paraphrase(text=original_text, prompt=CONFIG.PROMPT)
    # TODO: delete; try to debug paraphrase generation based on extracted info
    # paraphrased_text, total_costs = paraphraser_llm.generate_paraphrase(extracted_info=extracted_info)
    paraphrase_filename = f"paraphrase_{CONFIG.OPENAI_MODEL.replace('/', '_')}_{i}.txt"
    with open(paraphrase_filename, "r", encoding="utf-8") as f:
        paraphrased_text = f.read()
    print(paraphrased_text)

    # save dict as jsonl line
    # extr_info_filename = f"extacted_information_{CONFIG.OPENAI_MODEL.replace('/','_')}_{i}"
    # if type(paraphrased_text) == dict:
    #     extr_info_filename += ".jsonl"
    #     with open(extr_info_filename, "w", encoding="utf-8") as f:
    #         json.dump(paraphrased_text, f, ensure_ascii=False, indent=4)
    # paraphrase_filename = f"paraphrase_{CONFIG.OPENAI_MODEL.replace('/', '_')}_{i}"
    # if type(paraphrased_text) == str:
    #     paraphrase_filename += ".txt"
    #     with open(paraphrase_filename, "w", encoding="utf-8") as f:
    #         f.write(paraphrased_text)


    # Build the new document
    # paraphrase_doc = {
    #     "text_id": text_id,
    #     "intermediate_prompt": "bulletpoints dspy",
    #     "prompt": "bulletpoints dspy",#CONFIG.PROMPT,
    #     "llm": CONFIG.OPENAI_MODEL,
    #     "temperature": 1.0, # requirements for reasoning models like gpt-5-nano
    #     "paraphrase": paraphrased_text,
    #     "extracted_info": extracted_info,
    #     "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    #     "openai_costs": total_costs,
    # }

    # Insert into paraphrase collection
    # paraphrase_collection.insert_one(paraphrase_doc)
    print(f"Inserted paraphrase for document ID: {text_id}")

    update_fields = {
        "paraphrase": paraphrased_text,
        # "openai_costs": total_costs,
        "updated_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }

    # Update *the same document* in place
    result = paraphrase_collection.update_one(
        {"_id": text_id},
        {"$set": update_fields}
    )

    if result.modified_count > 0:
        print(f"✅ Updated document {text_id} with new paraphrase.")
    else:
        print(f"⚠️ No update performed for {text_id} (may already be up-to-date).")

print("Paraphrasing complete.")
