from pymongo import MongoClient
from genai_detection.config import CONFIG
from genai_detection.paraphrasing import paraphraser


def paraphrase(text, llm: str, temperature: float, prompt: str) -> str:
    paraphraser_llm = paraphraser.SAIAParaphraser(model_id=llm)
    return paraphraser_llm.paraphrase(text, temperature=temperature, prompt=prompt)[0]


# MongoDB setup
uri = f"mongodb://{CONFIG.MONGO_USER}:{CONFIG.MONGO_PASSWORD}@{CONFIG.MONGO_HOST}/"
client = MongoClient(uri)
db = client[CONFIG.MONGO_DATABASE or "impostors"]
original_collection = db[CONFIG.MONGO_ORIGINAL_TEXT_COLLECTION or "original_text"]
paraphrase_collection_name = CONFIG.MONGO_PARAPHRASE_COLLECTION or "paraphrase"

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
LLM = CONFIG.SAIA_MODEL
TEMPERATURE = 0.7
# TODO: delete following line
max_docs = 2  # for testing, limit to first 5 documents
for doc in original_collection.find(limit=max_docs):
    text_id = doc["_id"]
    original_text = doc["text"]

    # Call the paraphrase function
    paraphrased_text = paraphrase(
        original_text, llm=LLM, temperature=TEMPERATURE, prompt=CONFIG.PROMPT
    )

    # Build the new document
    paraphrase_doc = {
        "text_id": text_id,
        # "intermediate_prompt": "",
        "prompt": CONFIG.PROMPT,
        "llm": LLM,
        "temperature": TEMPERATURE,
        "paraphrase": paraphrased_text,
    }

    # Insert into paraphrase collection
    paraphrase_collection.insert_one(paraphrase_doc)
    print(f"Inserted paraphrase for document ID: {text_id}")

print("Paraphrasing complete.")
