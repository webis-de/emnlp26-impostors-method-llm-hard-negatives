from pymongo import MongoClient
from genai_detection.config import CONFIG
from genai_detection.paraphrasing import paraphraser_evaluation

# MongoDB setup
uri = f"mongodb://{CONFIG.MONGO_USER}:{CONFIG.MONGO_PASSWORD}@{CONFIG.MONGO_HOST}/"
client = MongoClient(uri)
db = client[CONFIG.MONGO_DATABASE or "impostors"]
original_collection = db[CONFIG.MONGO_ORIGINAL_TEXT_COLLECTION or "original_text"]
paraphrase_collection = db[CONFIG.MONGO_PARAPHRASE_COLLECTION or "paraphrase"]
score_collection_name = CONFIG.MONGO_SCORE_COLLECTION or "score"

# Create score collection if it doesn't exist
if score_collection_name not in db.list_collection_names():
    db.create_collection(score_collection_name)
    print(f"Created collection: {score_collection_name}")
else:
    print(f"Collection '{score_collection_name}' already exists. Skipping creation.")

score_collection = db[score_collection_name]

paraphrase_evaluator = paraphraser_evaluation.ParaphrasingEvaluator(
    paraphrasers={}, prompts=[], original_text="dummy"
)

# Iterate over all paraphrases
for paraphrase_doc in paraphrase_collection.find():
    text_id = paraphrase_doc["text_id"]
    paraphrase_id = paraphrase_doc["_id"]
    paraphrased_text = paraphrase_doc["paraphrase"]  # returns a list of str
    # print(
    #     f"Scoring paraphrase ID: {paraphrase_id} for text ID: {text_id}\n{paraphrased_text}\nType {type(paraphrased_text)}"
    # )

    # Fetch corresponding original text
    original_doc = original_collection.find_one({"_id": text_id})
    if not original_doc:
        print(f"Original text with _id={text_id} not found. Skipping.")
        continue

    original_text = original_doc["text"]

    # Compute scores
    bert_scores = paraphrase_evaluator._safe_compute_bertscore(
        paraphrases=[paraphrased_text], references=[original_text]
    )
    rouge_scores = paraphrase_evaluator._safe_compute_rouge(
        paraphrases=[paraphrased_text], references=[original_text]
    )
    assert (
        type(paraphrased_text) is str
    ), f"paraphrased_text is not str: {type(paraphrased_text)}"
    scores = paraphrase_evaluator._build_result_row(
        paraphraser_name="dumy",
        prompt="prompt",
        paraphrase=paraphrased_text,
        original_split=original_text.split(),
        bert_scores=bert_scores,
        rouge_scores=rouge_scores[0],
        idx=0,
    )
    for key in ["model", "prompt", "parameters", "paraphrased_text", "original_text"]:
        if key in scores:
            scores.pop(key)

    # Insert into score collection
    score_doc = {"text_id": text_id, "paraphrase_id": paraphrase_id, **scores}

    score_collection.insert_one(score_doc)

print("Scoring complete.")
