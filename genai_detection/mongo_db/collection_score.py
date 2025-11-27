import logging

from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB
from genai_detection.paraphrasing import paraphraser_evaluation

logger = logging.getLogger(__name__)
logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )

# MongoDB setup
mongoDB = ParaphraseMongoDB()
original_collection = mongoDB.original_collection
paraphrase_collection = mongoDB.paraphrase_collection
paraphrase_score_collection = mongoDB.paraphrase_score_collection

paraphrase_evaluator = paraphraser_evaluation.ParaphrasingEvaluator(
    paraphrasers={}, prompts=[], original_text="dummy"
)

# Iterate over all paraphrases
for paraphrase_doc in paraphrase_collection.find():
    text_id = paraphrase_doc["text_id"]
    paraphrase_id = paraphrase_doc["_id"]
    paraphrased_text = paraphrase_doc["paraphrase"]  # returns a list of str
    # logging.info(
    #     f"Scoring paraphrase ID: {paraphrase_id} for text ID: {text_id}\n{paraphrased_text}\nType {type(paraphrased_text)}"
    # )

    # Fetch corresponding original text
    cursor = mongoDB.find_document_by_id(collection=original_collection, document_id=text_id)
    original_docs = list(cursor)
    if not original_docs:
        logging.info(f"Original text with _id={text_id} not found. Skipping.")
        continue

    original_text = original_docs[0]["text"]

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
        paraphraser_name="dummy",
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
    # Do not use text_id or paraphrase_id as _id since a text will be paraphrased multiple times with different settings
    score_doc = {"text_id": text_id, "paraphrase_id": paraphrase_id, **scores}

    paraphrase_score_collection.insert_one(score_doc)

logging.info("Scoring complete.")
