import json
from pathlib import Path
from typing import List

from genai_detection.config import CONFIG
from genai_detection.impostor_generators.ImpostorGenerator import LLMImpostorGenerator
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

def save_overview_file(original_text:str, paraphrases:List[str], extracted_info_dict:dict, doc_id: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    paraphrase_filename = (
        f"{doc_id}_overview_{CONFIG.OPENAI_MODEL.replace('/', '_')}.txt"
    )
    file_path = save_dir / paraphrase_filename

    with open(file_path, "w", encoding="utf-8") as f:
        # Original text
        f.write(f"{'-'*100}\nORIGINAL TEXT\n{'-'*100}\n")
        f.write(original_text.strip() + "\n\n")

        # Extracted info
        f.write(f"{'-'*100}\nEXTRACTED INFO\n{'-'*100}\n")
        for key, value in extracted_info_dict.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # Paraphrases
        for i, paraphrase in enumerate(paraphrases, start=1):
            f.write(f"{'-'*100}\nPARAPHRASE {i}\n{'-'*100}\n")
            f.write(paraphrase.strip() + "\n\n")

    print(f"Overview file saved at: {file_path}")


for i, doc in enumerate(original_collection.find(limit=max_docs)):
    original_text_id = doc["_id"]
    original_text = doc["text"]

    llm_paraphraser = LLMImpostorGenerator(n_impostors=2)
    paraphrased_texts = llm_paraphraser.generate_impostors(
        text_id=original_text_id, text=None
    )

    # save results to files for manual inspection
    cursor = mongoDB.find_paraphrases(document_id=original_text_id)
    docs = list(cursor)  # materialize once, safe if the number is small
    extracted_info = next(
        (doc["extracted_info"] for doc in docs if "extracted_info" in doc), {}
    )
    save_overview_file(original_text=original_text, paraphrases=paraphrased_texts,
                       extracted_info_dict=extracted_info, doc_id=f"{original_text_id}_{i}")

print(f"Paraphrasing complete. Inserted/Updated {max_docs} documents.")
