import argparse
import logging
import math
import os
import random
from typing import Dict, List

from bson import ObjectId

from genai_detection.config import CONFIG
from genai_detection.detectors.impostor import ImpostorDetector
from genai_detection.experiments.reproduction.impostor_metrics import load_all_pairs
from genai_detection.impostor_generators.chatnoir_impostor_generator import (
    ChatNoirSearchImpostorGenerator,
)
from genai_detection.impostor_generators.google_search_impostor_generator import (
    GoogleSearchImpostorGenerator,
)
from genai_detection.impostor_generators.startpage_impostor_generator import (
    StartPageSearchImpostorGenerator,
)
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

def main(imp_gen_techniques, dataset_name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info(f"Dataset {dataset_name} with impostor generation techniques {imp_gen_techniques}.")

    text_id_pairs, ground_truth = load_all_pairs(dataset_name)
    text_id_pairs_len = len(text_id_pairs)
    assert (
        text_id_pairs_len % 2 == 0
    ), "Flattened text list must contain even number of elements but length is {}".format(
        text_id_pairs_len
    )
    logger.info(
        "Number of texts used %d, number of pairs %d",
        text_id_pairs_len,
        text_id_pairs_len // 2,
    )
    TEXT_BATCH_SIZE = 50

    # Build detectors once (avoid re-instantiating each batch)
    detectors = {
        technique: ImpostorDetector(
            impostor_technique=technique,
            n_impostors=50,
            dataset_name=dataset_name,
        )
        for technique in imp_gen_techniques
    }

    for start in range(0, text_id_pairs_len, TEXT_BATCH_SIZE):
        batch_texts = text_id_pairs[start : start + TEXT_BATCH_SIZE]

        # Ensure we never split a pair (should be guaranteed by TEXT_BATCH_SIZE, but keep safe)
        if len(batch_texts) % 2 != 0:
            raise ValueError(
                f"Batch starting at {start} has odd number of texts ({len(batch_texts)}), would break pairing."
            )

        batch_pair_count = len(batch_texts) // 2
        logger.info(
            "Scoring batch: texts [%d:%d] -> %d pairs",
            start,
            min(start + TEXT_BATCH_SIZE, text_id_pairs_len),
            batch_pair_count,
        )

        for technique in imp_gen_techniques:
            detector = detectors[technique]
            _ = detector.get_score(text=batch_texts)


if __name__ == "__main__":
    imp_gen_techniques = [
        "on_the_fly_startpage",
        "on_the_fly_chatnoir",
        "on_the_fly_serpapi",
    ]
    imp_per_appr = {CONFIG.BLOG:[imp_gen_techniques[0]], CONFIG.STUDENT_ESSAYS:[imp_gen_techniques[0], imp_gen_techniques[
        1]]}
    for dataset_name in [CONFIG.BLOG, CONFIG.STUDENT_ESSAYS]:
        main(imp_per_appr[dataset_name], dataset_name)
