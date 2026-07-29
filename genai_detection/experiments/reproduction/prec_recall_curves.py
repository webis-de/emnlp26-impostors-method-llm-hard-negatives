# Copyright 2026 Klara M. Gutekunst, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from bson import ObjectId
from matplotlib import pyplot as plt
from sklearn.metrics import auc

from genai_detection.config import CONFIG
from genai_detection.detectors.components.impostor_factory import IMPOSTOR_GENERATORS
from genai_detection.detectors.impostor import ImpostorDetector
from genai_detection.detectors.impostor_supervised_baseline import SupervisedImpostorBaseline
from genai_detection.detectors.impostor_unsupervised_baseline import UnSupervisedImpostorBaseline
from genai_detection.detectors.ppmd import PPMdDetector
from genai_detection.detectors.unmasking import UnmaskingDetector
from genai_detection.experiments.reproduction.impostor_metrics import (
    compute_metrics_for_thresholds,
    compute_metrics_parallel,
    load_all_pairs,
)
from genai_detection.experiments.reproduction.pan_metrics.pan_data_loader import (
    PANDataLoader,
)
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger(__name__)

ONE_STEP_LLM_KEY_PREFIX = "one_step_llm:"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[3]
    / CONFIG.SAVE_PATH
    / "reproduction"
)
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def _build_baselines(dataset_name: str):
    return {
        "unsupervised_baseline_min-max": UnSupervisedImpostorBaseline(
            use_cosine_simiarity=False,
            dataset_name=dataset_name,
        ),
        "unsupervised_baseline_cosine": UnSupervisedImpostorBaseline(
            use_cosine_simiarity=True,
            dataset_name=dataset_name,
        ),
        # TODO: uncomment
        # "supervised_baseline": SupervisedImpostorBaseline(
        #     dataset_name=dataset_name
        # ),
        "unmasking": UnmaskingDetector(),
        "ppmd": PPMdDetector(),
    }

def _iter_batches(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def _baseline_config_for(
    baseline_name: str,
    baseline: object,
) -> Dict[str, object]:
    if isinstance(baseline, UnSupervisedImpostorBaseline):
        return {"use_cosine_simiarity": "cosine" in baseline_name}
    if isinstance(baseline, SupervisedImpostorBaseline):
        return {"model": "LinearSVC"}
    if isinstance(baseline, UnmaskingDetector):
        return {
            "rounds": baseline.rounds,
            "top_n": baseline.top_n,
            "cv_folds": baseline.cv_folds,
            "n_delete": baseline.n_delete,
            "shared_vocab_only": baseline.shared_vocab_only,
            "chunk_size": baseline.chunk_size,
            "relative_freqs": baseline.relative_freqs,
            "bootstrap": baseline.bootstrap,
            "n_chunks": baseline.n_chunks,
            "smoothing_kernel_size": baseline.smoothing_kernel_size,
            "strict": baseline.strict,
        }
    if isinstance(baseline, PPMdDetector):
        return {"strict": baseline.strict}
    return {}

def _score_baselines_for_pair_batches(
    baselines: Dict[str, object],
    pair_batches: Iterable[List[Tuple[str, str]]],
    *,
    pair_id_batches: Optional[Iterable[List[Tuple[ObjectId, ObjectId]]]] = None,
    dataset_name: Optional[str] = None,
    mongoDB: Optional[ParaphraseMongoDB] = None,
) -> Dict[str, List[float]]:
    predictions: Dict[str, List[float]] = {name: [] for name in baselines.keys()}
    if pair_id_batches is None or dataset_name is None or mongoDB is None:
        for batch in pair_batches:
            if not batch:
                continue
            flat_texts = [text for pair in batch for text in pair]
            for name, baseline in baselines.items():
                preds = baseline.get_score(flat_texts)
                preds = preds.tolist() if hasattr(preds, "tolist") else preds
                predictions[name].extend(np.asarray(preds).ravel().tolist())
        return predictions

    baseline_configs = {
        name: _baseline_config_for(name, baseline)
        for name, baseline in baselines.items()
    }
    for batch, id_batch in zip(pair_batches, pair_id_batches):
        if not batch:
            continue
        if len(batch) != len(id_batch):
            raise ValueError(
                f"Text and ID batches must align, got {len(batch)} texts and {len(id_batch)} ids."
            )
        id_pairs = [(ObjectId(l), ObjectId(r)) for l, r in id_batch]
        for name, baseline in baselines.items():
            config = baseline_configs[name]
            existing_scores: Dict[Tuple[ObjectId, ObjectId], float] = {}
            for id_sub_batch in _iter_batches(id_pairs, 500):
                if not id_sub_batch:
                    continue
                or_conditions = [
                    {"left_id": left_id, "right_id": right_id}
                    for left_id, right_id in id_sub_batch
                ]
                cursor = mongoDB.baseline_output_collection.find(
                    {
                        "baseline": name,
                        "dataset_name": dataset_name,
                        "config": config,
                        "$or": or_conditions,
                    },
                    {"left_id": 1, "right_id": 1, "score": 1},
                ).sort("_id", 1)
                for doc in cursor:
                    existing_scores[(doc["left_id"], doc["right_id"])] = doc["score"]

            missing_pairs: List[Tuple[str, str]] = []
            missing_ids: List[Tuple[ObjectId, ObjectId]] = []
            for pair_texts, pair_ids in zip(batch, id_pairs):
                if pair_ids not in existing_scores:
                    missing_pairs.append(pair_texts)
                    missing_ids.append(pair_ids)

            computed_scores: List[float] = []
            if missing_pairs:
                missing_flat_texts = [
                    text for pair in missing_pairs for text in pair
                ]
                preds = baseline.get_score(missing_flat_texts)
                preds = preds.tolist() if hasattr(preds, "tolist") else preds
                computed_scores = np.asarray(preds).ravel().tolist()
                if len(computed_scores) != len(missing_pairs):
                    raise ValueError(
                        f"Baseline {name}: expected {len(missing_pairs)} scores, got {len(computed_scores)}"
                    )
                created_at = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                docs = [
                    {
                        "baseline": name,
                        "created_at": created_at,
                        "dataset_name": dataset_name,
                        "left_id": left_id,
                        "right_id": right_id,
                        "score": float(score),
                        "config": config,
                    }
                    for (left_id, right_id), score in zip(missing_ids, computed_scores)
                ]
                if docs:
                    try:
                        mongoDB.insert_documents(
                            collection=mongoDB.baseline_output_collection,
                            insert_data=docs,
                        )
                        logger.info(
                            "Inserted %d baseline scores into MongoDB collection %s.",
                            len(docs),
                            mongoDB.baseline_output_collection.name,
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to insert baseline scores into MongoDB: %s",
                            e,
                        )

            computed_iter = iter(computed_scores)
            ordered_scores: List[float] = []
            for pair_ids in id_pairs:
                score = existing_scores.get(pair_ids)
                if score is None:
                    score = next(computed_iter)
                ordered_scores.append(score)

            predictions[name].extend(ordered_scores)

    return predictions

def _compute_metrics_for_predictions(
    ground_truth: List[int],
    predictions: Dict[str, List[float]],
) -> Dict[str, pd.DataFrame]:
    logger.info(
        "Computing metrics for %d approaches",
        len(predictions),
    )
    return compute_metrics_parallel(
        ground_truth,
        predictions,
    )

def compute_prec_recall_f1_acc_dict(
    dataset_name: str,
    imp_gen_techniques: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Reproduction of Figures 4a and 4b from Koppel et al. (2014).
    """
    PAIR_BATCH_SIZE = 256  # number of PAIRS per batch (must be int)
    TEXT_BATCH_SIZE = PAIR_BATCH_SIZE * 2

    logger.info(
        "Reproducing Figure 4 with techniques: %s",
        imp_gen_techniques,
    )

    assert all(
        t in IMPOSTOR_GENERATORS for t in imp_gen_techniques
    ), f"Unsupported impostor generator in {imp_gen_techniques}"

    # -----------------------------------------------------------------
    # Load test data
    # -----------------------------------------------------------------

    text_id_pairs, ground_truth = load_all_pairs(dataset_name)
    text_id_pairs_len = len(text_id_pairs)
    assert text_id_pairs_len % 2 == 0, "Flattened text list must contain even number of elements but length is {}".format(text_id_pairs_len)
    logger.info("Number of texts used %d, number of pairs %d",text_id_pairs_len, text_id_pairs_len//2)

    # -----------------------------------------------------------------
    # Collect predictions (batched over texts, but compute all techniques per batch)
    # -----------------------------------------------------------------

    predictions: Dict[str, List[float]] = {tech: [] for tech in imp_gen_techniques}
    # Build detectors once (avoid re-instantiating each batch)
    detectors = {
        technique: ImpostorDetector(
            impostor_technique=technique,
            n_impostors=50,
            dataset_name=dataset_name,
            llm=CONFIG.HUGGINGFACE_FINETUNED_MODEL if technique == "one_step_llm" else None,
            # llm=CONFIG.BATCH_OPENAI_MODEL if technique == "one_step_llm" else None,     
        )
        for technique in imp_gen_techniques
    }

    for start in range(0, text_id_pairs_len, TEXT_BATCH_SIZE):
        batch_texts = text_id_pairs[start: start + TEXT_BATCH_SIZE]

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
            batch_scores = detector.get_score(text=batch_texts)
            assert batch_scores is not None
            if len(batch_scores) != batch_pair_count:
                raise ValueError(
                    f"Technique {technique}: expected {batch_pair_count} scores, got {len(batch_scores)}"
                )

            predictions[technique].extend(batch_scores)

    # Sanity check: all techniques produced full-length results
    total_pairs = text_id_pairs_len // 2
    for technique in imp_gen_techniques:
        if len(predictions[technique]) != total_pairs:
            raise ValueError(
                f"Technique {technique}: expected {total_pairs} total scores, got {len(predictions[technique])}"
            )

    baselines = _build_baselines(dataset_name=dataset_name)

    mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    text_test_pairs = mongoDB.get_texts_for_ids(text_ids=text_id_pairs)

    text_pairs = list(zip(text_test_pairs[0::2], text_test_pairs[1::2]))
    id_pairs = list(zip(text_id_pairs[0::2], text_id_pairs[1::2]))
    baseline_predictions = _score_baselines_for_pair_batches(
        baselines=baselines,
        pair_batches=[text_pairs],
        pair_id_batches=[id_pairs],
        dataset_name=dataset_name,
        mongoDB=mongoDB,
    )
    predictions.update(baseline_predictions)
    # -----------------------------------------------------------------
    # Metric computation (shared implementation)
    # -----------------------------------------------------------------

    results = _compute_metrics_for_predictions(
        ground_truth=ground_truth,
        predictions=predictions,
    )

    return results

def compute_prec_recall_f1_acc_dict_on_existing_impostor_scores(
    dataset_name: str,
    imp_gen_techniques: List[str],
    *,
    n_impostors: int = 50,
    n_potential_impostors: Optional[int] = None,
    rounds: int = 100,
    batch_size: int = 250,
    include_baselines: bool = True,
    save_artifacts: bool = True,
    one_step_llm_label_translations: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Same as compute_prec_recall_f1_acc_dict, but loads precomputed
    impostor scores from the MongoDB impostor_outputs collection instead of active computation.
    Baselines are computed on the fly unless include_baselines=False.
    Metrics are computed per method on its available pairs (no cross-method alignment).
    """
    logger.info(
        "Reproducing Figure 4 from stored impostor outputs for %s",
        imp_gen_techniques,
    )

    assert all(
        t in IMPOSTOR_GENERATORS for t in imp_gen_techniques
    ), f"Unsupported impostor generator in {imp_gen_techniques}"
    if not imp_gen_techniques:
        raise ValueError("imp_gen_techniques must not be empty.")

    mongoDB = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    loader = PANDataLoader(mongo=mongoDB, batch_size=batch_size)

    results: Dict[str, pd.DataFrame] = {}
    n_pairs_per_technique: dict[str, int] = defaultdict(int)

    for technique in imp_gen_techniques:
        if technique == "one_step_llm" and one_step_llm_label_translations is not None:
            scores_by_llm = loader.load_one_step_llm_scores_by_llm(
                dataset_name=dataset_name,
                n_impostors=10,
                n_potential_impostors=n_potential_impostors,
                rounds=rounds,
            )
            for llm, scores_by_pair in scores_by_llm.items():
                result_key = f"{ONE_STEP_LLM_KEY_PREFIX}{llm}" if llm else technique
                n_pairs_per_technique[result_key] = len(scores_by_pair)
                if not scores_by_pair:
                    logger.warning(
                        "No scores for one_step_llm llm=%s found in MongoDB.",
                        llm,
                    )
                    continue

                ordered_pairs = list(scores_by_pair.keys())
                gt_by_pair = loader.load_ground_truth(ordered_pairs, dataset_name)
                ordered_pairs = [pair for pair in ordered_pairs if pair in gt_by_pair]
                if not ordered_pairs:
                    logger.warning(
                        "No ground-truth found for one_step_llm llm=%s (dataset=%s).",
                        llm,
                        dataset_name,
                    )
                    continue

                ground_truth = [gt_by_pair[pair] for pair in ordered_pairs]
                scores = [scores_by_pair[pair] for pair in ordered_pairs]
                results[result_key] = compute_metrics_for_thresholds(
                    ground_truth=ground_truth,
                    scores=scores,
                    thresholds=CONFIG.THRESHOLDS,
                )
                logger.info(
                    "Results for %s (pairs=%d): %s",
                    result_key,
                    len(scores),
                    results[result_key],
                )
            continue

        scores_by_pair = loader.load_scores(
            method_name=technique,
            dataset_name=dataset_name,
            n_impostors=10 if technique == "one_step_llm" else n_impostors,
            n_potential_impostors=n_potential_impostors,
            rounds=rounds,
        )
        n_pairs_per_technique[technique] = len(scores_by_pair)
        if not scores_by_pair:
            logger.warning("No scores for technique %s found in MongoDB.", technique)
            continue

        ordered_pairs = list(scores_by_pair.keys())
        gt_by_pair = loader.load_ground_truth(ordered_pairs, dataset_name)
        ordered_pairs = [pair for pair in ordered_pairs if pair in gt_by_pair]
        if not ordered_pairs:
            logger.warning(
                "No ground-truth found for technique %s (dataset=%s).",
                technique,
                dataset_name,
            )
            continue
        if len(ordered_pairs) != len(scores_by_pair):
            logger.warning(
                "Missing ground-truth for %d/%d pairs for technique %s (dataset=%s).",
                len(scores_by_pair) - len(ordered_pairs),
                len(scores_by_pair),
                technique,
                dataset_name,
            )

        ground_truth = [gt_by_pair[pair] for pair in ordered_pairs]
        scores = [scores_by_pair[pair] for pair in ordered_pairs]
        results[technique] = compute_metrics_for_thresholds(
            ground_truth=ground_truth,
            scores=scores,
            thresholds=CONFIG.THRESHOLDS,
        )
        logger.info(
            "Results for %s (pairs=%d): %s",
            technique,
            len(scores),
            results[technique],
        )

    if include_baselines:
        baselines = _build_baselines(dataset_name=dataset_name)
        # load all pairs per dataset
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

        text_test_pairs = mongoDB.get_texts_for_ids(text_ids=text_id_pairs)

        text_pairs = list(zip(text_test_pairs[0::2], text_test_pairs[1::2]))
        id_pairs = list(zip(text_id_pairs[0::2], text_id_pairs[1::2]))
        baseline_predictions = _score_baselines_for_pair_batches(
            baselines=baselines,
            pair_batches=[text_pairs],
            pair_id_batches=[id_pairs],
            dataset_name=dataset_name,
            mongoDB=mongoDB,
        )

        for baseline, pred in baseline_predictions.items():
            if len(pred) != len(ground_truth):
                raise ValueError(
                    f"Baseline {baseline}: expected {len(ground_truth)} scores, got {len(pred)}"
                )
            n_pairs_per_technique[baseline] = len(pred)
            results[baseline] = compute_metrics_for_thresholds(
                ground_truth=ground_truth,
                scores=pred,
                thresholds=CONFIG.THRESHOLDS,
            )
            logger.info(
                "Results for %s (pairs=%d): %s",
                baseline,
                len(pred),
                results[baseline],
            )

    if save_artifacts:
        with open(LOCAL_SAVE_PATH / f"prec_rec_{dataset_name}_n_pairs_per_technique.json", "w") as f:
            json.dump(dict(n_pairs_per_technique), f, indent=2)

    return results

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def _one_step_llm_from_result_key(key: str) -> Optional[str]:
    if key.startswith(ONE_STEP_LLM_KEY_PREFIX):
        return key[len(ONE_STEP_LLM_KEY_PREFIX):]
    return None


def _label_for_result_key(
    key: str,
    label_translations: Optional[Dict[str, str]],
    one_step_llm_label_translations: Optional[Dict[str, str]],
) -> str:
    llm = _one_step_llm_from_result_key(key)
    if llm is not None:
        if one_step_llm_label_translations and llm in one_step_llm_label_translations:
            return one_step_llm_label_translations[llm]
        return CONFIG.LABEL_TRANSLATIONS["one_step_llm"]
    if label_translations is None:
        return key
    return label_translations.get(key, key)


def _color_for_result_key(key: str) -> str:
    llm = _one_step_llm_from_result_key(key)
    if llm is None:
        return CONFIG.LABEL_COLORS.get(key, "black")
    configured_color = CONFIG.ONE_STEP_LLM_LABEL_COLORS.get(llm)
    if configured_color:
        return configured_color
    palette = plt.get_cmap("tab20").colors
    color_index = int(hashlib.md5(llm.encode("utf-8")).hexdigest(), 16) % len(palette)
    return palette[color_index]

def plot_precision_recall_curve(
    results: Dict[str, pd.DataFrame],
    dataset_name: str,
    save_path: Path = None,
    title: str = None,
    filename_extra: str= None,
    label_translations: Optional[Dict[str, str]] = CONFIG.LABEL_TRANSLATIONS,
    one_step_llm_label_translations: Optional[Dict[str, str]] = None,
):
    """
    Precision–Recall curves (Figures 4a, 4b in Koppel et al., 2014).
    """
    for positive_class_id in [0, 1]:
        fig = plt.figure(figsize=(10, 7))
        positive_class = (
            "Same Author" if positive_class_id == 1 else "Different Author"
        )

        for key, df in results.items():
            label = _label_for_result_key(
                key=key,
                label_translations=label_translations,
                one_step_llm_label_translations=one_step_llm_label_translations,
            )
            color = _color_for_result_key(key)
            if "precision" in df.columns and "recall" in df.columns:
                precisions = df["precision"].apply(
                    lambda x: x[positive_class_id]
                )
                recalls = df["recall"].apply(
                    lambda x: x[positive_class_id]
                )

                # filter out points where both precision and recall are zero
                mask = ~(
                    ((precisions == 0) & (recalls == 0))
                    | ((precisions == 1) & (recalls == 0))
                    | ((precisions == 0) & (recalls == 1))
                )

                precisions = precisions[mask]
                recalls = recalls[mask]
                unique_pairs = set(zip(recalls, precisions))

                logger.info("%s: class %d", key, positive_class_id)

                plt.plot(
                    recalls,
                    precisions,
                    # marker="o",
                    # markersize=4,
                    color=color,
                    label=label,
                )
                logger.info("Plotted %d unique (x, y) pairs for %s", len(unique_pairs), label)

        FONTSIZE_AXIS = 16
        FONTSIZE_TICKS = 14

        plt.grid(True, linestyle="--", alpha=0.6)
        plt.ylim(0, 1)
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
        plt.xlabel("Recall", fontsize=FONTSIZE_AXIS) # $\\frac{TP}{TP + FN}$
        plt.ylabel("Precision", fontsize=FONTSIZE_AXIS) # $\\frac{TP}{TP + FP}$
        if title is None:
            title = "Precision–Recall Curve Across Impostor Generation Techniques"
        complete_title = title + f"\nDataset: {CONFIG.DATASET_TRANSLATIONS[dataset_name]} ({positive_class})"
        # plt.title(complete_title)
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.legend(fontsize=FONTSIZE_TICKS)

        for format in ["pdf", "svg"]:
            fname = (
                f"roc_prec_recall_curve_{dataset_name.replace(' ', '_')}_"
                f"{positive_class.lower().replace(' ', '_')}"
            )
            if filename_extra:
                fname = fname + "_" + filename_extra
            fname = fname + f".{format}"
            if not save_path:
                logger.warning(f"No save path for {fname}, using default save path: {LOCAL_SAVE_PATH}")
                save_path = LOCAL_SAVE_PATH
            plt.savefig(
                save_path / fname,
                bbox_inches="tight",
            )
            logger.info("Saved plot to %s", save_path/fname)

        fig.clear()


def _extract_best_pr_points_per_impostor(
    results: Dict[str, pd.DataFrame],
    dataset_name: str,
) -> pd.DataFrame:
    rows = []

    for impostor_method, df in results.items():

        for class_id, class_name in [(0, "Different Author"), (1, "Same Author")]:
            if "precision" in df.columns and "recall" in df.columns:
                precisions = df["precision"].apply(lambda x: x[class_id]).to_numpy()
                recalls = df["recall"].apply(lambda x: x[class_id]).to_numpy()
                thresholds = df["threshold"].to_numpy()

                # Sort by recall for valid PR AUC
                order = np.argsort(recalls)
                recalls_sorted = recalls[order]
                precisions_sorted = precisions[order]

                pr_auc = auc(recalls_sorted, precisions_sorted)

                # ---- Best precision (tie → recall) ----
                best_p_idx = np.lexsort((-recalls, -precisions))[0]

                # ---- Best recall (tie → precision) ----
                best_r_idx = np.lexsort((-precisions, -recalls))[0]

                # ---- Best PR operating point (proxy for PR-AUC) ----
                pr_product = precisions * recalls
                best_auc_idx = pr_product.argmax()

                rows.append({
                    "dataset_name": dataset_name.replace("_", " ").capitalize(),
                    "impostor_generation": _label_for_result_key(
                        key=impostor_method,
                        label_translations=CONFIG.LABEL_TRANSLATIONS,
                        one_step_llm_label_translations=CONFIG.ONE_STEP_LLM_LABEL_TRANSLATIONS,
                    ),
                    "class": class_name,

                    "n_test_samples": 50,   # TODO: adjust

                    "best_precision": precisions[best_p_idx],
                    "best_precision_recall": recalls[best_p_idx],
                    "best_precision_threshold": thresholds[best_p_idx],

                    "best_recall": recalls[best_r_idx],
                    "best_recall_precision": precisions[best_r_idx],
                    "best_recall_threshold": thresholds[best_r_idx],

                    "best_auc_precision": precisions[best_auc_idx],
                    "best_auc_recall": recalls[best_auc_idx],
                    "best_auc_threshold": thresholds[best_auc_idx],

                    "pr_auc": pr_auc,
                })

    return pd.DataFrame(rows)

def run_prec_recall_curves(
    dataset_name: str,
    imp_gen_techniques: List[str],
    compute_score_fn=compute_prec_recall_f1_acc_dict,
    one_step_llm_label_translations: Optional[Dict[str, str]] = None,
):
    compute_kwargs = {}
    if (
        compute_score_fn is compute_prec_recall_f1_acc_dict_on_existing_impostor_scores
        and one_step_llm_label_translations is not None
    ):
        compute_kwargs["one_step_llm_label_translations"] = one_step_llm_label_translations
    results_dict = compute_score_fn(
        dataset_name=dataset_name,
        imp_gen_techniques=imp_gen_techniques,
        **compute_kwargs,
    )
    logger.info("Obtained scores for approaches %s", results_dict.keys())

    # results_dict: {approach_name: DataFrame}
    dfs = []
    pr_aucs = {}
    for approach, df in results_dict.items():
        temp_df = df.copy()
        temp_df["approach"] = approach
        dfs.append(temp_df)
        # Precision/recall are per-class arrays; compute PR-AUC per class.
        if "precision" in df.columns and "recall" in df.columns:
            class_aucs = {}
            for class_id, class_name in [(0, "Different Author"), (1, "Same Author")]:
                precisions = df["precision"].apply(lambda x: x[class_id]).to_numpy()
                recalls = df["recall"].apply(lambda x: x[class_id]).to_numpy()
                # Sort by recall for a valid PR-AUC.
                order = np.argsort(recalls)
                class_aucs[class_name] = auc(recalls[order], precisions[order])
            pr_aucs[approach] = class_aucs

    # Combine all approaches
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save to CSV
    combined_df.to_csv(LOCAL_SAVE_PATH / f"effectiveness_scores_{dataset_name}.csv", index=False)
    logger.info(f"Saved effectiveness scores as csv to {LOCAL_SAVE_PATH}/effectiveness_scores_{dataset_name}.csv.")

    # save precision-recall AUC values
    with open(LOCAL_SAVE_PATH / f"prec_rec_auc_{dataset_name}.json", "w") as f:
        json.dump(pr_aucs, f, indent=2)

    # best PR operating points
    best_pr_df = _extract_best_pr_points_per_impostor(
        results=results_dict,
        dataset_name=dataset_name,
    )

    best_pr_df.to_csv(
        LOCAL_SAVE_PATH / f"best_precision_recall_points_{dataset_name}.csv",
        index=False,
        float_format="%.2f"
    )
    logger.info(
        "Saved best precision–recall operating points to %s",
        LOCAL_SAVE_PATH / "best_precision_recall_points.csv",
    )

    plot_precision_recall_curve(
        results=results_dict,
        dataset_name=dataset_name,
        one_step_llm_label_translations=one_step_llm_label_translations,
    )
