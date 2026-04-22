from __future__ import annotations

"""Visualization helpers for PAN metrics."""

import logging
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from genai_detection.config import CONFIG
from genai_detection.experiments.reproduction.pan_metrics.pan_metric_computation import PANMetricComputer
from genai_detection.experiments.reproduction.pan_metrics.pan_storage import PANMetricsStore

logger = logging.getLogger(__name__)

def _default_heatmap_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / CONFIG.SAVE_PATH / "reproduction" / "pan_metrics" / "heatmaps"


def _ordered_methods(methods: Sequence[str]) -> list[str]:
    preferred = list(CONFIG.LABEL_TRANSLATIONS.keys())
    preferred_pos = {m: i for i, m in enumerate(preferred)}
    return sorted(methods, key=lambda m: (preferred_pos.get(m, 10_000), m))


def _valid_metric_names(names: Sequence[str]) -> list[str]:
    valid = set(CONFIG.SCORE_TRANSLATIONS.keys())
    filtered = [name for name in names if name in valid]
    ordered = [m for m in list(CONFIG.SCORE_TRANSLATIONS.keys()) if m in filtered]
    ordered += sorted([m for m in filtered if m not in ordered])
    return ordered


def _split_labels(n_splits_total: int, split_config: dict | None) -> list[str]:
    if split_config and "n_splits" in split_config and "n_repeats" in split_config:
        try:
            n_splits = int(split_config["n_splits"])
            n_repeats = int(split_config["n_repeats"])
        except (TypeError, ValueError):
            n_splits = 0
            n_repeats = 0
        if n_splits > 0 and n_repeats > 0 and n_splits_total == (n_splits * n_repeats):
            labels: list[str] = []
            for idx in range(n_splits_total):
                repeat = idx // n_splits + 1
                fold = idx % n_splits + 1
                labels.append(f"r{repeat:02d}-f{fold:02d}")
            return labels
    return [f"split_{i + 1:03d}" for i in range(n_splits_total)]


def _load_pan_metrics_docs(store: PANMetricsStore, dataset_name: str) -> list[dict]:
    cursor = store.mongo.pan_metrics_collection.find(
        {"dataset_name": dataset_name},
        {"_id": 0},
    )
    docs = [store._normalize_doc(doc) for doc in cursor]
    return [doc for doc in docs if doc.get("dataset_name") and doc.get("method_name")]


def _load_per_split_by_method(
    *,
    store: PANMetricsStore,
    dataset_name: str,
    methods: Sequence[str],
    per_split_feature_name: str,
) -> tuple[dict[str, list[dict]], dict | None]:
    """
    Load per-split PAN metrics grouped by method.
    """
    cursor = store.mongo.pan_metrics_collection.find(
        {"dataset_name": dataset_name, "method_name": {"$in": list(methods)}},
        {"_id": 0},
    )
    docs = list(cursor)
    if docs:
        per_split_by_method: dict[str, list[dict]] = {m: [] for m in methods}
        for doc in docs:
            method_name = doc.get("method_name")
            if method_name not in per_split_by_method:
                continue
            payload = doc.get(per_split_feature_name)
            per_split_by_method[method_name].append(payload)

        def sort_key(d: dict) -> int:
            # [{"lower_threshold": 0.52, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("112"), "n_unanswered_c_at_1": new NumberInt("15"), "precision": 0.5913978494623656, "recall": 0.873015873015873, "f1": 0.7051282051282052, "accuracy": 0.6377952755905512, "c_at_1": 0.6426932853865708, "auroc": 0.6893601190476191, "auroc_c_at_1": 0.44304711972519184}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("102"), "n_unanswered_c_at_1": new NumberInt("25"), "precision": 0.6111111111111112, "recall": 0.873015873015873, "f1": 0.7189542483660131, "accuracy": 0.6614173228346457, "c_at_1": 0.6973773947547894, "auroc": 0.7240823412698413, "auroc_c_at_1": 0.5049586567427102}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("114"), "n_unanswered_c_at_1": new NumberInt("13"), "precision": 0.6395348837209303, "recall": 0.859375, "f1": 0.7333333333333333, "accuracy": 0.6850393700787402, "c_at_1": 0.7378014756029512, "auroc": 0.7442956349206349, "auroc_c_at_1": 0.5491424177292799}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("103"), "n_unanswered_c_at_1": new NumberInt("24"), "precision": 0.6265060240963856, "recall": 0.8125, "f1": 0.7074829931972789, "accuracy": 0.6614173228346457, "c_at_1": 0.6834273668547337, "auroc": 0.7038690476190477, "auroc_c_at_1": 0.4810433698248349}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("107"), "n_unanswered_c_at_1": new NumberInt("19"), "precision": 0.6, "recall": 0.8095238095238095, "f1": 0.6891891891891891, "accuracy": 0.6349206349206349, "c_at_1": 0.6758629377676997, "auroc": 0.6899722852103805, "auroc_c_at_1": 0.4663266956605809}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("102"), "n_unanswered_c_at_1": new NumberInt("24"), "precision": 0.5913978494623656, "recall": 0.873015873015873, "f1": 0.7051282051282052, "accuracy": 0.6349206349206349, "c_at_1": 0.6708238851095993, "auroc": 0.7035777273872512, "auroc_c_at_1": 0.47197674456249844}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("96"), "n_unanswered_c_at_1": new NumberInt("30"), "precision": 0.5578947368421052, "recall": 0.8412698412698413, "f1": 0.6708860759493671, "accuracy": 0.5873015873015873, "c_at_1": 0.5993953136810279, "auroc": 0.6315192743764172, "auroc_c_at_1": 0.37852969356046773}, {"lower_threshold": 0.52, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("108"), "n_unanswered_c_at_1": new NumberInt("18"), "precision": 0.6105263157894737, "recall": 0.9206349206349206, "f1": 0.7341772151898734, "accuracy": 0.6666666666666666, "c_at_1": 0.6984126984126984, "auroc": 0.7344419249181153, "auroc_c_at_1": 0.5129435666094774}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("104"), "n_unanswered_c_at_1": new NumberInt("22"), "precision": 0.5934065934065934, "recall": 0.8571428571428571, "f1": 0.7012987012987013, "accuracy": 0.6349206349206349, "c_at_1": 0.6991685563114135, "auroc": 0.7155454774502394, "auroc_c_at_1": 0.500286898444045}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.44, "n_answered_c_at_1": new NumberInt("105"), "n_unanswered_c_at_1": new NumberInt("21"), "precision": 0.6329113924050633, "recall": 0.7936507936507936, "f1": 0.704225352112676, "accuracy": 0.6666666666666666, "c_at_1": 0.6759259259259259, "auroc": 0.7074829931972789, "auroc_c_at_1": 0.4782060972537163}, {"lower_threshold": 0.52, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("109"), "n_unanswered_c_at_1": new NumberInt("18"), "precision": 0.5681818181818182, "recall": 0.7936507936507936, "f1": 0.6622516556291391, "accuracy": 0.5984251968503937, "c_at_1": 0.6293012586025172, "auroc": 0.6563740079365079, "auroc_c_at_1": 0.41305698930842305}, {"lower_threshold": 0.46, "upper_threshold": 0.62, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("104"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.5531914893617021, "recall": 0.8253968253968254, "f1": 0.6624203821656051, "accuracy": 0.5826771653543307, "c_at_1": 0.5673011346022693, "auroc": 0.5987103174603174, "auroc_c_at_1": 0.3396490423933229}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("104"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.6067415730337079, "recall": 0.84375, "f1": 0.7058823529411765, "accuracy": 0.6456692913385826, "c_at_1": 0.7254014508029016, "auroc": 0.7368551587301588, "auroc_c_at_1": 0.5345158011744596}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("109"), "n_unanswered_c_at_1": new NumberInt("18"), "precision": 0.6022727272727273, "recall": 0.828125, "f1": 0.6973684210526315, "accuracy": 0.6377952755905512, "c_at_1": 0.692231384462769, "auroc": 0.6980406746031746, "auroc_c_at_1": 0.4832056625918808}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("102"), "n_unanswered_c_at_1": new NumberInt("24"), "precision": 0.6236559139784946, "recall": 0.9206349206349206, "f1": 0.7435897435897436, "accuracy": 0.6825396825396826, "c_at_1": 0.7180650037792895, "auroc": 0.754472159234064, "auroc_c_at_1": 0.5417600538717768}, {"lower_threshold": 0.52, "upper_threshold": 0.7, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("109"), "n_unanswered_c_at_1": new NumberInt("17"), "precision": 0.6129032258064516, "recall": 0.9047619047619048, "f1": 0.7307692307692307, "accuracy": 0.6666666666666666, "c_at_1": 0.6845553036029226, "auroc": 0.7128999748047367, "auroc_c_at_1": 0.48801945869097235}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.42, "n_answered_c_at_1": new NumberInt("95"), "n_unanswered_c_at_1": new NumberInt("31"), "precision": 0.5813953488372093, "recall": 0.7936507936507936, "f1": 0.6711409395973155, "accuracy": 0.6111111111111112, "c_at_1": 0.6329050138573947, "auroc": 0.6603678508440414, "auroc_c_at_1": 0.417950123789426}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("104"), "n_unanswered_c_at_1": new NumberInt("22"), "precision": 0.625, "recall": 0.873015873015873, "f1": 0.7284768211920529, "accuracy": 0.6746031746031746, "c_at_1": 0.7178130511463844, "auroc": 0.7425044091710757, "auroc_c_at_1": 0.5329793554367334}, {"lower_threshold": 0.52, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("112"), "n_unanswered_c_at_1": new NumberInt("14"), "precision": 0.6404494382022472, "recall": 0.9047619047619048, "f1": 0.75, "accuracy": 0.6984126984126984, "c_at_1": 0.7142857142857142, "auroc": 0.7538422776518015, "auroc_c_at_1": 0.5384587697512867}, {"lower_threshold": 0.46, "upper_threshold": 0.62, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("110"), "n_unanswered_c_at_1": new NumberInt("16"), "precision": 0.625, "recall": 0.873015873015873, "f1": 0.7284768211920529, "accuracy": 0.6746031746031746, "c_at_1": 0.7066011589821113, "auroc": 0.7311665406903503, "auroc_c_at_1": 0.5166431250607425}, {"lower_threshold": 0.52, "upper_threshold": 0.7, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("112"), "n_unanswered_c_at_1": new NumberInt("15"), "precision": 0.6363636363636364, "recall": 0.8888888888888888, "f1": 0.7417218543046358, "accuracy": 0.6929133858267716, "c_at_1": 0.7043214086428172, "auroc": 0.7477678571428571, "auroc_c_at_1": 0.526668910480678}, {"lower_threshold": 0.46, "upper_threshold": 0.62, "f1_threshold": 0.44, "n_answered_c_at_1": new NumberInt("107"), "n_unanswered_c_at_1": new NumberInt("20"), "precision": 0.5875, "recall": 0.746031746031746, "f1": 0.6573426573426573, "accuracy": 0.6141732283464567, "c_at_1": 0.6288672577345155, "auroc": 0.6639384920634921, "auroc_c_at_1": 0.4175291788083576}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("102"), "n_unanswered_c_at_1": new NumberInt("25"), "precision": 0.6470588235294118, "recall": 0.859375, "f1": 0.738255033557047, "accuracy": 0.6929133858267716, "c_at_1": 0.716225432450865, "auroc": 0.7553323412698412, "auroc_c_at_1": 0.5409882327701163}, {"lower_threshold": 0.52, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("113"), "n_unanswered_c_at_1": new NumberInt("14"), "precision": 0.6179775280898876, "recall": 0.859375, "f1": 0.7189542483660131, "accuracy": 0.6614173228346457, "c_at_1": 0.6818773637547275, "auroc": 0.7140376984126984, "auroc_c_at_1": 0.48688614341514397}, {"lower_threshold": 0.54, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("117"), "n_unanswered_c_at_1": new NumberInt("9"), "precision": 0.6086956521739131, "recall": 0.8888888888888888, "f1": 0.7225806451612903, "accuracy": 0.6587301587301587, "c_at_1": 0.6547619047619048, "auroc": 0.6899722852103805, "auroc_c_at_1": 0.45176756769727294}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("99"), "n_unanswered_c_at_1": new NumberInt("27"), "precision": 0.59375, "recall": 0.9047619047619048, "f1": 0.7169811320754716, "accuracy": 0.6428571428571429, "c_at_1": 0.6746031746031745, "auroc": 0.7031997984378937, "auroc_c_at_1": 0.4743808164065155}, {"lower_threshold": 0.52, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("114"), "n_unanswered_c_at_1": new NumberInt("12"), "precision": 0.5670103092783505, "recall": 0.873015873015873, "f1": 0.6875, "accuracy": 0.6031746031746031, "c_at_1": 0.6432350718065004, "auroc": 0.6631393298059964, "auroc_c_at_1": 0.4265544744254747}, {"lower_threshold": 0.46, "upper_threshold": 0.62, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("98"), "n_unanswered_c_at_1": new NumberInt("28"), "precision": 0.5789473684210527, "recall": 0.873015873015873, "f1": 0.6962025316455697, "accuracy": 0.6190476190476191, "c_at_1": 0.6790123456790124, "auroc": 0.7102544721592341, "auroc_c_at_1": 0.4822715551698503}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("104"), "n_unanswered_c_at_1": new NumberInt("22"), "precision": 0.6309523809523809, "recall": 0.8412698412698413, "f1": 0.7210884353741497, "accuracy": 0.6746031746031746, "c_at_1": 0.745779793398841, "auroc": 0.7619047619047619, "auroc_c_at_1": 0.5682131759229265}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("108"), "n_unanswered_c_at_1": new NumberInt("18"), "precision": 0.5666666666666667, "recall": 0.8095238095238095, "f1": 0.6666666666666666, "accuracy": 0.5952380952380952, "c_at_1": 0.6349206349206349, "auroc": 0.6510456034265558, "auroc_c_at_1": 0.41336228788987667}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("104"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.5824175824175825, "recall": 0.8412698412698413, "f1": 0.6883116883116883, "accuracy": 0.6220472440944882, "c_at_1": 0.6603013206026412, "auroc": 0.6866319444444444, "auroc_c_at_1": 0.45338397968462607}, {"lower_threshold": 0.54, "upper_threshold": 0.66, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("121"), "n_unanswered_c_at_1": new NumberInt("6"), "precision": 0.6547619047619048, "recall": 0.873015873015873, "f1": 0.7482993197278912, "accuracy": 0.7086614173228346, "c_at_1": 0.7009114018228036, "auroc": 0.746155753968254, "auroc_c_at_1": 0.5229890754920399}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("99"), "n_unanswered_c_at_1": new NumberInt("28"), "precision": 0.6105263157894737, "recall": 0.90625, "f1": 0.7295597484276729, "accuracy": 0.6614173228346457, "c_at_1": 0.7015314030628061, "auroc": 0.7373511904761905, "auroc_c_at_1": 0.5172750152047924}, {"lower_threshold": 0.46, "upper_threshold": 0.62, "f1_threshold": 0.42, "n_answered_c_at_1": new NumberInt("102"), "n_unanswered_c_at_1": new NumberInt("25"), "precision": 0.5789473684210527, "recall": 0.859375, "f1": 0.6918238993710691, "accuracy": 0.6141732283464567, "c_at_1": 0.6314092628185256, "auroc": 0.6573660714285714, "auroc_c_at_1": 0.4150670265626245}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("95"), "n_unanswered_c_at_1": new NumberInt("31"), "precision": 0.5697674418604651, "recall": 0.7777777777777778, "f1": 0.6577181208053692, "accuracy": 0.5952380952380952, "c_at_1": 0.5834593096497858, "auroc": 0.6216931216931216, "auroc_c_at_1": 0.362732639597089}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("100"), "n_unanswered_c_at_1": new NumberInt("26"), "precision": 0.6067415730337079, "recall": 0.8571428571428571, "f1": 0.7105263157894737, "accuracy": 0.6507936507936508, "c_at_1": 0.6893424036281179, "auroc": 0.7155454774502394, "auroc_c_at_1": 0.4932558393307773}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("96"), "n_unanswered_c_at_1": new NumberInt("30"), "precision": 0.5978260869565217, "recall": 0.873015873015873, "f1": 0.7096774193548387, "accuracy": 0.6428571428571429, "c_at_1": 0.6681783824640967, "auroc": 0.7011841773746537, "auroc_c_at_1": 0.46851610944761435}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("103"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.6341463414634146, "recall": 0.8253968253968254, "f1": 0.7172413793103448, "accuracy": 0.6746031746031746, "c_at_1": 0.722663139329806, "auroc": 0.7431342907533384, "auroc_c_at_1": 0.5370357594994363}, {"lower_threshold": 0.52, "upper_threshold": 0.7, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("115"), "n_unanswered_c_at_1": new NumberInt("11"), "precision": 0.5760869565217391, "recall": 0.8412698412698413, "f1": 0.6838709677419355, "accuracy": 0.6111111111111112, "c_at_1": 0.6299445704207609, "auroc": 0.6599899218946838, "auroc_c_at_1": 0.4157570678299781}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("113"), "n_unanswered_c_at_1": new NumberInt("13"), "precision": 0.6195652173913043, "recall": 0.9047619047619048, "f1": 0.7354838709677419, "accuracy": 0.6746031746031746, "c_at_1": 0.7704711514235323, "auroc": 0.7816830435878055, "auroc_c_at_1": 0.6022642346413477}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.42, "n_answered_c_at_1": new NumberInt("96"), "n_unanswered_c_at_1": new NumberInt("31"), "precision": 0.5888888888888889, "recall": 0.8412698412698413, "f1": 0.6928104575163399, "accuracy": 0.6299212598425197, "c_at_1": 0.6563333126666253, "auroc": 0.6965525793650794, "auroc_c_at_1": 0.457170661861165}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("104"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.65, "recall": 0.8253968253968254, "f1": 0.7272727272727273, "accuracy": 0.6929133858267716, "c_at_1": 0.7533015066030132, "auroc": 0.7653769841269842, "auroc_c_at_1": 0.5765596352621277}, {"lower_threshold": 0.46, "upper_threshold": 0.62, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("107"), "n_unanswered_c_at_1": new NumberInt("20"), "precision": 0.5543478260869565, "recall": 0.796875, "f1": 0.6538461538461539, "accuracy": 0.5748031496062992, "c_at_1": 0.6653233306466613, "auroc": 0.6759672619047619, "auroc_c_at_1": 0.44973679009858014}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("107"), "n_unanswered_c_at_1": new NumberInt("20"), "precision": 0.6136363636363636, "recall": 0.84375, "f1": 0.7105263157894737, "accuracy": 0.6535433070866141, "c_at_1": 0.6835513671027342, "auroc": 0.7075892857142857, "auroc_c_at_1": 0.4836736235972472}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.44, "n_answered_c_at_1": new NumberInt("111"), "n_unanswered_c_at_1": new NumberInt("15"), "precision": 0.6, "recall": 0.7619047619047619, "f1": 0.6713286713286714, "accuracy": 0.626984126984127, "c_at_1": 0.6572184429327286, "auroc": 0.6807760141093475, "auroc_c_at_1": 0.44741855197889463}, {"lower_threshold": 0.54, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("119"), "n_unanswered_c_at_1": new NumberInt("7"), "precision": 0.6105263157894737, "recall": 0.9206349206349206, "f1": 0.7341772151898734, "accuracy": 0.6666666666666666, "c_at_1": 0.6869488536155203, "auroc": 0.7193247669438145, "auroc_c_at_1": 0.49413932402930466}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("94"), "n_unanswered_c_at_1": new NumberInt("32"), "precision": 0.5833333333333334, "recall": 0.8888888888888888, "f1": 0.7044025157232704, "accuracy": 0.626984126984127, "c_at_1": 0.646888384983623, "auroc": 0.6718316956412195, "auroc_c_at_1": 0.4346001205741574}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.44, "n_answered_c_at_1": new NumberInt("107"), "n_unanswered_c_at_1": new NumberInt("19"), "precision": 0.6818181818181818, "recall": 0.7142857142857143, "f1": 0.6976744186046512, "accuracy": 0.6904761904761905, "c_at_1": 0.70326278659612, "auroc": 0.7465356512975561, "auroc_c_at_1": 0.5250107424248686}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("99"), "n_unanswered_c_at_1": new NumberInt("27"), "precision": 0.5684210526315789, "recall": 0.8571428571428571, "f1": 0.6835443037974683, "accuracy": 0.6031746031746031, "c_at_1": 0.6842403628117913, "auroc": 0.6972789115646258, "auroc_c_at_1": 0.47710637542999057}, {"lower_threshold": 0.52, "upper_threshold": 0.7, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("116"), "n_unanswered_c_at_1": new NumberInt("10"), "precision": 0.59375, "recall": 0.9047619047619048, "f1": 0.7169811320754716, "accuracy": 0.6428571428571429, "c_at_1": 0.6681783824640967, "auroc": 0.6962711010330058, "auroc_c_at_1": 0.46523329804472946}, {"lower_threshold": 0.52, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("113"), "n_unanswered_c_at_1": new NumberInt("14"), "precision": 0.6091954022988506, "recall": 0.8412698412698413, "f1": 0.7066666666666667, "accuracy": 0.6535433070866141, "c_at_1": 0.6556513113026226, "auroc": 0.6811755952380952, "auroc_c_at_1": 0.4466136722452016}, {"lower_threshold": 0.52, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("110"), "n_unanswered_c_at_1": new NumberInt("17"), "precision": 0.6263736263736264, "recall": 0.9047619047619048, "f1": 0.7402597402597403, "accuracy": 0.6850393700787402, "c_at_1": 0.6606733213466426, "auroc": 0.7299107142857142, "auroc_c_at_1": 0.48223253589364307}, {"lower_threshold": 0.54, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("118"), "n_unanswered_c_at_1": new NumberInt("9"), "precision": 0.6082474226804123, "recall": 0.921875, "f1": 0.7329192546583851, "accuracy": 0.6614173228346457, "c_at_1": 0.6661293322586646, "auroc": 0.7098214285714286, "auroc_c_at_1": 0.47283287423717707}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("101"), "n_unanswered_c_at_1": new NumberInt("26"), "precision": 0.6236559139784946, "recall": 0.90625, "f1": 0.7388535031847133, "accuracy": 0.6771653543307087, "c_at_1": 0.7304234608469217, "auroc": 0.7514880952380953, "auroc_c_at_1": 0.5489045353090707}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("107"), "n_unanswered_c_at_1": new NumberInt("19"), "precision": 0.5978260869565217, "recall": 0.873015873015873, "f1": 0.7096774193548387, "accuracy": 0.6428571428571429, "c_at_1": 0.7306626354245401, "auroc": 0.7530864197530864, "auroc_c_at_1": 0.5502521081592215}, {"lower_threshold": 0.54, "upper_threshold": 0.66, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("111"), "n_unanswered_c_at_1": new NumberInt("15"), "precision": 0.6629213483146067, "recall": 0.9365079365079365, "f1": 0.7763157894736842, "accuracy": 0.7301587301587301, "c_at_1": 0.7637944066515495, "auroc": 0.8022171831695641, "auroc_c_at_1": 0.6127289974246747}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("104"), "n_unanswered_c_at_1": new NumberInt("22"), "precision": 0.5617977528089888, "recall": 0.7936507936507936, "f1": 0.6578947368421053, "accuracy": 0.5873015873015873, "c_at_1": 0.6525573192239859, "auroc": 0.6678004535147392, "auroc_c_at_1": 0.43577807372214017}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("99"), "n_unanswered_c_at_1": new NumberInt("27"), "precision": 0.5698924731182796, "recall": 0.8412698412698413, "f1": 0.6794871794871795, "accuracy": 0.6031746031746031, "c_at_1": 0.6071428571428571, "auroc": 0.6434870244394054, "auroc_c_at_1": 0.3906885505524961}, {"lower_threshold": 0.46, "upper_threshold": 0.7, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("105"), "n_unanswered_c_at_1": new NumberInt("21"), "precision": 0.6097560975609756, "recall": 0.7936507936507936, "f1": 0.6896551724137931, "accuracy": 0.6428571428571429, "c_at_1": 0.6666666666666666, "auroc": 0.690098261526833, "auroc_c_at_1": 0.4600655076845553}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("112"), "n_unanswered_c_at_1": new NumberInt("14"), "precision": 0.5681818181818182, "recall": 0.7936507936507936, "f1": 0.6622516556291391, "accuracy": 0.5952380952380952, "c_at_1": 0.6261022927689593, "auroc": 0.6315192743764172, "auroc_c_at_1": 0.39539566561486433}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("98"), "n_unanswered_c_at_1": new NumberInt("29"), "precision": 0.5670103092783505, "recall": 0.873015873015873, "f1": 0.6875, "accuracy": 0.6062992125984252, "c_at_1": 0.6673693347386694, "auroc": 0.6928323412698413, "auroc_c_at_1": 0.4623750586786888}, {"lower_threshold": 0.52, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("111"), "n_unanswered_c_at_1": new NumberInt("16"), "precision": 0.5934065934065934, "recall": 0.8571428571428571, "f1": 0.7012987012987013, "accuracy": 0.6377952755905512, "c_at_1": 0.6649513299026598, "auroc": 0.6896081349206349, "auroc_c_at_1": 0.458555846427169}, {"lower_threshold": 0.52, "upper_threshold": 0.7, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("119"), "n_unanswered_c_at_1": new NumberInt("8"), "precision": 0.5813953488372093, "recall": 0.78125, "f1": 0.6666666666666666, "accuracy": 0.6062992125984252, "c_at_1": 0.6026412052824106, "auroc": 0.6214037698412699, "auroc_c_at_1": 0.3744835168241766}, {"lower_threshold": 0.52, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("110"), "n_unanswered_c_at_1": new NumberInt("17"), "precision": 0.625, "recall": 0.859375, "f1": 0.7236842105263158, "accuracy": 0.6692913385826772, "c_at_1": 0.6785293570587141, "auroc": 0.7114335317460317, "auroc_c_at_1": 0.48272853688564515}, {"lower_threshold": 0.46, "upper_threshold": 0.62, "f1_threshold": 0.44, "n_answered_c_at_1": new NumberInt("103"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.5679012345679012, "recall": 0.7301587301587301, "f1": 0.6388888888888888, "accuracy": 0.5873015873015873, "c_at_1": 0.6194255479969766, "auroc": 0.6720836482741244, "auroc_c_at_1": 0.41630578213200675}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("100"), "n_unanswered_c_at_1": new NumberInt("26"), "precision": 0.6067415730337079, "recall": 0.8571428571428571, "f1": 0.7105263157894737, "accuracy": 0.6507936507936508, "c_at_1": 0.7180650037792895, "auroc": 0.7391030486268582, "auroc_c_at_1": 0.5307240334055293}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("103"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.632183908045977, "recall": 0.873015873015873, "f1": 0.7333333333333333, "accuracy": 0.6825396825396826, "c_at_1": 0.7414336104812295, "auroc": 0.7510707986898464, "auroc_c_at_1": 0.5568691339996334}, {"lower_threshold": 0.52, "upper_threshold": 0.66, "f1_threshold": 0.44, "n_answered_c_at_1": new NumberInt("117"), "n_unanswered_c_at_1": new NumberInt("9"), "precision": 0.6235294117647059, "recall": 0.8412698412698413, "f1": 0.7162162162162162, "accuracy": 0.6666666666666666, "c_at_1": 0.6717687074829931, "auroc": 0.7122700932224741, "auroc_c_at_1": 0.4784807599028525}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("103"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.6086956521739131, "recall": 0.8888888888888888, "f1": 0.7225806451612903, "accuracy": 0.6587301587301587, "c_at_1": 0.7414336104812295, "auroc": 0.7590073066263542, "auroc_c_at_1": 0.5627535277336114}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("103"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.6162790697674418, "recall": 0.8412698412698413, "f1": 0.7114093959731543, "accuracy": 0.6587301587301587, "c_at_1": 0.6757369614512471, "auroc": 0.6985386747291509, "auroc_c_at_1": 0.47202840151765746}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("104"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.5757575757575758, "recall": 0.9047619047619048, "f1": 0.7037037037037037, "accuracy": 0.6220472440944882, "c_at_1": 0.69750139500279, "auroc": 0.7177579365079365, "auroc_c_at_1": 0.5006371619886096}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("98"), "n_unanswered_c_at_1": new NumberInt("29"), "precision": 0.5416666666666666, "recall": 0.8253968253968254, "f1": 0.6540880503144654, "accuracy": 0.5669291338582677, "c_at_1": 0.5803211606423212, "auroc": 0.6010664682539683, "auroc_c_at_1": 0.3488115904803238}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("110"), "n_unanswered_c_at_1": new NumberInt("17"), "precision": 0.627906976744186, "recall": 0.84375, "f1": 0.72, "accuracy": 0.6692913385826772, "c_at_1": 0.7053134106268212, "auroc": 0.7154017857142857, "auroc_c_at_1": 0.5045824734506611}, {"lower_threshold": 0.46, "upper_threshold": 0.62, "f1_threshold": 0.42, "n_answered_c_at_1": new NumberInt("104"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.5604395604395604, "recall": 0.796875, "f1": 0.6580645161290323, "accuracy": 0.5826771653543307, "c_at_1": 0.651001302002604, "auroc": 0.6791914682539684, "auroc_c_at_1": 0.4421545301423937}, {"lower_threshold": 0.54, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("118"), "n_unanswered_c_at_1": new NumberInt("8"), "precision": 0.6021505376344086, "recall": 0.8888888888888888, "f1": 0.717948717948718, "accuracy": 0.6507936507936508, "c_at_1": 0.675233056185437, "auroc": 0.7088687326782566, "auroc_c_at_1": 0.4786516008006368}, {"lower_threshold": 0.52, "upper_threshold": 0.7, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("108"), "n_unanswered_c_at_1": new NumberInt("18"), "precision": 0.6588235294117647, "recall": 0.8888888888888888, "f1": 0.7567567567567568, "accuracy": 0.7142857142857143, "c_at_1": 0.6893424036281179, "auroc": 0.7466616276140086, "auroc_c_at_1": 0.5147055210763234}, {"lower_threshold": 0.46, "upper_threshold": 0.62, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("103"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.6, "recall": 0.8095238095238095, "f1": 0.6891891891891891, "accuracy": 0.6349206349206349, "c_at_1": 0.6757369614512471, "auroc": 0.7117661879566641, "auroc_c_at_1": 0.4809667211135734}, {"lower_threshold": 0.52, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("111"), "n_unanswered_c_at_1": new NumberInt("15"), "precision": 0.6352941176470588, "recall": 0.8571428571428571, "f1": 0.7297297297297297, "accuracy": 0.6825396825396826, "c_at_1": 0.6838624338624338, "auroc": 0.719702695893172, "auroc_c_at_1": 0.4921776372708597}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("107"), "n_unanswered_c_at_1": new NumberInt("19"), "precision": 0.6352941176470588, "recall": 0.8571428571428571, "f1": 0.7297297297297297, "accuracy": 0.6825396825396826, "c_at_1": 0.70326278659612, "auroc": 0.7242378432854624, "auroc_c_at_1": 0.5093295238272983}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("103"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.6206896551724138, "recall": 0.8571428571428571, "f1": 0.72, "accuracy": 0.6666666666666666, "c_at_1": 0.7132779037540942, "auroc": 0.7321743512219703, "auroc_c_at_1": 0.5222437864221209}, {"lower_threshold": 0.46, "upper_threshold": 0.62, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("106"), "n_unanswered_c_at_1": new NumberInt("21"), "precision": 0.6363636363636364, "recall": 0.8888888888888888, "f1": 0.7417218543046358, "accuracy": 0.6929133858267716, "c_at_1": 0.7340814681629364, "auroc": 0.7513640873015873, "auroc_c_at_1": 0.5515624523312539}, {"lower_threshold": 0.52, "upper_threshold": 0.7, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("117"), "n_unanswered_c_at_1": new NumberInt("10"), "precision": 0.627906976744186, "recall": 0.8571428571428571, "f1": 0.7248322147651006, "accuracy": 0.6771653543307087, "c_at_1": 0.6540393080786161, "auroc": 0.6996527777777778, "auroc_c_at_1": 0.45760041867305956}, {"lower_threshold": 0.52, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("110"), "n_unanswered_c_at_1": new NumberInt("17"), "precision": 0.5858585858585859, "recall": 0.90625, "f1": 0.7116564417177914, "accuracy": 0.6299212598425197, "c_at_1": 0.6785293570587141, "auroc": 0.7023809523809523, "auroc_c_at_1": 0.4765860960293349}, {"lower_threshold": 0.46, "upper_threshold": 0.62, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("105"), "n_unanswered_c_at_1": new NumberInt("22"), "precision": 0.5930232558139535, "recall": 0.796875, "f1": 0.68, "accuracy": 0.6220472440944882, "c_at_1": 0.6928513857027714, "auroc": 0.7185019841269842, "auroc_c_at_1": 0.4978150953325716}, {"lower_threshold": 0.52, "upper_threshold": 0.7, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("113"), "n_unanswered_c_at_1": new NumberInt("13"), "precision": 0.6626506024096386, "recall": 0.873015873015873, "f1": 0.7534246575342466, "accuracy": 0.7142857142857143, "c_at_1": 0.6916729654824892, "auroc": 0.7383471907281431, "auroc_c_at_1": 0.5106947909665999}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("103"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.5806451612903226, "recall": 0.8571428571428571, "f1": 0.6923076923076923, "accuracy": 0.6190476190476191, "c_at_1": 0.6757369614512471, "auroc": 0.6882086167800454, "auroc_c_at_1": 0.46504799954751364}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.44, "n_answered_c_at_1": new NumberInt("105"), "n_unanswered_c_at_1": new NumberInt("21"), "precision": 0.5662650602409639, "recall": 0.746031746031746, "f1": 0.6438356164383562, "accuracy": 0.5873015873015873, "c_at_1": 0.6203703703703703, "auroc": 0.6507936507936508, "auroc_c_at_1": 0.4037330981775426}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.44, "n_answered_c_at_1": new NumberInt("99"), "n_unanswered_c_at_1": new NumberInt("27"), "precision": 0.5975609756097561, "recall": 0.7777777777777778, "f1": 0.6758620689655173, "accuracy": 0.626984126984127, "c_at_1": 0.6360544217687074, "auroc": 0.6757369614512472, "auroc_c_at_1": 0.42980548228361637}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("100"), "n_unanswered_c_at_1": new NumberInt("26"), "precision": 0.6235294117647059, "recall": 0.8412698412698413, "f1": 0.7162162162162162, "accuracy": 0.6666666666666666, "c_at_1": 0.7372134038800705, "auroc": 0.7630385487528344, "auroc_c_at_1": 0.5625222458177862}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("101"), "n_unanswered_c_at_1": new NumberInt("25"), "precision": 0.5656565656565656, "recall": 0.8888888888888888, "f1": 0.691358024691358, "accuracy": 0.6031746031746031, "c_at_1": 0.6372511967750063, "auroc": 0.6602418745275889, "auroc_c_at_1": 0.42073992470367955}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("104"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.6179775280898876, "recall": 0.873015873015873, "f1": 0.7236842105263158, "accuracy": 0.6692913385826772, "c_at_1": 0.69750139500279, "auroc": 0.7212301587301588, "auroc_c_at_1": 0.5030590418323694}, {"lower_threshold": 0.54, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("113"), "n_unanswered_c_at_1": new NumberInt("14"), "precision": 0.5862068965517241, "recall": 0.8095238095238095, "f1": 0.68, "accuracy": 0.6220472440944882, "c_at_1": 0.6294252588505177, "auroc": 0.661954365079365, "auroc_c_at_1": 0.4166507975873094}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("103"), "n_unanswered_c_at_1": new NumberInt("24"), "precision": 0.625, "recall": 0.859375, "f1": 0.7236842105263158, "accuracy": 0.6692913385826772, "c_at_1": 0.7208754417508835, "auroc": 0.7410714285714286, "auroc_c_at_1": 0.5342201934403868}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("107"), "n_unanswered_c_at_1": new NumberInt("20"), "precision": 0.6263736263736264, "recall": 0.890625, "f1": 0.7354838709677419, "accuracy": 0.6771653543307087, "c_at_1": 0.7017794035588071, "auroc": 0.7290426587301588, "auroc_c_at_1": 0.5116271222125778}, {"lower_threshold": 0.46, "upper_threshold": 0.62, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("101"), "n_unanswered_c_at_1": new NumberInt("25"), "precision": 0.5806451612903226, "recall": 0.8571428571428571, "f1": 0.6923076923076923, "accuracy": 0.6190476190476191, "c_at_1": 0.6467624086671705, "auroc": 0.6696900982615268, "auroc_c_at_1": 0.43313038101217916}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("108"), "n_unanswered_c_at_1": new NumberInt("18"), "precision": 0.6153846153846154, "recall": 0.8888888888888888, "f1": 0.7272727272727273, "accuracy": 0.6666666666666666, "c_at_1": 0.6893424036281179, "auroc": 0.7136558327034517, "auroc_c_at_1": 0.4919532270790234}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.26, "n_answered_c_at_1": new NumberInt("97"), "n_unanswered_c_at_1": new NumberInt("29"), "precision": 0.6, "recall": 0.8571428571428571, "f1": 0.7058823529411765, "accuracy": 0.6428571428571429, "c_at_1": 0.6541320231796423, "auroc": 0.6870748299319729, "auroc_c_at_1": 0.44943764857921004}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("103"), "n_unanswered_c_at_1": new NumberInt("23"), "precision": 0.5567010309278351, "recall": 0.8571428571428571, "f1": 0.675, "accuracy": 0.5873015873015873, "c_at_1": 0.6475812547241119, "auroc": 0.6727135298563871, "auroc_c_at_1": 0.43563667173428544}, {"lower_threshold": 0.52, "upper_threshold": 0.7, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("111"), "n_unanswered_c_at_1": new NumberInt("15"), "precision": 0.6046511627906976, "recall": 0.8253968253968254, "f1": 0.697986577181208, "accuracy": 0.6428571428571429, "c_at_1": 0.6572184429327286, "auroc": 0.68190980095742, "auroc_c_at_1": 0.4481636976058025}, {"lower_threshold": 0.46, "upper_threshold": 0.66, "f1_threshold": 0.32, "n_answered_c_at_1": new NumberInt("104"), "n_unanswered_c_at_1": new NumberInt("22"), "precision": 0.6292134831460674, "recall": 0.8888888888888888, "f1": 0.7368421052631579, "accuracy": 0.6825396825396826, "c_at_1": 0.7551020408163265, "auroc": 0.7709750566893425, "auroc_c_at_1": 0.5821648387246056}]
            for key in ("split_idx", "split_id", "fold_idx"):
                if key in d:
                    try:
                        return int(d[key])
                    except (TypeError, ValueError):
                        break
            split = d.get("split")
            if isinstance(split, dict):
                r = split.get("repeat")
                f = split.get("fold")
                if r is not None and f is not None:
                    try:
                        return int(r) * 10_000 + int(f)
                    except (TypeError, ValueError):
                        pass
            return 0

        for method_name, rows in per_split_by_method.items():
            if any(k in (rows[0] if rows else {}) for k in ("split_idx", "split_id", "fold_idx", "split")):
                per_split_by_method[method_name] = sorted(rows, key=sort_key)

        return per_split_by_method, None

    logger.warning(
        "No documents found in MongoDB collection '%s' for dataset=%s and methods=%s. Falling back to "
        "`pan_metrics.pan_metrics_per_split`.",
        per_split_feature_name,
        dataset_name,
        list(methods),
    )
    pan_docs = _load_pan_metrics_docs(store, dataset_name)
    per_split_by_method = {
        doc["method_name"]: list(doc.get("pan_metrics_per_split") or [])
        for doc in pan_docs
        if doc.get("method_name") in set(methods)
    }
    split_config = None
    for doc in pan_docs:
        if doc.get("split_config"):
            split_config = doc.get("split_config")
            break
    return per_split_by_method, split_config


def plot_pan_metrics_heatmap_per_split(
    dataset_names: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    scores: Sequence[str] | None = None,
    save_path: Path | None = None,
    *,
    store: PANMetricsStore | None = None,
    per_split_collection_name: str = "pan_metrics_per_split",
) -> dict[str, tuple[plt.Figure, plt.Axes]]:
    """
    Create and save one heatmap figure per dataset.

    - Y axis: CV splits (folds × repeats).
    - X axis: methods (repeated once per metric score).
    - Blocks: one contiguous block per metric score (from `pan_metrics.metrics_mean` keys).

    Figures are saved as PDF and SVG to `results/reproduction/pan_metrics/heatmaps`.
    """
    store = store or PANMetricsStore()
    save_path = save_path or _default_heatmap_dir()
    save_path.mkdir(parents=True, exist_ok=True)

    if dataset_names is None:
        dataset_names = sorted(store.mongo.pan_metrics_collection.distinct("dataset_name"))
    if not dataset_names:
        raise ValueError("No datasets found in MongoDB collection 'pan_metrics'.")

    figures: dict[str, tuple[plt.Figure, plt.Axes]] = {}

    for dataset_name in dataset_names:
        pan_docs = _load_pan_metrics_docs(store, dataset_name)
        if not pan_docs:
            logger.warning("No PAN metrics found for dataset=%s; skipping.", dataset_name)
            continue

        available_methods = sorted({doc.get("method_name") for doc in pan_docs if doc.get("method_name")})
        if not available_methods:
            logger.warning("No methods found for dataset=%s; skipping.", dataset_name)
            continue

        if methods is None:
            methods_list = _ordered_methods(available_methods)
        else:
            methods_list = [m for m in methods if m in set(available_methods)]
            missing = [m for m in methods if m not in set(available_methods)]
            if missing:
                logger.warning("Missing PAN metrics for methods (dataset=%s): %s", dataset_name, ", ".join(missing))

        available_scores = sorted(
            {
                metric
                for doc in pan_docs
                for metric in (doc.get("metrics_mean") or {}).keys()
            }
        )
        scores_list = _valid_metric_names(scores or available_scores)
        if not scores_list:
            raise ValueError(
                f"No valid metric scores found for dataset={dataset_name}. "
                f"Expected subset of: {sorted(CONFIG.SCORE_TRANSLATIONS.keys())}."
            )

        split_config = None
        for doc in pan_docs:
            if doc.get("split_config"):
                split_config = doc.get("split_config")
                break

        per_split_by_method, split_config_fallback = _load_per_split_by_method(
            store=store,
            dataset_name=dataset_name,
            methods=methods_list,
            per_split_feature_name=per_split_collection_name,
        )
        if split_config is None:
            split_config = split_config_fallback

        n_splits_total = 0
        for method_name in methods_list:
            n_splits_total = max(n_splits_total, len(per_split_by_method.get(method_name, [])))
        if n_splits_total == 0:
            raise ValueError(
                f"No per-split PAN metrics found for dataset={dataset_name}. "
                f"Tried MongoDB collection '{per_split_collection_name}' and embedded `pan_metrics_per_split`."
            )

        split_labels = _split_labels(n_splits_total, split_config)

        method_labels = [CONFIG.LABEL_TRANSLATIONS.get(m, m) for m in methods_list]
        xticklabels = method_labels * len(scores_list)

        matrix = np.full((n_splits_total, len(methods_list) * len(scores_list)), np.nan, dtype=float)
        for score_idx, score in enumerate(scores_list):
            for method_idx, method_name in enumerate(methods_list):
                per_split_rows = per_split_by_method.get(method_name, [])
                col = score_idx * len(methods_list) + method_idx
                for split_idx in range(min(n_splits_total, len(per_split_rows))):
                    raw = per_split_rows[split_idx].get(score)
                    if raw is None:
                        continue
                    try:
                        matrix[split_idx, col] = float(raw)
                    except (TypeError, ValueError):
                        continue

        # Figure sizing: keep readable but avoid extreme layouts.
        fig_w = max(14.0, min(60.0, 0.22 * matrix.shape[1]))
        fig_h = max(8.0, min(30.0, 0.20 * matrix.shape[0]))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.set_theme(context="paper", style="white")
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            xticklabels=xticklabels,
            yticklabels=split_labels,
            cbar_kws={"label": "Score"},
            # annot=True,
        )

        # Reduce y tick density for large numbers of splits.
        max_yticks = 30
        if n_splits_total > max_yticks:
            step = int(math.ceil(n_splits_total / max_yticks))
            ax.set_yticks([i + 0.5 for i in range(0, n_splits_total, step)])
            ax.set_yticklabels([split_labels[i] for i in range(0, n_splits_total, step)], fontsize=12)
        else:
            ax.tick_params(axis="y", labelsize=12)

        ax.tick_params(axis="x", labelrotation=90, labelsize=12)
        ax.set_xlabel("Method", fontsize=15)
        ax.set_ylabel("Split", fontsize=15)

        dataset_label = CONFIG.DATASET_TRANSLATIONS.get(dataset_name, dataset_name)
        ax.set_title(f"PAN metrics per split - {dataset_label}", fontsize=16)

        # Draw vertical separators between score blocks.
        n_methods = len(methods_list)
        for i in range(1, len(scores_list)):
            ax.axvline(i * n_methods, color="white", linewidth=3)
            ax.axvline(i * n_methods, color="black", linewidth=0.6, alpha=0.5)

        # Add a top axis with score labels centered per block.
        top = ax.secondary_xaxis("top")
        centers = [(i * n_methods) + (n_methods / 2) for i in range(len(scores_list))]
        top.set_xticks(centers)
        top.set_xticklabels([CONFIG.SCORE_TRANSLATIONS.get(s, s) for s in scores_list], fontsize=10)
        top.tick_params(axis="x", length=0)
        top.set_xlabel("Metric", fontsize=12)

        fig.tight_layout()

        safe_dataset = str(dataset_name).replace(" ", "_").replace("/", "_")
        base = f"pan_metrics_heatmap_per_split_{safe_dataset}"
        for fmt in ["pdf", "svg"]:
            out_path = save_path / f"{base}.{fmt}"
            fig.savefig(out_path, bbox_inches="tight")
            logger.info("Saved heatmap to %s", out_path)

        figures[dataset_name] = (fig, ax)

    return figures


def plot_pan_metrics_boxplots(
    dataset_name: str | None = None,
    methods: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    save_path: Path | None = None,
    title: str | None = None,
    *,
    store: PANMetricsStore | None = None,
):
    """
    Plot per-fold PAN metrics as grouped boxplots (grouped by metric, colored by method).
    """
    store = store or PANMetricsStore()

    query: dict = {}
    if dataset_name is not None:
        query["dataset_name"] = dataset_name
    if methods is not None:
        query["method_name"] = {"$in": list(methods)}

    cursor = store.mongo.pan_metrics_collection.find(query, {"_id": 0})

    metrics_by_method: dict[str, dict[str, list[float]]] = {}
    split_config: dict[str, float | int] | None = None
    for doc in cursor:
        normalized = store._normalize_doc(doc)
        method_name = normalized.get("method_name")
        if not method_name:
            continue
        normalized = store.ensure_metric_values(
            normalized,
            PANMetricComputer.extract_metric_values,
        )
        metric_values = normalized.get("metric_values") or {}
        if metric_values:
            metrics_by_method[method_name] = metric_values
            if split_config is None:
                split_config = normalized.get("split_config")

    if not metrics_by_method:
        raise ValueError("No PAN metrics found for the given query.")

    if methods is None:
        methods_list = sorted(metrics_by_method.keys())
    else:
        methods_list = [method for method in methods if method in metrics_by_method]
        missing = [method for method in methods if method not in metrics_by_method]
        if missing:
            logger.warning("Missing PAN metrics for methods: %s", ", ".join(missing))

    if metrics is None:
        available = {m for vals in metrics_by_method.values() for m in vals.keys()}
        metrics_list = [m for m in list(CONFIG.SCORE_TRANSLATIONS.keys()) if m in available]
        metrics_list += sorted(m for m in available if m not in metrics_list)
    else:
        metrics_list = list(metrics)

    n_methods = len(methods_list)
    n_metrics = len(metrics_list)

    box_data: list[list[float]] = []
    box_positions: list[float] = []
    box_method_idx: list[int] = []

    for metric_idx, metric in enumerate(metrics_list):
        group_start = metric_idx * (n_methods + 1) + 1
        for method_idx, method in enumerate(methods_list):
            values = metrics_by_method.get(method, {}).get(metric, [])
            if not values:
                continue
            box_positions.append(group_start + method_idx)
            box_data.append(values)
            box_method_idx.append(method_idx)

    if not box_data:
        raise ValueError("No metric values found for plotting.")

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 2.5), 6))
    boxplot = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
    )

    colors = [
        CONFIG.LABEL_COLORS.get(method, "#4d4d4d")
        for method in methods_list
    ]
    for patch, method_idx in zip(boxplot["boxes"], box_method_idx):
        patch.set_facecolor(colors[method_idx])
        patch.set_edgecolor("black")

    group_centers = []
    for metric_idx in range(n_metrics):
        group_start = metric_idx * (n_methods + 1) + 1
        center = group_start + (n_methods - 1) / 2
        group_centers.append(center)

    metric_labels = [CONFIG.SCORE_TRANSLATIONS.get(m, m) for m in metrics_list]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(metric_labels, rotation=0)
    ax.set_xlabel("Metric", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.tick_params(axis="both", labelsize=13)

    if title is None:
        n_splits = None
        n_repeats = None
        if split_config and "n_splits" in split_config and "n_repeats" in split_config:
            n_splits = split_config.get("n_splits")
            n_repeats = split_config.get("n_repeats")
            title = f"PAN Metrics ({int(n_splits)} Folds, {int(n_repeats)} Repetitions)"
        else:
            title = f"PAN Metrics"
        if dataset_name is not None:
            dataset_label = CONFIG.DATASET_TRANSLATIONS.get(dataset_name, dataset_name)
            title = f"{title} - {dataset_label}"
    ax.set_title(title, fontsize=16)

    legend_handles = []
    for i, method in enumerate(methods_list):
        label = CONFIG.LABEL_TRANSLATIONS.get(method, method)
        legend_handles.append(
            plt.Line2D([0], [0], color=colors[i], lw=6, label=label)
        )
    ax.legend(
        handles=legend_handles,
        title="Technique",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=13,
        title_fontsize=13,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=(0, 0, 1, 1))

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        suffix = dataset_name if dataset_name is not None else "all"
        for format in ["svg", "pdf"]:
            out_path = save_path / f"pan_metrics_boxplot_{suffix}.{format}"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            logger.info("Saved boxplot to %s", out_path)

    return fig, ax
