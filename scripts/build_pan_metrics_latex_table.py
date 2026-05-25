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

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)


LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[1] / CONFIG.SAVE_PATH / "reproduction"
)

OUTPUT_DIR = LOCAL_SAVE_PATH / "pan_metrics_tex"

CITE_ALIAS_LINES = [
    r"\defcitealias{koppel_determining_2014}{Koppel, Winter} % in-domain (Blog Posts updated, 01.04.2026)",
    r"\defcitealias{khonji_slightly_modified_2014}{Khonji, Iraqi} % ASGALF",
    r"\defcitealias{gutierrez2015homotopy}{Hern{\'a}ndez et al.}  % Homotopy (Blog Posts updated, 01.04.2026)",
    r"\defcitealias{kestemont_authenticating_2016}{Kestemont et al.}  % std impostors",
    r"\defcitealias{potha_improved_2017}{Potha, Stamatat.}    % potha 2017 (Blog Posts updated, 01.04.2026)",
    r"\defcitealias{nagy_bootstrap_2024}{Nagy}    % bdi (Blog Posts updated, 01.04.2026)",
]


METRIC_ORDER = [
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auroc",
    "c_at_1",
    "auroc_c_at_1",
]


METRIC_LABELS = {
    "accuracy": "Acc.",
    "f1": "F1",
    "precision": "Prec.",
    "recall": "Rec.",
    "auroc": "AUROC",
    "c_at_1": "c@1",
    "auroc_c_at_1": (
        r"\raisebox{0.75ex}[0em][0em]{\begin{tabular}{@{}c@{}}AUROC\\[-0.5ex]$\times$~c@1\end{tabular}}"
    ),
}


@dataclass(frozen=True)
class RowSpec:
    label: str
    year: str
    method_keys: Sequence[str]
    is_baseline: bool = False


ROW_SPECS: Sequence[RowSpec] = [
    RowSpec(
        label=r"\citetalias{koppel_determining_2014}",
        year=r"\citeyear{koppel_determining_2014}",
        method_keys=("in_domain",),
    ),
    RowSpec(
        label=r"\citetalias{khonji_slightly_modified_2014}",
        year=r"\citeyear{khonji_slightly_modified_2014}",
        method_keys=("asgalf",),
    ),
    RowSpec(
        label=r"\citetalias{gutierrez2015homotopy}",
        year=r"\citeyear{gutierrez2015homotopy}",
        method_keys=("homotopy",),
    ),
    RowSpec(
        label=r"\citetalias{kestemont_authenticating_2016}",
        year=r"\citeyear{kestemont_authenticating_2016}",
        method_keys=("std_impostor",),
    ),
    RowSpec(
        label=r"\citetalias{potha_improved_2017}",
        year=r"\citeyear{potha_improved_2017}",
        method_keys=("potha2017",),
    ),
    RowSpec(
        label=r"\citetalias{nagy_bootstrap_2024}",
        year=r"\citeyear{nagy_bootstrap_2024}",
        method_keys=("bdi",),
    ),
    RowSpec(
        label="LLM impostors",
        year="2026",
        method_keys=("two_step_llm", "one_step_llm"),
    ),
    RowSpec(
        label="Min-max (B)",
        year="",
        method_keys=("unsupervised_baseline_min-max",),
        is_baseline=True,
    ),
    RowSpec(
        label="Cosine (B)",
        year="",
        method_keys=("unsupervised_baseline_cosine",),
        is_baseline=True,
    ),
    # RowSpec(
    #     label="  SVM (B)",
    #     year="",
    #     method_keys=("supervised_baseline",),
    # ),
    RowSpec(
        label="Unmasking (B)",
        year="",
        method_keys=("unmasking",),
        is_baseline=True,
    ),
    RowSpec(
        label="PPMd (B)",
        year="",
        method_keys=("ppmd",),
        is_baseline=True,
    ),
]


DATASET_DISPLAY = {
    CONFIG.BLOG: "Blogs",
    CONFIG.STUDENT_ESSAYS: "Student Essays",
}


def _extract_metrics_mean(
    doc: dict,
    method_key: str,
) -> dict[str, float]:
    if not isinstance(doc, dict):
        return {}
    if "metrics_mean" in doc:
        return doc.get("metrics_mean", {}) or {}
    pan_metrics = doc.get("pan_metrics", {})
    if "metrics_mean" in pan_metrics:
        return pan_metrics.get("metrics_mean", {}) or {}
    nested = pan_metrics.get(method_key)
    if isinstance(nested, dict) and "metrics_mean" in nested:
        return nested.get("metrics_mean", {}) or {}
    return {}


def _load_metrics_for_method(
    mongo: ParaphraseMongoDB,
    dataset_name: str,
    method_keys: Sequence[str],
) -> dict[str, float] | None:
    for method_key in method_keys:
        docs = list(
            mongo.pan_metrics_collection.find(
                {"dataset_name": dataset_name, "method_name": method_key},
                {"_id": 0, "pan_metrics": 1, "metrics_mean": 1, "n_samples": 1},
            )
        )
        if not docs:
            continue
        best_doc = max(docs, key=lambda item: _n_samples_value(item.get("n_samples")))
        metrics_mean = _extract_metrics_mean(best_doc, method_key)
        if metrics_mean:
            return metrics_mean
        logger.warning(
            "No metrics_mean in pan_metrics for dataset=%s method=%s.",
            dataset_name,
            method_key,
        )
    return None


def _n_samples_value(value: float | int | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _round_metric(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not (value_f == value_f):  # NaN check
        return None
    return round(value_f, 3)


def _format_metric(value: float | None, bold: bool) -> str:
    if value is None:
        return "--"
    formatted = f"{value:.3f}"
    if bold:
        return r"\bf " + formatted
    return formatted


def _dataset_title(dataset_name: str) -> str:
    return f"{DATASET_DISPLAY.get(dataset_name, dataset_name)} Dataset"


def _collect_dataset_metrics(
    mongo: ParaphraseMongoDB,
    dataset_name: str,
    rows: Sequence[RowSpec],
) -> list[tuple[RowSpec, dict[str, float] | None]]:
    collected: list[tuple[RowSpec, dict[str, float] | None]] = []
    for row in rows:
        metrics = _load_metrics_for_method(mongo, dataset_name, row.method_keys)
        if metrics is None:
            logger.warning(
                "Missing pan_metrics for dataset=%s methods=%s.",
                dataset_name,
                ", ".join(row.method_keys),
            )
        collected.append((row, metrics))
    return collected


def _compute_bold_maxima(
    rows_with_metrics: list[tuple[RowSpec, dict[str, float] | None]],
) -> dict[str, float]:
    maxima: dict[str, float] = {}
    for metric in METRIC_ORDER:
        values = []
        for _, metrics in rows_with_metrics:
            if not metrics:
                continue
            value = _round_metric(metrics.get(metric))
            if value is not None:
                values.append(value)
        if values:
            maxima[metric] = max(values)
    return maxima


def build_table(
    mongo: ParaphraseMongoDB,
    datasets: Iterable[str],
    bold_max: bool = True,
) -> str:
    lines: list[str] = []
    lines.append("% Auto-generated; do not edit by hand.")
    lines.extend(CITE_ALIAS_LINES)
    lines.append("")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\fontsize{7.5pt}{8pt}\selectfont")
    lines.append(r"\renewcommand{\tabcolsep}{2pt}")
    lines.append(r"\begin{tabular}{@{}lccccc@{}c@{}c@{}c@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"\multicolumn{2}{@{}l@{}}{\textbf{Impostors Method}} & \multicolumn{7}{@{}c@{}}{\textbf{Reproduction Scores}} \\"
    )
    lines.append(
        r"\cmidrule(r@{\tabcolsep}){1-2}\cmidrule(l@{\tabcolsep}){3-8}"
    )
    lines.append(
        r"Authors & Year & "
        + " & ".join(METRIC_LABELS[metric] for metric in METRIC_ORDER)
        + r" \\"
    )
    lines.append(r"\midrule")

    dataset_list = list(datasets)
    for idx, dataset_name in enumerate(dataset_list):
        rows_with_metrics = _collect_dataset_metrics(mongo, dataset_name, ROW_SPECS)
        maxima = _compute_bold_maxima(rows_with_metrics) if bold_max else {}
        baseline_row_count = sum(1 for row, _ in rows_with_metrics if row.is_baseline)
        baseline_label_written = False

        lines.append(
            f"  \\multicolumn{{9}}{{@{{}}c@{{}}}}{{\\emph{{{_dataset_title(dataset_name)}}}}}  \\\\"
        )
        for row_index, (row, metrics) in enumerate(rows_with_metrics):
            if row_index > 0:
                previous_row = rows_with_metrics[row_index - 1][0]
                if row.method_keys == ("two_step_llm", "one_step_llm"):
                    lines.append(r"\addlinespace[0.75ex]")
                elif row.is_baseline and not previous_row.is_baseline:
                    lines.append(r"\addlinespace[0.5ex]")

            cells: list[str] = []
            for metric in METRIC_ORDER:
                value = _round_metric(metrics.get(metric) if metrics else None)
                is_bold = bold_max and (value is not None) and (value == maxima.get(metric))
                cells.append(_format_metric(value, is_bold))

            year = row.year
            if row.is_baseline and not baseline_label_written:
                year = (
                    rf"\multirow{{{baseline_row_count}}}{{*}}"
                    r"{\rotatebox[origin=l]{-90}{Baselines}}"
                )
                baseline_label_written = True
            elif row.is_baseline:
                year = ""

            lines.append(
                f"  {row.label} & {year} & " + " & ".join(cells) + r" \\"
            )

        if idx < len(dataset_list) - 1:
            lines.append(r"\midrule")
        else:
            lines.append(r"\bottomrule")

    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Reproduction\,of\,the\,Impostors\,Method\,variants and baselines. The scores are computed as mean "
        r"over ten 10-fold CVs.}"
    )
    lines.append(r"\label{table-impostor-method-reproduction-results}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "pan_metrics_impostor_table.tex"

    mongo = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))

    latex = build_table(
        mongo=mongo,
        datasets=[CONFIG.BLOG, CONFIG.STUDENT_ESSAYS],
        bold_max=True
    )

    out_path.write_text(latex, encoding="utf-8")
    print(f"Wrote LaTeX table to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
