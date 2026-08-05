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
"""Run deeper descriptive analyses for authorship-verification datasets.

The existing :mod:`datasets_visualization` module already knows how to load
MongoDB-backed pair datasets. This module reuses that loading path and adds
per-text measurements for style, surface form, language, readability, and a
lightweight register assessment.
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from langdetect import DetectorFactory, LangDetectException, detect_langs

from genai_detection.config import CONFIG
from genai_detection.visualization.datasets_visualization import (
    BlogVisualization,
    StudentEssaysVisualization,
)

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVE_BASE = PROJECT_ROOT / CONFIG.SAVE_PATH / "datasets"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")
EMOTICON_RE = re.compile(
    r"(?:(?<=\s)|^)(?:[:;=8xX][-o*']?[\)\]\(\[dDpPoO0/:\}\{@\|\\]|[<>]?3|[xX]'?D)(?=\s|$)"
)
EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),
    (0x2600, 0x27BF),
)
INFORMAL_MARKERS = {
    "lol",
    "omg",
    "btw",
    "idk",
    "yeah",
    "yep",
    "nope",
    "gonna",
    "wanna",
    "kinda",
    "sorta",
}
FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
ACADEMIC_MARKERS = {
    "therefore",
    "however",
    "moreover",
    "furthermore",
    "nevertheless",
    "consequently",
    "hypothesis",
    "analysis",
    "evidence",
    "argument",
    "conclusion",
}


@dataclass(frozen=True)
class DatasetAnalysisOutput:
    """Paths written for one analyzed dataset."""

    dataset_name: str
    per_text_path: Path
    summary_path: Path
    language_distribution_path: Path
    register_distribution_path: Path


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return a stable ratio for empty or degenerate texts."""
    return 0.0 if denominator == 0 else numerator / denominator


def _is_emoji(char: str) -> bool:
    """Detect common emoji codepoint ranges without adding another dependency."""
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in EMOJI_RANGES)


def _tokenize_words(text: str) -> list[str]:
    """Tokenize alphabetic words for coarse descriptive metrics."""
    return WORD_RE.findall(text)


def _split_sentences(text: str) -> list[str]:
    """Split sentences with a lightweight regex that avoids NLTK downloads."""
    return [sentence.strip() for sentence in SENTENCE_RE.findall(text) if sentence.strip()]


def _count_syllables(word: str) -> int:
    """Estimate English syllables for readability formulas.

    This is intentionally dependency-free. It is a coarse estimator, but it is
    sufficient for comparing register/readability across datasets.
    """
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _detect_language(text: str) -> tuple[str, float]:
    """Return the top detected language and probability from langdetect."""
    if len(text.strip()) < 20:
        return "unknown", 0.0
    try:
        candidates = detect_langs(text[:5000])
    except LangDetectException:
        return "unknown", 0.0
    if not candidates:
        return "unknown", 0.0
    return candidates[0].lang, float(candidates[0].prob)


def _register_label(row: pd.Series) -> str:
    """Assign a coarse language-register label from interpretable surface cues."""
    if row["academic_marker_rate"] >= 0.015 and row["avg_sentence_len_words"] >= 18:
        return "academic/formal"
    if row["informal_marker_rate"] >= 0.01 or row["emoticon_emoji_rate"] >= 0.002:
        return "informal/conversational"
    if row["avg_sentence_len_words"] < 10 and row["first_person_rate"] >= 0.04:
        return "personal/narrative"
    if row["avg_sentence_len_words"] >= 22:
        return "formal/dense"
    return "neutral"


def compute_text_metrics(text: str) -> dict[str, float | int | str]:
    """Compute per-text descriptive features.

    Metrics cover additional ideas beyond the requested items: digit rate,
    uppercase rate, lexical diversity, average sentence length, question and
    exclamation rates, and paragraph count. These often reveal dataset genre
    differences and preprocessing artifacts.
    """
    chars = list(text)
    char_count = len(chars)
    words = _tokenize_words(text)
    word_count = len(words)
    lower_words = [word.lower() for word in words]
    sentences = _split_sentences(text)
    sentence_count = len(sentences)
    paragraphs = [part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]

    punctuation_count = sum(1 for char in chars if char in string.punctuation)
    digit_count = sum(1 for char in chars if char.isdigit())
    whitespace_count = sum(1 for char in chars if char.isspace())
    uppercase_count = sum(1 for char in chars if char.isupper())
    special_count = sum(
        1
        for char in chars
        if not char.isalnum()
        and not char.isspace()
        and char not in string.punctuation
        and not _is_emoji(char)
    )
    emoji_count = sum(1 for char in chars if _is_emoji(char))
    emoticon_count = len(EMOTICON_RE.findall(text))
    syllable_count = sum(_count_syllables(word) for word in lower_words)

    avg_sentence_len_words = _safe_divide(word_count, sentence_count)
    avg_word_len_chars = _safe_divide(sum(len(word) for word in words), word_count)
    flesch_reading_ease = (
        206.835
        - 1.015 * avg_sentence_len_words
        - 84.6 * _safe_divide(syllable_count, word_count)
        if word_count and sentence_count
        else np.nan
    )
    flesch_kincaid_grade = (
        0.39 * avg_sentence_len_words
        + 11.8 * _safe_divide(syllable_count, word_count)
        - 15.59
        if word_count and sentence_count
        else np.nan
    )

    language, language_probability = _detect_language(text)
    informal_count = sum(1 for word in lower_words if word in INFORMAL_MARKERS)
    first_person_count = sum(1 for word in lower_words if word in FIRST_PERSON)
    academic_count = sum(1 for word in lower_words if word in ACADEMIC_MARKERS)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": len(paragraphs),
        "punctuation_count": punctuation_count,
        "special_character_count": special_count,
        "emoji_count": emoji_count,
        "emoticon_count": emoticon_count,
        "digit_count": digit_count,
        "whitespace_count": whitespace_count,
        "uppercase_count": uppercase_count,
        "punctuation_percentage": 100 * _safe_divide(punctuation_count, char_count),
        "special_character_percentage": 100 * _safe_divide(special_count, char_count),
        "emoji_percentage": 100 * _safe_divide(emoji_count, char_count),
        "emoticon_emoji_rate": _safe_divide(emoticon_count + emoji_count, word_count),
        "digit_percentage": 100 * _safe_divide(digit_count, char_count),
        "uppercase_percentage": 100 * _safe_divide(uppercase_count, char_count),
        "avg_word_len_chars": avg_word_len_chars,
        "avg_sentence_len_words": avg_sentence_len_words,
        "type_token_ratio": _safe_divide(len(set(lower_words)), word_count),
        "question_mark_rate": _safe_divide(text.count("?"), sentence_count),
        "exclamation_mark_rate": _safe_divide(text.count("!"), sentence_count),
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "language": language,
        "language_probability": language_probability,
        "informal_marker_rate": _safe_divide(informal_count, word_count),
        "first_person_rate": _safe_divide(first_person_count, word_count),
        "academic_marker_rate": _safe_divide(academic_count, word_count),
    }


class DeepDatasetAnalysis:
    """Create per-text and aggregate descriptive analyses for one dataset."""

    def __init__(self, visualization):
        self.visualization = visualization
        self.dataset_name = visualization.name
        self.output_dir = SAVE_BASE / self.dataset_name / "deep_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _extract_unique_texts(self) -> pd.DataFrame:
        """Fetch each unique original text once from MongoDB for this dataset."""
        dataset = self.visualization.dataset
        id_author_pairs = {
            row["left_id"]: row["left_author"] for _, row in dataset.iterrows()
        }
        id_author_pairs.update(
            {row["right_id"]: row["right_author"] for _, row in dataset.iterrows()}
        )

        rows = []
        for text_id, author in id_author_pairs.items():
            text, _ = self.visualization.mongoDB.get_text_or_id_from_orginal_collection(
                text=None,
                text_id=text_id,
            )
            rows.append(
                {
                    "dataset_name": self.dataset_name,
                    "text_id": str(text_id),
                    "author": author,
                    "text": text or "",
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _summarize_metrics(per_text_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate numeric metrics with robust descriptive statistics."""
        numeric_cols = per_text_df.select_dtypes(include=[np.number]).columns
        summary = per_text_df[numeric_cols].agg(["mean", "std", "min", "median", "max"]).T
        summary = summary.reset_index(names="metric")
        summary["missing_count"] = per_text_df[numeric_cols].isna().sum().values
        return summary

    @staticmethod
    def _language_distribution(per_text_df: pd.DataFrame) -> pd.DataFrame:
        """Build absolute and relative language counts."""
        counts = Counter(per_text_df["language"].fillna("unknown"))
        total = sum(counts.values())
        return pd.DataFrame(
            [
                {
                    "language": language,
                    "absolute_count": count,
                    "relative_percentage": 100 * _safe_divide(count, total),
                }
                for language, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            ]
        )

    def run(self) -> DatasetAnalysisOutput:
        """Compute all metrics and write CSV outputs for this dataset."""
        logger.info("Extracting unique texts for dataset=%s.", self.dataset_name)
        text_df = self._extract_unique_texts()
        logger.info("Computing metrics for %d texts.", len(text_df))

        metrics_df = pd.DataFrame(
            [compute_text_metrics(text) for text in text_df["text"]]
        )
        per_text_df = pd.concat([text_df.drop(columns=["text"]), metrics_df], axis=1)
        per_text_df["register_label"] = per_text_df.apply(_register_label, axis=1)

        summary_df = self._summarize_metrics(per_text_df)
        summary_df.insert(0, "dataset_name", self.dataset_name)

        register_distribution = (
            per_text_df["register_label"]
            .value_counts(normalize=False)
            .rename_axis("register_label")
            .reset_index(name="absolute_count")
        )
        register_distribution["relative_percentage"] = (
            100 * register_distribution["absolute_count"] / len(per_text_df)
        )
        register_distribution.insert(0, "dataset_name", self.dataset_name)

        language_df = self._language_distribution(per_text_df)
        language_df.insert(0, "dataset_name", self.dataset_name)

        per_text_path = self.output_dir / f"{self.dataset_name}_per_text_deep_analysis.csv"
        summary_path = self.output_dir / f"{self.dataset_name}_deep_analysis_summary.csv"
        language_path = self.output_dir / f"{self.dataset_name}_language_distribution.csv"
        register_path = self.output_dir / f"{self.dataset_name}_register_distribution.csv"

        per_text_df.to_csv(per_text_path, index=False, float_format="%.6f")
        summary_df.to_csv(summary_path, index=False, float_format="%.6f")
        language_df.to_csv(language_path, index=False, float_format="%.6f")
        register_distribution.to_csv(register_path, index=False, float_format="%.6f")

        logger.info("Saved per-text analysis to %s.", per_text_path)
        logger.info("Saved summary analysis to %s.", summary_path)
        logger.info("Saved language distribution to %s.", language_path)
        logger.info("Saved register distribution to %s.", register_path)

        return DatasetAnalysisOutput(
            dataset_name=self.dataset_name,
            per_text_path=per_text_path,
            summary_path=summary_path,
            language_distribution_path=language_path,
            register_distribution_path=register_path,
        )


class DeepDatasetPlots:
    """Create comparison plots from deep-analysis CSV outputs."""

    def __init__(self, output_dir: Path = SAVE_BASE / "deep_analysis_plots"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="paper")

    def _save_fig(self, fig, filename_stem: str) -> None:
        """Save a figure as SVG and PDF."""
        for extension in ["svg", "pdf"]:
            path = self.output_dir / f"{filename_stem}.{extension}"
            fig.savefig(path, bbox_inches="tight")
            logger.info("Saved plot to %s.", path)
        plt.close(fig)

    @staticmethod
    def _dataset_label(dataset_name: str) -> str:
        """Return a publication-friendly dataset label."""
        return CONFIG.DATASET_TRANSLATIONS.get(dataset_name, dataset_name)

    def plot_language_distribution(self, language_df: pd.DataFrame) -> None:
        """Plot language percentages as a stacked bar chart per dataset."""
        if language_df.empty:
            logger.warning("Skipping language distribution plot because input is empty.")
            return

        pivot = language_df.pivot_table(
            index="dataset_name",
            columns="language",
            values="relative_percentage",
            fill_value=0,
        )
        pivot = pivot.loc[sorted(pivot.index)]
        pivot.index = [self._dataset_label(dataset_name) for dataset_name in pivot.index]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        pivot.plot(kind="bar", stacked=True, ax=ax, width=0.75)
        ax.set_title("Language Distribution per Dataset")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Texts (%)")
        ax.set_ylim(0, 100)
        ax.legend(title="Language", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=0)
        self._save_fig(fig, "language_distribution_stacked")

    def plot_register_distribution(self, register_df: pd.DataFrame) -> None:
        """Plot register-label percentages as a stacked bar chart per dataset."""
        if register_df.empty:
            logger.warning("Skipping register distribution plot because input is empty.")
            return

        pivot = register_df.pivot_table(
            index="dataset_name",
            columns="register_label",
            values="relative_percentage",
            fill_value=0,
        )
        pivot = pivot.loc[sorted(pivot.index)]
        pivot.index = [self._dataset_label(dataset_name) for dataset_name in pivot.index]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        pivot.plot(kind="bar", stacked=True, ax=ax, width=0.75)
        ax.set_title("Language Register Distribution per Dataset")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Texts (%)")
        ax.set_ylim(0, 100)
        ax.legend(title="Register", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=0)
        self._save_fig(fig, "register_distribution_stacked")

    def plot_surface_feature_means(self, per_text_df: pd.DataFrame) -> None:
        """Plot selected surface-form percentages averaged by dataset."""
        if per_text_df.empty:
            logger.warning("Skipping surface-feature plot because input is empty.")
            return

        feature_cols = [
            "punctuation_percentage",
            "special_character_percentage",
            "emoji_percentage",
            "digit_percentage",
            "uppercase_percentage",
        ]
        available_cols = [col for col in feature_cols if col in per_text_df.columns]
        if not available_cols:
            logger.warning("Skipping surface-feature plot because no feature columns exist.")
            return

        plot_df = (
            per_text_df.groupby("dataset_name")[available_cols]
            .mean()
            .reset_index()
            .melt(
                id_vars="dataset_name",
                var_name="feature",
                value_name="mean_percentage",
            )
        )
        plot_df["dataset_label"] = plot_df["dataset_name"].map(self._dataset_label)
        plot_df["feature"] = plot_df["feature"].str.replace("_percentage", "", regex=False)
        plot_df["feature"] = plot_df["feature"].str.replace("_", " ").str.title()

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(
            data=plot_df,
            x="dataset_label",
            y="mean_percentage",
            hue="feature",
            ax=ax,
        )
        ax.set_title("Mean Surface-Form Percentages per Dataset")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Mean percentage of characters")
        ax.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left")
        self._save_fig(fig, "surface_feature_percentages")

    def plot_readability_distribution(self, per_text_df: pd.DataFrame) -> None:
        """Plot readability-score distributions for each dataset."""
        if per_text_df.empty or "flesch_reading_ease" not in per_text_df.columns:
            logger.warning("Skipping readability plot because input is empty.")
            return

        plot_df = per_text_df.copy()
        plot_df["dataset_label"] = plot_df["dataset_name"].map(self._dataset_label)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(
            data=plot_df,
            x="dataset_label",
            y="flesch_reading_ease",
            ax=ax,
        )
        sns.stripplot(
            data=plot_df,
            x="dataset_label",
            y="flesch_reading_ease",
            color="black",
            alpha=0.15,
            size=2,
            ax=ax,
        )
        ax.set_title("Readability Distribution per Dataset")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Flesch Reading Ease")
        self._save_fig(fig, "readability_distribution")

    def plot_all(
        self,
        per_text_df: pd.DataFrame,
        language_df: pd.DataFrame,
        register_df: pd.DataFrame,
    ) -> None:
        """Create all deep-analysis plots."""
        self.plot_language_distribution(language_df)
        self.plot_register_distribution(register_df)
        self.plot_surface_feature_means(per_text_df)
        self.plot_readability_distribution(per_text_df)


def run_datasets(dataset_names: Optional[Iterable[str]] = None) -> list[DatasetAnalysisOutput]:
    """Run deep analysis for the requested datasets."""
    dataset_names = list(dataset_names or [CONFIG.BLOG, CONFIG.STUDENT_ESSAYS])
    visualization_by_name = {
        CONFIG.BLOG: BlogVisualization,
        CONFIG.STUDENT_ESSAYS: StudentEssaysVisualization,
    }

    outputs = []
    for dataset_name in dataset_names:
        if dataset_name not in visualization_by_name:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. Supported datasets: "
                f"{sorted(visualization_by_name)}"
            )
        visualization = visualization_by_name[dataset_name]()
        outputs.append(DeepDatasetAnalysis(visualization).run())

    per_text_frames = [pd.read_csv(output.per_text_path) for output in outputs]
    language_frames = [pd.read_csv(output.language_distribution_path) for output in outputs]
    register_frames = [pd.read_csv(output.register_distribution_path) for output in outputs]
    DeepDatasetPlots().plot_all(
        per_text_df=pd.concat(per_text_frames, ignore_index=True),
        language_df=pd.concat(language_frames, ignore_index=True),
        register_df=pd.concat(register_frames, ignore_index=True),
    )
    return outputs


def main() -> None:
    """Run the default deep analysis for blog and student essays datasets."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    outputs = run_datasets([CONFIG.BLOG, CONFIG.STUDENT_ESSAYS])
    for output in outputs:
        logger.info(
            "Finished %s: %s, %s, %s, %s",
            output.dataset_name,
            output.per_text_path,
            output.summary_path,
            output.language_distribution_path,
            output.register_distribution_path,
        )


if __name__ == "__main__":
    main()
