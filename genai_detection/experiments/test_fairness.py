"""Compute MAUI_k fairness scores for impostor-generation approaches.

This experiment implements the Misattribution Unfairness Index (MAUI_k)
introduced by Alipoormolabashi, Patel, and Balasubramanian (ACL 2025),
"Quantifying Misattribution Unfairness in Authorship Attribution".

The paper defines MAUI_k over author rankings in a needle-in-the-haystack
authorship attribution setup. This repository stores pairwise authorship
verification scores, so this module builds an author ranking for each query
document by sorting candidate authors by their verification score. The ranking
step follows the same idea as the Potha2017 ablation in
``ablations/impostor_potha2017.py``: use the rank position induced by
impostor-based scores instead of reducing the evidence to a single thresholded
decision.

By default, scores are loaded from the existing MongoDB output collections.
This avoids triggering live impostor generation, which may require private
datasets or external APIs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Sequence, Tuple

import pandas as pd
from bson import ObjectId

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)

ImpostorTechnique = Literal[
    "on_the_fly_chatnoir",
    "on_the_fly_startpage",
    "in_domain",
    "two_step_llm",
]

DEFAULT_IMPOSTOR_APPROACHES: List[ImpostorTechnique] = [
    "on_the_fly_chatnoir",
    "on_the_fly_startpage",
    "in_domain",
    "two_step_llm",
]
DEFAULT_METHODS = ["potha2017"]
DEFAULT_K_VALUES = [5, 10, 15, 20]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / CONFIG.SAVE_PATH / "fairness"
RESULT_COLUMNS = [
    "dataset_name",
    "method_name",
    "method_label",
    "impostor_technique",
    "impostor_technique_label",
    "k",
    "maui_k",
    "expected_count",
    "excess_count",
    "authors_over_expected",
    "max_top_k_count",
    "n_queries_total",
    "n_queries_used",
    "n_haystack_authors",
    "n_scores_loaded",
    "n_pairs_aligned",
    "valid",
    "note",
]
METHOD_ALIASES = {
    "ablations/impostor_potha2017.py": "potha2017",
    "impostor_potha2017.py": "potha2017",
    "potha2017impostordetector": "potha2017",
    "ablations/impostor_bdi.py": "bdi",
    "impostor_bdi.py": "bdi",
    "bdiimpostordetector": "bdi",
    "ablations/impostor_homotopy.py": "homotopy",
    "impostor_homotopy.py": "homotopy",
    "hbcimpostordetector": "homotopy",
    "ablations/impostor_asgalf.py": "asgalf",
    "impostor_asgalf.py": "asgalf",
    "asgalfimpostordetector": "asgalf",
    "ablations/std_impostor.py": "std_impostor",
    "std_impostor.py": "std_impostor",
    "stdimpostor": "std_impostor",
}

PairKey = Tuple[ObjectId, ObjectId]


@dataclass(frozen=True)
class PairMetadata:
    """Metadata needed to turn a pairwise score into an author ranking item."""

    left_id: ObjectId
    right_id: ObjectId
    left_author: str
    right_author: str
    same: bool | None = None

    @property
    def pair_key(self) -> PairKey:
        return self.left_id, self.right_id


@dataclass(frozen=True)
class RankedCandidate:
    """One candidate author in a score-sorted ranking for a query document."""

    author: str
    score: float
    rank: int
    is_query_author: bool


@dataclass(frozen=True)
class QueryRanking:
    """Score-sorted author candidates for one query document."""

    query_id: ObjectId
    query_author: str
    candidates: List[RankedCandidate]


@dataclass(frozen=True)
class MauiResult:
    """Serializable output row for one dataset, method, and k value."""

    dataset_name: str
    method_name: str
    method_label: str
    impostor_technique: str
    impostor_technique_label: str
    k: int
    maui_k: float | None
    expected_count: int | None
    excess_count: float | None
    authors_over_expected: int
    max_top_k_count: int
    n_queries_total: int
    n_queries_used: int
    n_haystack_authors: int
    n_scores_loaded: int
    n_pairs_aligned: int
    valid: bool
    note: str

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "method_name": self.method_name,
            "method_label": self.method_label,
            "impostor_technique": self.impostor_technique,
            "impostor_technique_label": self.impostor_technique_label,
            "k": self.k,
            "maui_k": self.maui_k,
            "expected_count": self.expected_count,
            "excess_count": self.excess_count,
            "authors_over_expected": self.authors_over_expected,
            "max_top_k_count": self.max_top_k_count,
            "n_queries_total": self.n_queries_total,
            "n_queries_used": self.n_queries_used,
            "n_haystack_authors": self.n_haystack_authors,
            "n_scores_loaded": self.n_scores_loaded,
            "n_pairs_aligned": self.n_pairs_aligned,
            "valid": self.valid,
            "note": self.note,
        }


class PairMetadataLoader:
    """Load pair metadata and author IDs from MongoDB pair collections."""

    def __init__(
        self,
        mongo: ParaphraseMongoDB,
        *,
        author_field: str = "author",
        batch_size: int = 1000,
    ) -> None:
        self.mongo = mongo
        self.author_field = author_field
        self.batch_size = batch_size

    def load(self, dataset_name: str, split: str = "all") -> Dict[PairKey, PairMetadata]:
        """Load pair metadata for a dataset.

        Pair documents usually already contain ``left_author`` and
        ``right_author``. If not, authors are read from the original text
        collection using ``author_field``.
        """
        collection = self._collection_for_split(split)
        cursor = collection.find(
            {"dataset_name": dataset_name},
            {
                "left_id": 1,
                "right_id": 1,
                "left_author": 1,
                "right_author": 1,
                "same": 1,
            },
            batch_size=self.batch_size,
        ).sort("_id", 1)

        raw_docs = list(cursor)
        author_by_text_id = self._load_missing_authors(raw_docs)

        metadata: Dict[PairKey, PairMetadata] = {}
        missing_author_ids: set[ObjectId] = set()
        for doc in raw_docs:
            left_id = ObjectId(doc["left_id"])
            right_id = ObjectId(doc["right_id"])
            left_author = doc.get("left_author") or author_by_text_id.get(left_id)
            right_author = doc.get("right_author") or author_by_text_id.get(right_id)

            if left_author is None:
                missing_author_ids.add(left_id)
            if right_author is None:
                missing_author_ids.add(right_id)
            if left_author is None or right_author is None:
                continue

            pair = PairMetadata(
                left_id=left_id,
                right_id=right_id,
                left_author=str(left_author),
                right_author=str(right_author),
                same=doc.get("same"),
            )
            metadata[pair.pair_key] = pair

        if missing_author_ids:
            raise ValueError(
                "Missing author metadata for "
                f"{len(missing_author_ids)} text IDs in dataset '{dataset_name}'. "
                f"Expected pair fields left_author/right_author or original text field '{self.author_field}'."
            )

        logger.info(
            "Loaded %d pair metadata records for dataset=%s split=%s.",
            len(metadata),
            dataset_name,
            split,
        )
        return metadata

    def _collection_for_split(self, split: str):
        if split == "all":
            return self.mongo.all_pairs_collection
        if split == "train":
            return self.mongo.train_pairs_collection
        if split == "test":
            return self.mongo.test_pairs_collection
        raise ValueError(f"Unsupported split: {split}")

    def _load_missing_authors(self, raw_docs: Sequence[Mapping]) -> Dict[ObjectId, str]:
        missing_ids = {
            ObjectId(doc[field])
            for doc in raw_docs
            for author_key, field in [
                ("left_author", "left_id"),
                ("right_author", "right_id"),
            ]
            if doc.get(author_key) is None
        }
        if not missing_ids:
            return {}

        cursor = self.mongo.original_collection.find(
            {"_id": {"$in": list(missing_ids)}},
            {self.author_field: 1},
            batch_size=self.batch_size,
        )
        return {
            doc["_id"]: str(doc[self.author_field])
            for doc in cursor
            if self.author_field in doc and doc[self.author_field] is not None
        }


class VerificationRankingBuilder:
    """Build author rankings from pairwise verification scores."""

    def build(
        self,
        scores_by_pair: Mapping[PairKey, float],
        metadata_by_pair: Mapping[PairKey, PairMetadata],
    ) -> Dict[ObjectId, QueryRanking]:
        """Group scores by query document and rank candidate authors.

        If multiple right-side documents belong to the same candidate author
        for a query, the author's strongest verification score is used. This
        creates one ranking position per author, which is the unit required by
        MAUI_k.
        """
        grouped: dict[ObjectId, dict[str, float]] = {}
        query_authors: dict[ObjectId, str] = {}

        for pair_key, score in scores_by_pair.items():
            metadata = metadata_by_pair.get(pair_key)
            if metadata is None:
                continue

            grouped.setdefault(metadata.left_id, {})
            query_authors[metadata.left_id] = metadata.left_author
            previous = grouped[metadata.left_id].get(metadata.right_author)
            if previous is None or score > previous:
                grouped[metadata.left_id][metadata.right_author] = float(score)

        rankings: Dict[ObjectId, QueryRanking] = {}
        for query_id, author_scores in grouped.items():
            query_author = query_authors[query_id]
            sorted_candidates = sorted(
                author_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            rankings[query_id] = QueryRanking(
                query_id=query_id,
                query_author=query_author,
                candidates=[
                    RankedCandidate(
                        author=author,
                        score=score,
                        rank=rank,
                        is_query_author=(author == query_author),
                    )
                    for rank, (author, score) in enumerate(sorted_candidates, start=1)
                ],
            )

        return rankings


class MauiKCalculator:
    """Compute the Misattribution Unfairness Index for ranked authors.

    For a haystack of ``N_h`` authors and ``N_q`` query rankings, the ACL 2025
    paper defines the expected top-k count under a random ranking as
    ``E_k = ceil(k / N_h * N_q)``. MAUI_k sums the excess top-k counts over
    that expectation and divides by the worst-case excess ``k * (N_q - E_k)``.

    Counts are only incremented for candidates that are not the query author;
    the correct author may still occupy a top-k rank and thereby reduce
    misattribution exposure.
    """

    def compute(
        self,
        rankings: Mapping[ObjectId, QueryRanking],
        *,
        k: int,
        dataset_name: str,
        method_name: str,
        impostor_technique: str,
        n_scores_loaded: int,
        n_pairs_aligned: int,
    ) -> MauiResult:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")

        haystack_authors = {
            candidate.author
            for ranking in rankings.values()
            for candidate in ranking.candidates
        }
        eligible_rankings = [
            ranking for ranking in rankings.values() if len(ranking.candidates) >= k
        ]
        n_haystack_authors = len(haystack_authors)
        n_queries = len(eligible_rankings)

        base = self._empty_result(
            dataset_name=dataset_name,
            method_name=method_name,
            impostor_technique=impostor_technique,
            k=k,
            rankings=rankings,
            n_haystack_authors=n_haystack_authors,
            n_queries_used=n_queries,
            n_scores_loaded=n_scores_loaded,
            n_pairs_aligned=n_pairs_aligned,
        )

        if not rankings:
            return replace(base, note="No query rankings could be built.")
        if n_haystack_authors == 0:
            return replace(base, note="No candidate authors found.")
        if n_queries == 0:
            return replace(base, note=f"No query has at least k={k} ranked candidates.")
        if k > n_haystack_authors:
            return replace(base, note=f"k={k} exceeds haystack size {n_haystack_authors}.")

        expected_count = math.ceil((k / n_haystack_authors) * n_queries)
        denominator = k * (n_queries - expected_count)
        if denominator <= 0:
            return replace(
                base,
                expected_count=expected_count,
                note="MAUI_k denominator is zero; use a smaller k or more queries.",
            )

        top_k_counts: Counter[str] = Counter()
        for ranking in eligible_rankings:
            for candidate in ranking.candidates[:k]:
                if not candidate.is_query_author:
                    top_k_counts[candidate.author] += 1

        excess_by_author = {
            author: max(0, top_k_counts.get(author, 0) - expected_count)
            for author in haystack_authors
        }
        excess_count = float(sum(excess_by_author.values()))
        maui_k = excess_count / denominator

        return MauiResult(
            dataset_name=dataset_name,
            method_name=method_name,
            method_label=CONFIG.LABEL_TRANSLATIONS.get(method_name, method_name),
            impostor_technique=impostor_technique,
            impostor_technique_label=CONFIG.LABEL_TRANSLATIONS.get(
                impostor_technique,
                impostor_technique,
            ),
            k=k,
            maui_k=float(maui_k),
            expected_count=expected_count,
            excess_count=excess_count,
            authors_over_expected=sum(v > 0 for v in excess_by_author.values()),
            max_top_k_count=max(top_k_counts.values(), default=0),
            n_queries_total=len(rankings),
            n_queries_used=n_queries,
            n_haystack_authors=n_haystack_authors,
            n_scores_loaded=n_scores_loaded,
            n_pairs_aligned=n_pairs_aligned,
            valid=True,
            note="",
        )

    def _empty_result(
        self,
        *,
        dataset_name: str,
        method_name: str,
        impostor_technique: str,
        k: int,
        rankings: Mapping[ObjectId, QueryRanking],
        n_haystack_authors: int,
        n_queries_used: int,
        n_scores_loaded: int,
        n_pairs_aligned: int,
    ) -> MauiResult:
        return MauiResult(
            dataset_name=dataset_name,
            method_name=method_name,
            method_label=CONFIG.LABEL_TRANSLATIONS.get(method_name, method_name),
            impostor_technique=impostor_technique,
            impostor_technique_label=CONFIG.LABEL_TRANSLATIONS.get(
                impostor_technique,
                impostor_technique,
            ),
            k=k,
            maui_k=None,
            expected_count=None,
            excess_count=None,
            authors_over_expected=0,
            max_top_k_count=0,
            n_queries_total=len(rankings),
            n_queries_used=n_queries_used,
            n_haystack_authors=n_haystack_authors,
            n_scores_loaded=n_scores_loaded,
            n_pairs_aligned=n_pairs_aligned,
            valid=False,
            note="",
        )


class MauiFairnessExperiment:
    """Compare MAUI_k across methods and impostor-generation approaches."""

    def __init__(
        self,
        *,
        mongo: ParaphraseMongoDB,
        methods: Sequence[str] = DEFAULT_METHODS,
        impostor_techniques: Sequence[str] = DEFAULT_IMPOSTOR_APPROACHES,
        k_values: Sequence[int] = DEFAULT_K_VALUES,
        n_impostors: int = 50,
        n_potential_impostors: int | None = None,
        rounds: int = 100,
        split: str = "all",
        batch_size: int = 1000,
        author_field: str = "author",
    ) -> None:
        self.methods = self._normalize_methods(methods)
        self.impostor_techniques = [
            technique
            for technique in impostor_techniques
            if technique in DEFAULT_IMPOSTOR_APPROACHES
        ]
        skipped_techniques = sorted(set(impostor_techniques) - set(self.impostor_techniques))
        if skipped_techniques:
            logger.warning(
                "Skipping unsupported impostor techniques: %s.",
                skipped_techniques,
            )
        self.k_values = list(k_values)
        self.n_impostors = n_impostors
        self.n_potential_impostors = n_potential_impostors
        self.rounds = rounds
        self.split = split
        self.metadata_loader = PairMetadataLoader(
            mongo,
            author_field=author_field,
            batch_size=batch_size,
        )
        from genai_detection.experiments.reproduction.pan_metrics.pan_data_loader import (
            PANDataLoader,
        )

        self.score_loader = PANDataLoader(mongo=mongo, batch_size=batch_size)
        self.ranking_builder = VerificationRankingBuilder()
        self.calculator = MauiKCalculator()

    def _normalize_methods(self, methods: Sequence[str]) -> List[str]:
        from genai_detection.experiments.reproduction.ablation_args import (
            ABLATION_DETECTORS,
        )

        normalized = []
        skipped = []
        for method in methods:
            method_path = Path(method)
            key = method
            for candidate in (
                method,
                method.lower(),
                method_path.as_posix(),
                method_path.as_posix().lower(),
                method_path.name,
                method_path.name.lower(),
            ):
                if candidate in METHOD_ALIASES:
                    key = METHOD_ALIASES[candidate]
                    break
            if key in ABLATION_DETECTORS:
                normalized.append(key)
            else:
                skipped.append(method)
        if skipped:
            logger.warning("Skipping unsupported methods: %s.", sorted(skipped))
        return normalized

    def run(self, dataset_names: Sequence[str]) -> pd.DataFrame:
        rows: List[dict] = []
        for dataset_name in dataset_names:
            rows.extend(self._run_for_dataset(dataset_name))
        return pd.DataFrame(rows, columns=RESULT_COLUMNS)

    def _run_for_dataset(self, dataset_name: str) -> List[dict]:
        metadata_by_pair = self.metadata_loader.load(
            dataset_name=dataset_name,
            split=self.split,
        )
        rows: List[dict] = []

        for method_name in self.methods:
            for impostor_technique in self.impostor_techniques:
                rows.extend(
                    self._run_for_method_and_technique(
                        dataset_name=dataset_name,
                        method_name=method_name,
                        impostor_technique=impostor_technique,
                        metadata_by_pair=metadata_by_pair,
                    )
                )

        return rows

    def _run_for_method_and_technique(
        self,
        *,
        dataset_name: str,
        method_name: str,
        impostor_technique: str,
        metadata_by_pair: Mapping[PairKey, PairMetadata],
    ) -> List[dict]:
        logger.info(
            "Loading scores for method=%s impostor_technique=%s dataset=%s.",
            method_name,
            impostor_technique,
            dataset_name,
        )
        try:
            scores_by_pair = self.score_loader.load_scores(
                method_name=method_name,
                dataset_name=dataset_name,
                impostor_technique=impostor_technique,
                n_impostors=self.n_impostors,
                n_potential_impostors=self.n_potential_impostors,
                rounds=self.rounds,
            )
        except ValueError as exc:
            logger.info(
                "Skipping method=%s impostor_technique=%s dataset=%s: %s",
                method_name,
                impostor_technique,
                dataset_name,
                exc,
            )
            return []
        if not scores_by_pair:
            logger.info(
                "Skipping method=%s impostor_technique=%s dataset=%s: no stored scores found.",
                method_name,
                impostor_technique,
                dataset_name,
            )
            return []

        aligned_scores = {
            pair_key: score
            for pair_key, score in scores_by_pair.items()
            if pair_key in metadata_by_pair
        }
        if not aligned_scores:
            logger.info(
                "Skipping method=%s impostor_technique=%s dataset=%s: no scores align with pair metadata.",
                method_name,
                impostor_technique,
                dataset_name,
            )
            return []
        if len(aligned_scores) != len(scores_by_pair):
            logger.warning(
                "Aligned %d/%d scores with pair metadata for method=%s impostor_technique=%s dataset=%s.",
                len(aligned_scores),
                len(scores_by_pair),
                method_name,
                impostor_technique,
                dataset_name,
            )

        rankings = self.ranking_builder.build(
            scores_by_pair=aligned_scores,
            metadata_by_pair=metadata_by_pair,
        )

        rows = []
        for k in self.k_values:
            result = self.calculator.compute(
                rankings,
                k=k,
                dataset_name=dataset_name,
                method_name=method_name,
                impostor_technique=impostor_technique,
                n_scores_loaded=len(scores_by_pair),
                n_pairs_aligned=len(aligned_scores),
            )
            rows.append(result.to_dict())

        return rows


class FairnessResultsWriter:
    """Persist MAUI_k results as CSV, JSON, and optional LaTeX."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        results: pd.DataFrame,
        *,
        output_format: str,
        generate_tex: bool,
        caption: str,
        label: str,
        tex_path: Path | None = None,
    ) -> dict[str, Path]:
        written: dict[str, Path] = {}
        if output_format in {"csv", "both"}:
            csv_path = self.output_dir / "maui_scores.csv"
            results.to_csv(csv_path, index=False)
            written["csv"] = csv_path
        if output_format in {"json", "both"}:
            json_path = self.output_dir / "maui_scores.json"
            records = json.loads(results.to_json(orient="records"))
            json_path.write_text(
                json.dumps(records, indent=2),
                encoding="utf-8",
            )
            written["json"] = json_path
        if generate_tex:
            tex_output_path = tex_path or (self.output_dir / "maui_scores.tex")
            tex_output_path.parent.mkdir(parents=True, exist_ok=True)
            tex_output_path.write_text(
                self.to_latex_table(results, caption=caption, label=label),
                encoding="utf-8",
            )
            written["tex"] = tex_output_path
        return written

    def to_latex_table(self, results: pd.DataFrame, *, caption: str, label: str) -> str:
        """Render a compact ACM-style LaTeX table using booktabs commands."""
        columns = [
            "Dataset",
            "Method",
            "Impostor generation",
            "$k$",
            "$MAUI_k$",
            "Queries",
            "Authors",
        ]
        lines = [
            "\\begin{table}",
            f"\\caption{{{_latex_escape(caption)}}}",
            f"\\label{{{_latex_escape(label)}}}",
            "\\begin{tabular}{lllrrrr}",
            "\\toprule",
            " & ".join(columns) + " \\\\",
            "\\midrule",
        ]

        if results.empty:
            lines.append("\\multicolumn{7}{l}{No stored score combinations found.} \\\\")
        else:
            for _, row in results.sort_values(
                ["dataset_name", "method_name", "impostor_technique", "k"]
            ).iterrows():
                maui_value = row["maui_k"]
                maui_text = "--" if pd.isna(maui_value) else f"{float(maui_value):.3f}"
                values = [
                    _latex_escape(str(row["dataset_name"])),
                    _latex_escape(str(row["method_label"])),
                    _latex_escape(str(row["impostor_technique_label"])),
                    str(int(row["k"])),
                    maui_text,
                    str(int(row["n_queries_used"])),
                    str(int(row["n_haystack_authors"])),
                ]
                lines.append(" & ".join(values) + " \\\\")

        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ]
        )
        return "\n".join(lines)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MAUI_k fairness scores for impostor-generation approaches."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[CONFIG.STUDENT_ESSAYS, CONFIG.BLOG],
        help="Dataset names to evaluate.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help=(
            "Detector/scoring methods to evaluate, e.g. potha2017 or "
            "ablations/impostor_potha2017.py. Unsupported methods are skipped."
        ),
    )
    parser.add_argument(
        "--impostor-techniques",
        nargs="+",
        default=DEFAULT_IMPOSTOR_APPROACHES,
        help=(
            "Impostor-generation approaches to compare. Unsupported values are skipped. "
            f"Defaults to: {' '.join(DEFAULT_IMPOSTOR_APPROACHES)}."
        ),
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        help="Top-k values for MAUI_k. The paper reports 5 10 15 20.",
    )
    parser.add_argument("--n-impostors", type=int, default=50)
    parser.add_argument("--n-potential-impostors", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument(
        "--split",
        choices=["all", "train", "test"],
        default="all",
        help="Pair split used as the ranking universe.",
    )
    parser.add_argument(
        "--author-field",
        default="author",
        help="Field in original_texts used when pair docs lack left_author/right_author.",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for fairness result files.",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json", "both"],
        default="both",
        help="Tabular result format to write.",
    )
    parser.add_argument(
        "--generate-tex",
        action="store_true",
        help="Also generate an ACM-style LaTeX table.",
        default=True,
    )
    parser.add_argument(
        "--tex-path",
        type=Path,
        default=None,
        help="Optional path for the generated .tex table.",
    )
    parser.add_argument(
        "--tex-caption",
        default="Misattribution unfairness (MAUI_k) across impostor-generation approaches.",
        help="Caption for the generated LaTeX table.",
    )
    parser.add_argument(
        "--tex-label",
        default="tab:maui-fairness",
        help="Label for the generated LaTeX table.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    mongo = ParaphraseMongoDB(local_ray=os.path.exists("/Users/klara"))
    experiment = MauiFairnessExperiment(
        mongo=mongo,
        methods=args.methods,
        impostor_techniques=args.impostor_techniques,
        k_values=args.k_values,
        n_impostors=args.n_impostors,
        n_potential_impostors=args.n_potential_impostors,
        rounds=args.rounds,
        split=args.split,
        batch_size=args.batch_size,
        author_field=args.author_field,
    )
    results = experiment.run(args.datasets)
    written = FairnessResultsWriter(args.output_dir).save(
        results,
        output_format=args.output_format,
        generate_tex=args.generate_tex,
        caption=args.tex_caption,
        label=args.tex_label,
        tex_path=args.tex_path,
    )
    for kind, path in written.items():
        logger.info("Saved %s results to %s.", kind, path)
    return results


if __name__ == "__main__":
    main()
