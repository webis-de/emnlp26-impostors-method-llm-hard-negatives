# Copyright 2024 Klara M. Gutekunst, Webis
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

import logging
import re
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from genai_detection.config import CONFIG


class ParaphrasePlotter:
    def __init__(self, save_base_path: Path):
        self.save_base_path = save_base_path
        self.paper_palette_name = "colorblind"
        sns.set_theme(context="paper", style="whitegrid", palette=self.paper_palette_name)
        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "axes.titlesize": 12,
                "axes.labelsize": 10.5,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "legend.title_fontsize": 10,
                # "font.family": "serif",
                # "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            }
        )

    def plot_metric_scatter(
        self,
        df: pd.DataFrame,
        data_category: Optional[str] = None,
        group_by: Optional[str] = "model",
        display_plot: bool = True,
    ):
        save_path = self.save_base_path / "metric_scatter"
        save_path.mkdir(parents=True, exist_ok=True)
        if group_by == "model" and "Paraphraser" not in df.columns:
            data = df.rename(columns={group_by: "Paraphraser"}, inplace=False)
            group_by = "Paraphraser"
        else:
            data = df.copy()
        required_cols = ["sem_sim_avg", "syn_sim_avg", group_by]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

        x_min, x_max = data["sem_sim_avg"].min(), data["sem_sim_avg"].max()
        y_min, y_max = data["syn_sim_avg"].min(), data["syn_sim_avg"].max()

        x_pad = (x_max - x_min) * 0.05 if (x_max - x_min) > 0 else 0.05
        y_pad = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.05

        x_lim = (max(0, x_min - x_pad), min(1, x_max + x_pad))
        y_lim = (max(0, y_min - y_pad), min(1, y_max + y_pad))

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        unique_labels = data[group_by].unique()
        palette = sns.color_palette("tab20", n_colors=len(unique_labels))
        label_to_color = {
            label: palette[i % len(palette)] for i, label in enumerate(unique_labels)
        }
        sns.scatterplot(
            data=data,
            x="sem_sim_avg",
            y="syn_sim_avg",
            hue=group_by,
            palette=label_to_color,
            alpha=0.7,
            s=100,
            edgecolor="k",
            ax=ax,
        )

        ax.set_xlabel("Semantic Similarity (sem_sim_avg)")
        ax.set_ylabel("Syntactic Similarity (syn_sim_avg)")
        title = (
            f"Semantic vs Syntactic Similarity\non {' '.join(word.capitalize() for word in data_category.split())} Dataset, grouped by {group_by.capitalize()}"
            if data_category
            else f"Semantic vs Syntactic Similarity\ngrouped by {group_by.capitalize()}"
        )
        ax.set_title(title)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.grid(True)

        legend_patches = [
            mpatches.Patch(color=color, label=self._wrap_label(label=label))
            for label, color in label_to_color.items()
        ]
        ax.legend(
            handles=legend_patches,
            title=group_by.capitalize(),
            loc="upper left",
            bbox_to_anchor=(1.03, 1),
            borderaxespad=0.0,
            frameon=True,
            fontsize=9,
        )

        inset_size = 0.25
        inset_ax = fig.add_axes([0.9, 0.1, inset_size, inset_size])

        sns.scatterplot(
            data=data,
            x="sem_sim_avg",
            y="syn_sim_avg",
            hue=group_by,
            palette="tab20",
            alpha=0.7,
            s=40,
            edgecolor="k",
            legend=False,
            ax=inset_ax,
        )

        inset_ax.set_xlim(0, 1)
        inset_ax.set_ylim(0, 1)
        inset_ax.set_title("Full range")
        inset_ax.grid(True)
        inset_ax.set_xticks([0, 0.5, 1])
        inset_ax.set_yticks([0, 0.5, 1])
        inset_ax.tick_params(axis="both", which="major", labelsize=8)

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            for format in ["svg", "pdf"]:
                full_path = (
                    save_path
                    / f"{data_category.replace(' ', '_')}_sem_syn_scatter_grouped_by_{group_by}.{format}"
                )
                plt.savefig(
                    full_path, bbox_inches="tight", transparent=True, format=format
                )
                logging.info(f"Plot saved to {full_path}")

        if display_plot:
            plt.show()
        plt.close()

    def _plot_one_plot_per_metric_distribution(
        self,
        data: pd.DataFrame,
        metric_names: list,
        group_by: str,
        data_category: str,
        display_plot: bool = False,
    ):
        unique_labels = data[group_by].unique()
        palette = sns.color_palette("tab20", n_colors=len(unique_labels))

        for metric in metric_names:
            for scale in ["linear", "symlog"]:
                sns.kdeplot(
                    data=data,
                    x=metric,
                    hue=group_by,
                    fill=True,
                    common_norm=False,
                    alpha=0.4,
                    palette=palette,
                    legend=True,
                )
                plt.yscale(scale)
                if scale == "symlog":
                    plt.grid(which="both", linestyle="--", linewidth=0.5)

                metric_for_tile = " ".join([t.capitalize() for t in metric.split("_")])
                if data_category:
                    plt.title(
                        f"Distribution of {metric_for_tile}\non {data_category} Dataset, grouped by {group_by.capitalize()}"
                    )
                else:
                    plt.title(
                        f"Distribution of {metric_for_tile}\ngrouped by {group_by.capitalize()}"
                    )
                plt.xlabel(metric)
                plt.ylabel("Density")
                min_val = data[metric].min()
                max_val = data[metric].max()
                plt.xlim(left=max(0, min_val), right=min(1, max_val))
                handles = [
                    mpatches.Patch(color=palette[i], label=self._wrap_label(str(label)))
                    for i, label in enumerate(unique_labels)
                ]
                plt.legend(
                    handles=handles,
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1),
                    title=group_by.capitalize(),
                    fontsize=10,
                    title_fontsize=12,
                )

                plt.tight_layout()

                safe_metric = str(metric).replace(" ", "_").replace("/", "_")
                safe_category = (
                    str(data_category).replace(" ", "_") if data_category else "dataset"
                )

                path2dir = (
                    self.save_base_path
                    / "metric_distributions"
                    / "distribution_per_metric"
                    / f"{scale}_scale"
                )
                path2dir.mkdir(parents=True, exist_ok=True)

                for format in ["svg", "pdf"]:
                    file_name = f"{safe_category}_{safe_metric}_grouped_by_{group_by}_{scale}_scale.{format}"
                    full_path = path2dir / file_name
                    plt.savefig(
                        full_path, bbox_inches="tight", transparent=True, format=format
                    )
                logging.info(f"Plot saved to {full_path}")

                if display_plot:
                    plt.show()
                plt.close()

    def _prepare_plot_data(
        self,
        df: pd.DataFrame,
        data_category: Optional[str],
        group_by: str,
        technique_label_mode: str = "model_prompt_approach",
        prompt_words: int = 6,
        include_approaches: tuple[str, ...] = ("non_naive", "naive"),
    ) -> tuple[pd.DataFrame, str]:
        data = df.copy()
        if data_category and "dataset_name" in data.columns:
            data = data[data["dataset_name"] == data_category].copy()

        if include_approaches and "paraphrase_approach" in data.columns:
            data = data[data["paraphrase_approach"].isin(include_approaches)].copy()

        resolved_group_by = group_by
        if group_by == "paraphrase_technique":
            data[group_by] = self._build_technique_labels(
                data=data,
                label_mode=technique_label_mode,
                prompt_words=prompt_words,
            )
        elif group_by not in data.columns:
            raise ValueError(f"Group by column '{group_by}' not found in DataFrame.")

        if data.empty:
            raise ValueError("No rows left to plot after filtering.")
        return data, resolved_group_by

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        prompt = str(prompt).replace("<TEXT>", "").strip()
        prompt = re.sub(r"\s+", " ", prompt)
        return prompt

    def _prompt_signature(self, prompt: str, words: int = 6) -> str:
        norm = self._normalize_prompt(prompt)
        if not norm:
            return "prompt: n/a"
        tokens = norm.split()
        if len(tokens) <= words:
            return f"prompt: {norm}"
        return f"prompt: {' '.join(tokens[:words])}..."

    def _build_technique_labels(
        self, data: pd.DataFrame, label_mode: str, prompt_words: int
    ) -> pd.Series:
        valid_modes = {"model", "prompt", "model_prompt", "model_prompt_approach"}
        if label_mode not in valid_modes:
            raise ValueError(
                f"Unsupported technique_label_mode '{label_mode}'. Use one of: "
                f"{sorted(valid_modes)}"
            )

        model_series = (
            data["model"].astype(str)
            if "model" in data.columns
            else (
                data["llm"].astype(str)
                if "llm" in data.columns
                else pd.Series(["unknown_model"] * len(data), index=data.index)
            )
        )
        approach_series = (
            data["paraphrase_approach"].astype(str)
            if "paraphrase_approach" in data.columns
            else pd.Series(["unknown_approach"] * len(data), index=data.index)
        )
        prompt_series = (
            data["prompt"].fillna("").astype(str)
            if "prompt" in data.columns
            else pd.Series([""] * len(data), index=data.index)
        )
        prompt_signatures = prompt_series.apply(
            lambda p: self._prompt_signature(prompt=p, words=prompt_words)
        )

        if label_mode == "model":
            return model_series
        if label_mode == "prompt":
            return prompt_signatures
        if label_mode == "model_prompt":
            return model_series + " | " + prompt_signatures
        return approach_series + " | " + model_series + " | " + prompt_signatures

    def _labels_with_counts(
        self, counts: pd.Series, words_per_line: int = 4
    ) -> dict[str, str]:
        label_map = {}
        for label, count in counts.items():
            wrapped = self._wrap_label(str(label), words_per_line=words_per_line)
            label_map[label] = f"{wrapped} [n={int(count)}]"
        return label_map

    def plot_length_percentage_boxplot(
        self,
        df: pd.DataFrame,
        data_category: Optional[str] = None,
        group_by: str = "paraphrase_technique",
        technique_label_mode: str = "model_prompt_approach",
        prompt_words: int = 6,
        display_plot: bool = True,
    ):
        data, group_by = self._prepare_plot_data(
            df=df,
            data_category=data_category,
            group_by=group_by,
            technique_label_mode=technique_label_mode,
            prompt_words=prompt_words,
        )
        if "paraphrase_length_pct_words" not in data.columns:
            if {"paraphrased_text", "original_text"}.issubset(data.columns):
                original_lengths = data["original_text"].fillna("").astype(str).str.split().str.len()
                paraphrase_lengths = data["paraphrased_text"].fillna("").astype(str).str.split().str.len()
                data["paraphrase_length_pct_words"] = np.where(
                    original_lengths > 0,
                    (paraphrase_lengths / original_lengths) * 100,
                    np.nan,
                )
            else:
                raise ValueError(
                    "Missing 'paraphrase_length_pct_words' and source text columns."
                )

        plot_df = data[[group_by, "paraphrase_length_pct_words"]].dropna().copy()
        if plot_df.empty:
            raise ValueError("No valid rows to plot for paraphrase length percentage.")

        counts = plot_df.groupby(group_by, sort=True)["paraphrase_length_pct_words"].count()
        label_map = self._labels_with_counts(counts=counts)
        plot_df["_group_label"] = plot_df[group_by].map(label_map)
        order = [label_map[label] for label in counts.index]
        palette = sns.color_palette(self.paper_palette_name, n_colors=len(order))
        label_to_color = {
            label: palette[i % len(palette)] for i, label in enumerate(order)
        }

        fig, ax = plt.subplots(
            figsize=(max(10, len(order) * 1.2), 6), constrained_layout=True
        )
        sns.boxplot(
            data=plot_df,
            x="_group_label",
            y="paraphrase_length_pct_words",
            hue="_group_label",
            order=order,
            palette=label_to_color,
            dodge=False,
            showfliers=False,
            linewidth=1.1,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_xlabel("Paraphrase technique")
        ax.set_ylabel("Paraphrase length (% of original words)")
        title = (
            f"Paraphrase Length in Words (% of Original)\n{data_category} Dataset"
            if data_category
            else "Paraphrase Length in Words (% of Original)"
        )
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        sns.despine(ax=ax)

        save_path = self.save_base_path / "metric_boxplots"
        save_path.mkdir(parents=True, exist_ok=True)
        safe_category = (
            str(data_category).replace(" ", "_").replace("/", "_")
            if data_category
            else "all_datasets"
        )
        for format in ["svg", "pdf"]:
            out = save_path / f"{safe_category}_paraphrase_length_pct_words_grouped_by_{group_by}.{format}"
            fig.savefig(out, bbox_inches="tight", transparent=True, format=format)
            logging.info(f"Plot saved to {out}")
        if display_plot:
            plt.show()
        plt.close()

    def plot_metric_boxplots_per_metric(
        self,
        df: pd.DataFrame,
        metric_names: list[str],
        data_category: Optional[str] = None,
        group_by: str = "paraphrase_technique",
        technique_label_mode: str = "model_prompt_approach",
        prompt_words: int = 6,
        display_plot: bool = True,
    ):
        data, group_by = self._prepare_plot_data(
            df=df,
            data_category=data_category,
            group_by=group_by,
            technique_label_mode=technique_label_mode,
            prompt_words=prompt_words,
        )
        metric_names = [metric for metric in metric_names if metric in data.columns]
        if not metric_names:
            raise ValueError("No valid metrics found in DataFrame.")

        save_path = self.save_base_path / "metric_boxplots" / "per_metric"
        save_path.mkdir(parents=True, exist_ok=True)
        safe_category = (
            str(data_category).replace(" ", "_").replace("/", "_")
            if data_category
            else "all_datasets"
        )

        for metric in metric_names:
            plot_df = data[[group_by, metric]].dropna().copy()
            if plot_df.empty:
                logging.warning(f"Skipping metric '{metric}' because it has no data.")
                continue

            counts = plot_df.groupby(group_by, sort=True)[metric].count()
            label_map = self._labels_with_counts(counts=counts)
            plot_df["_group_label"] = plot_df[group_by].map(label_map)
            order = [label_map[label] for label in counts.index]

            palette = sns.color_palette(self.paper_palette_name, n_colors=len(order))
            label_to_color = {
                label: palette[i % len(palette)] for i, label in enumerate(order)
            }

            fig, ax = plt.subplots(
                figsize=(max(10, len(order) * 1.2), 6), constrained_layout=True
            )
            sns.boxplot(
                data=plot_df,
                x="_group_label",
                y=metric,
                hue="_group_label",
                order=order,
                palette=label_to_color,
                dodge=False,
                showfliers=False,
                linewidth=1.1,
                ax=ax,
            )
            if ax.legend_ is not None:
                ax.legend_.remove()

            metric_for_title = " ".join([t.capitalize() for t in metric.split("_")])
            ax.set_xlabel("Paraphrase technique")
            ax.set_ylabel(metric_for_title)
            title = (
                f"{metric_for_title} by Paraphrase Technique\n{CONFIG.DATASET_TRANSLATIONS[data_category]} Dataset"
                if data_category
                else f"{metric_for_title} by Paraphrase Technique"
            )
            ax.set_title(title)
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            sns.despine(ax=ax)

            min_val = plot_df[metric].min()
            max_val = plot_df[metric].max()
            if np.isfinite(min_val) and np.isfinite(max_val) and min_val != max_val:
                padding = (max_val - min_val) * 0.05
                ax.set_ylim(min_val - padding, max_val + padding)

            safe_metric = str(metric).replace(" ", "_").replace("/", "_")
            for format in ["svg", "pdf"]:
                out = (
                    save_path
                    / f"{safe_category}_{safe_metric}_boxplot_grouped_by_{group_by}.{format}"
                )
                fig.savefig(out, bbox_inches="tight", transparent=True, format=format)
                logging.info(f"Plot saved to {out}")

            if display_plot:
                plt.show()
            plt.close()

    def _wrap_label(self, label: str, words_per_line: int = 6) -> str:
        assert (
            isinstance(words_per_line, int) and words_per_line > 0
        ), "words_per_line must be a positive integer."
        words = str(label).split()
        return "\n".join(
            [
                " ".join(words[i : i + words_per_line])
                for i in range(0, len(words), words_per_line)
            ]
        )
