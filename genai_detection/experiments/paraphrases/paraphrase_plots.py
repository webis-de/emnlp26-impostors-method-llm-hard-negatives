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
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class ParaphrasePlotter:
    def __init__(self, save_base_path: Path):
        self.save_base_path = save_base_path

    def plot_metric_radar_per_dataset(
        self,
        df_all: pd.DataFrame,
        metrics: list[str],
        save_path: Path | None = None,
        dataset_col: str = "dataset",
        display_plot: bool = True,
    ):
        metrics = [
            m
            for m in metrics
            if m in df_all.columns and pd.api.types.is_numeric_dtype(df_all[m])
        ]
        assert metrics, "No numeric metrics found to plot."
        grouped_mean = df_all.groupby(dataset_col)[metrics].mean().round(2)
        grouped_std = df_all.groupby(dataset_col)[metrics].std().round(2)
        grouped_median = df_all.groupby(dataset_col)[metrics].median().round(2)

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            mean_std_df = pd.concat(
                {"mean": grouped_mean, "std": grouped_std, "median": grouped_median},
                axis=1,
            )
            csv_out = save_path / "extraction_metrics_mean_std_median_per_dataset.csv"
            mean_std_df.to_csv(csv_out)
            logging.info(f"Saved mean/std values as CSV file to {csv_out}")

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        unique_labels = grouped_mean.index
        palette = sns.color_palette(
            "tab20" if len(unique_labels) > 10 else "tab10", n_colors=len(unique_labels)
        )
        label_to_color = {
            label: palette[i % len(palette)] for i, label in enumerate(unique_labels)
        }
        line_styles = [
            "solid",
            "dashed",
            "dotted",
            "dashdot",
            (0, (3, 1, 1, 1)),
            (0, (5, 1)),
        ]

        for i, groupby_value in enumerate(unique_labels):
            mean_values = grouped_mean.loc[groupby_value].tolist()
            std_values = grouped_std.loc[groupby_value].tolist()

            mean_values += mean_values[:1]
            std_values += std_values[:1]

            lower = np.maximum(-1, np.array(mean_values) - np.array(std_values))
            upper = np.minimum(1, np.array(mean_values) + np.array(std_values))

            ax.plot(
                angles,
                mean_values,
                label=self._wrap_label(groupby_value),
                alpha=0.7,
                color=label_to_color[groupby_value],
                linewidth=2,
                linestyle=line_styles[i % len(line_styles)],
            )
            ax.fill_between(
                angles, lower, upper, color=label_to_color[groupby_value], alpha=0.2
            )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.tick_params(axis="y")

        ax.legend(
            loc="lower left",
            bbox_to_anchor=(1.1, 0.8),
            fontsize=10,
            title=dataset_col.capitalize(),
        )

        title = "Radar plot of Metric Distributions by Dataset"
        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            for format in ["svg", "pdf"]:
                out = save_path / f"radar_extraction_quality_per_dataset.{format}"
                fig.savefig(out, bbox_inches="tight", transparent=True, format=format)
                logging.info(f"Saved radar plot to {out}")
        else:
            logging.info("No save path provided, plot not saved.")
        if display_plot:
            plt.show()
        plt.close()

    def plot_models_metrics(
        self,
        df: pd.DataFrame,
        metric_names: list[str],
        data_category: Optional[str] = None,
        group_by: Optional[str] = "model",
        display_plot: bool = True,
    ):
        save_path = self.save_base_path / "radar_charts"
        save_path.mkdir(parents=True, exist_ok=True)
        labels = [metric for metric in metric_names if metric in df.columns]
        assert (
            group_by in df.columns
        ), f"Group by column '{group_by}' not found in DataFrame."
        if group_by == "model" and "Paraphraser" not in df.columns:
            data = df.rename(columns={group_by: "Paraphraser"}, inplace=False)
            group_by = "Paraphraser"
        else:
            data = df.copy()
        grouped_mean = data.groupby(group_by)[labels].mean()
        grouped_std = data.groupby(group_by)[labels].std()

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        unique_labels = grouped_mean.index
        palette = sns.color_palette(
            "tab20" if len(unique_labels) > 10 else "tab10", n_colors=len(unique_labels)
        )
        label_to_color = {
            label: palette[i % len(palette)] for i, label in enumerate(unique_labels)
        }

        for groupby_value in unique_labels:
            mean_values = grouped_mean.loc[groupby_value].tolist()
            std_values = grouped_std.loc[groupby_value].tolist()

            mean_values += mean_values[:1]
            std_values += std_values[:1]

            lower = np.maximum(0, np.array(mean_values) - np.array(std_values))
            upper = np.minimum(1, np.array(mean_values) + np.array(std_values))

            ax.plot(
                angles,
                mean_values,
                label=self._wrap_label(groupby_value),
                alpha=0.7,
                color=label_to_color[groupby_value],
            )
            ax.fill_between(
                angles, lower, upper, color=label_to_color[groupby_value], alpha=0.2
            )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

        ax.set_ylim(0, 1)

        ax.legend(
            loc="lower left",
            bbox_to_anchor=(1.1, 0.7),
            fontsize=9,
            title=group_by.capitalize(),
        )
        title = (
            f"Radar Chart of Paraphrasing Metrics\non {' '.join(word.capitalize() for word in data_category.split())} Dataset, grouped by {group_by.capitalize()}"
            if data_category
            else f"Radar Chart of Paraphrasing Metrics\ngrouped by {group_by.capitalize()}"
        )
        plt.title(title, fontsize=12)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            for format in ["svg", "pdf"]:
                file_name = (
                    save_path
                    / f"{data_category.replace(' ', '_')}_paraphrasing_metrics_grouped_by_{group_by}_radar_chart.{format}"
                )
                plt.savefig(
                    file_name, bbox_inches="tight", transparent=True, format=format
                )
                logging.info(f"Plot saved to {file_name}")
        if display_plot:
            plt.show()
        plt.close()

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

    def plot_metric_distributions(
        self,
        df: pd.DataFrame,
        metric_names: list[str],
        data_category: Optional[str] = None,
        group_by: Optional[str] = "model",
        display_plot: bool = True,
    ):
        save_path = self.save_base_path / "metric_distributions"
        save_path.mkdir(parents=True, exist_ok=True)
        metric_names = [metric for metric in metric_names if metric in df.columns]
        if group_by == "model" and "Paraphraser" not in df.columns:
            data = df.rename(columns={group_by: "Paraphraser"}, inplace=False)
            group_by = "Paraphraser"
        else:
            data = df.copy()
        assert len(metric_names) > 0, "No valid metrics found in DataFrame."
        assert (
            group_by in data.columns
        ), f"Group by column '{group_by}' not found in DataFrame."

        n_metrics = len(metric_names)
        n_cols = 2
        n_rows = (n_metrics + 1) // n_cols

        unique_labels = data[group_by].unique()
        max_words_in_label = max(len(str(label).split()) for label in unique_labels)
        use_shared_legend = max_words_in_label > 3 or len(unique_labels) > 5
        palette = sns.color_palette("tab20", n_colors=len(unique_labels))
        label_to_color = {
            label: palette[i % len(palette)] for i, label in enumerate(unique_labels)
        }

        for scale in ["linear", "symlog"]:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
            axes = axes.flatten()
            for i, metric in enumerate(metric_names):
                ax = axes[i]

                counts = data.groupby(group_by)[metric].count()
                assert (
                    counts != 0
                ).any(), (
                    f"No data points found for metric '{metric}' in group '{group_by}'."
                )

                models_multi = counts[counts > 1].index
                models_single = counts[counts == 1].index

                if len(models_multi) > 0:
                    sns.kdeplot(
                        data=data[data[group_by].isin(models_multi)],
                        x=metric,
                        hue=group_by,
                        fill=True,
                        common_norm=False,
                        alpha=0.4,
                        ax=ax,
                        palette=label_to_color,
                        legend=False,
                    )

                for model in models_single:
                    single_val = data[(data[group_by] == model)][metric].values[0]
                    label = model if not use_shared_legend else None
                    color = label_to_color[model]
                    ax.scatter(
                        single_val,
                        1,
                        label=label,
                        color=color,
                        s=50,
                        edgecolor="k",
                        zorder=5,
                    )
                if scale == "symlog":
                    ax.grid(which="both", linestyle="--", color="gray", alpha=0.5)
                ax.set_yscale(scale)
                metric_for_tile = " ".join([t.capitalize() for t in metric.split("_")])
                ax.set_title(f"Distribution of {metric_for_tile}")
                min_val = data[metric].min()
                max_val = data[metric].max()
                ax.set_xlim(left=max(0, min_val), right=min(1, max_val))
                ax.set_xlabel(metric_for_tile)
                ax.set_ylabel("Density")

            legend_patches = [
                mpatches.Patch(color=color, label=self._wrap_label(label))
                for label, color in label_to_color.items()
            ]
            fig.legend(
                handles=legend_patches,
                loc="upper left",
                bbox_to_anchor=(
                    0.95,
                    0.95,
                ),
                title=group_by.capitalize(),
                frameon=True,
                borderaxespad=0,
                fontsize=10,
                title_fontsize=12,
            )

            for j in range(len(metric_names), len(axes)):
                fig.delaxes(axes[j])
            title = (
                f"Metric Distributions\non {' '.join(word.capitalize() for word in data_category.split())} Dataset, grouped by {group_by.capitalize()}"
                if data_category
                else f"Metric Distributions\ngrouped by {group_by.capitalize()}"
            )
            fig.suptitle(title, fontsize=18)
            plt.tight_layout(rect=[0, 0, 0.95, 0.95])

            if save_path:
                scale_save_path = Path(save_path) / f"{scale}_scale"
                scale_save_path.mkdir(parents=True, exist_ok=True)
                for format in ["svg", "pdf"]:
                    filenaname = f"{data_category.replace(' ', '_')}_metric_distributions_grouped_by_{group_by}_{scale}_scale.{format}"
                    full_path = scale_save_path / filenaname
                    plt.savefig(
                        full_path, bbox_inches="tight", transparent=True, format=format
                    )
                    logging.info(f"Plot saved to {full_path}")

            if display_plot:
                plt.show()
            plt.close()
        self._plot_one_plot_per_metric_distribution(
            data=data,
            metric_names=metric_names,
            group_by=group_by,
            data_category=data_category,
            display_plot=False,
        )

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
