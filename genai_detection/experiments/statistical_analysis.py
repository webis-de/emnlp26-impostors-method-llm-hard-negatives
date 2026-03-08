import logging
from pathlib import Path
from time import sleep

import bson
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

LOCAL_SAVE_PATH = (
    Path(__file__).resolve().parents[2]
    / CONFIG.SAVE_PATH
    / "statistical_analysis"
)
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)


class StatisticalAnalysis:

    def __init__(self):
        self.mongodb = ParaphraseMongoDB()

    def _save_fig(self, fname:str, fig):
        for fmt in ["pdf", "svg"]:
            fname_fmt = f"{fname}.{fmt}"
            out_path = LOCAL_SAVE_PATH / fname_fmt
            fig.savefig(out_path, bbox_inches="tight")
            logger.info("Saved %s", out_path)

    def _get_impostor_outputs(self):
        # --------------------------------------------------------------
        # Load impostor outputs
        # --------------------------------------------------------------
        impostor_outputs_cursor = self.mongodb.impostor_output_collection.find({})
        impostor_outputs = pd.DataFrame(impostor_outputs_cursor)
        logger.info(f"Obtained {impostor_outputs.shape[0]} impostor outputs.")

        required_cols = {
            "left_id",
            "right_id",
            "impostor_generation_technique",
            "right_disputed_left_candidate_uncorrected_p_value",
            "left_disputed_right_candidate_uncorrected_p_value",
            "right_disputed_left_candidate_uncorrected_p_value_pred",  # assumed boolean or {0,1}
            "left_disputed_right_candidate_uncorrected_p_value_pred",  # assumed boolean or {0,1}
        }

        missing = required_cols - set(impostor_outputs.columns)
        if missing:
            raise ValueError(f"Missing columns in impostor outputs: {missing}")

        left_ids = impostor_outputs["left_id"].dropna().unique().tolist()
        on_the_fly_cursor = self.mongodb.on_the_fly_collection.find(
            {"text_id": {"$in": left_ids}},
            {"_id": 1, "text_id": 1, "index": 1},
        )
        on_the_fly_df = pd.DataFrame(on_the_fly_cursor)
        if on_the_fly_df.empty:
            on_the_fly_df = pd.DataFrame(columns=["left_id", "index"])
        else:
            # Keep only the first match per text_id and preserve missing matches via left-join.
            on_the_fly_df = (
                on_the_fly_df.sort_values("_id")
                .drop_duplicates(subset=["text_id"], keep="first")
                .rename(columns={"text_id": "left_id"})
            )[["left_id", "index"]]
        impostor_outputs = impostor_outputs.merge(
            on_the_fly_df,
            on=["left_id"],
            how="left",
            validate="many_to_one",
        )

        # --------------------------------------------------------------
        # Load ground truth
        # --------------------------------------------------------------
        gt_cursor = self.mongodb.test_pairs_collection.find(
            {
                # "left_id": {"$in": impostor_outputs["left_id"].unique().tolist()},
                # "right_id": {"$in": impostor_outputs["right_id"].unique().tolist()},
            },
            {
                "_id": 0,
                "left_id": 1,
                "right_id": 1,
                "dataset_name": 1,
                "same": 1,
            },
        )
        gt_data = pd.DataFrame(gt_cursor)
        logger.info(f"Obtained ground truth {gt_data.shape[0]} pairs")
        assert "left_id" in gt_data.columns and "right_id" in gt_data.columns, f"Missing columns in ground truth data with columns: {gt_data.columns}"

        # --------------------------------------------------------------
        # Merge
        # --------------------------------------------------------------
        assert isinstance(gt_data["left_id"][0], bson.objectid.ObjectId) and isinstance(gt_data["right_id"][0],
                                                                                        bson.objectid.ObjectId), f"Columns of gt_data have incorrect data type: {type(gt_data['left_id'][0])}, {type(gt_data['right_id'][0])}"
        assert isinstance(impostor_outputs["left_id"][0], bson.objectid.ObjectId) and isinstance(
            impostor_outputs["right_id"][0],
            bson.objectid.ObjectId), f"Columns of impostor_outputs have incorrect data type: {type(impostor_outputs['left_id'][0])}, {type(impostor_outputs['right_id'][0])}"
        df = impostor_outputs.merge(
            gt_data,
            on=["left_id", "right_id"],
            how="left",
            validate="many_to_one",
        )
        assert len(df['same'].isna()) - len(df) < 0, f"None of {len(df)} impostor outputs have a ground-truth match."

        if df["same"].isna().any():
            logger.warning(f"{len(df['same'].isna())}/{len(df)} impostor outputs have no ground-truth match")
        logger.info(f"Merged ground truth impostor outputs: {df.shape}")

        # --------------------------------------------------------------
        # Confusion matrix label
        # --------------------------------------------------------------
        def confusion_type(row, pred_col: str):
            if row["same"] and row[pred_col]:
                return "TP"
            if not row["same"] and row[pred_col]:
                return "FP"
            if row["same"] and not row[pred_col]:
                return "FN"
            return "TN"

        df["confusion_left"] = df.apply(
            lambda x: confusion_type(row=x, pred_col="left_disputed_right_candidate_uncorrected_p_value_pred"), axis=1)
        df["confusion_right"] = df.apply(
            lambda x: confusion_type(row=x, pred_col="right_disputed_left_candidate_uncorrected_p_value_pred"), axis=1)

        return df

    def display_p_val_per_side(self):
        """
        Display p-value distributions per side per dataset per impostor
        generation technique per TP, FP, FN, TN.
        """
        df = self._get_impostor_outputs()
        # --------------------------------------------------------------
        # Long-format p-values (left / right)
        # --------------------------------------------------------------
        # returns df with side, p_value
        long_df = pd.concat(
            [
                df.assign(
                    side="left",
                    p_value=df[
                        "left_disputed_right_candidate_uncorrected_p_value"
                    ],
                ),
                df.assign(
                    side="right",
                    p_value=df[
                        "right_disputed_left_candidate_uncorrected_p_value"
                    ],
                ),
            ],
            ignore_index=True,
        )
        long_df["confusion"] = long_df.apply(
            lambda r: r["confusion_left"] if r["side"] == "left" else r["confusion_right"],
            axis=1,
        )
        long_df = long_df.dropna(subset=["p_value"])
        logger.info(f"Shape of long dataframe: {long_df.shape}")

        # --------------------------------------------------------------
        # Plotting
        # --------------------------------------------------------------
        sns.set_theme(style="whitegrid", context="paper")

        for dataset_name in long_df["dataset_name"].dropna().unique():
            ds_df = long_df[long_df["dataset_name"] == dataset_name]

            for technique in ds_df["impostor_generation_technique"].unique():
                tech_df = ds_df[
                    ds_df["impostor_generation_technique"] == technique
                    ]
                if tech_df.empty:
                    continue
                if technique == "on_the_fly":
                    print(tech_df.keys())
                    for index in tech_df["index"].unique():
                        tech_df = tech_df[tech_df["index"] == index]
                        self._violine_summary_per_df(tech_df=tech_df, dataset_name=dataset_name,
                                                     technique=f"{technique} ({index}))")
                else:
                    self._violine_summary_per_df(
                        tech_df=tech_df, dataset_name=dataset_name, technique=technique
                    )

    def _violine_summary_per_df(self, tech_df: pd.DataFrame, dataset_name: str, technique:str):
        if tech_df.empty:
            return
        summary = tech_df.groupby(["confusion", "side"])["p_value"].agg(
            ["count", "nunique", "min", "max"]
        )
        print("Summary:\n", summary)

        confusion_order = [
            c
            for c in ["TP", "FP", "FN", "TN"]
            if c in tech_df["confusion"].dropna().unique()
        ]
        if not confusion_order:
            confusion_order = sorted(tech_df["confusion"].dropna().unique())

        confusion_counts = tech_df["confusion"].value_counts()
        fig, ax = plt.subplots(figsize=(12, 6))

        # for side in ["left", "right"]:
        sns.violinplot(
            data=tech_df,
            x="confusion",
            y="p_value",
            hue="side",
            order=confusion_order,
            split=True,
            inner="quart",
            bw_adjust=0.5,
            density_norm="width",
            # cut=0,
            # inner=None,
            ax=ax,
        )

        sns.stripplot(
            data=tech_df,
            x="confusion",
            y="p_value",
            hue="side",
            order=confusion_order,
            legend=False,
            dodge=False,
            alpha=0.1,
            ax=ax,
        )

        ax.set_title(
            f"P-value distribution – {dataset_name} ({len(tech_df)//2} pairs)\n"
            f"Impostor generation: {technique}"
        )
        ax.set_ylabel("Uncorrected p-value")
        ax.set_xlabel("Confusion category")
        tick_labels = [
            f"{label} ({int(confusion_counts.get(label, 0))})"
            for label in confusion_order
        ]
        ax.set_xticks(range(len(confusion_order)))
        ax.set_xticklabels(tick_labels)
        for tick in ax.get_xticklabels():
            if tick.get_text().startswith(("TN ", "TP ")):
                tick.set_fontweight("bold")

        plt.legend(title="Side", loc="upper right")
        plt.tight_layout()

        # --------------------------------------------------
        # Save
        # --------------------------------------------------
        safe_dataset = dataset_name.replace(" ", "_")
        safe_tech = technique.replace(" ", "_").replace("(","").replace(")","")

        self._save_fig(fname=f"p_vals_{safe_dataset}_{safe_tech}", fig=fig)

        plt.close(fig)

    def histogram_per_approach(self):
        """
        Plot histograms showing the number of instances per TP, FP, TN, FN
        for each impostor generation approach, per dataset,
        with one row per side (left / right).
        """
        df = self._get_impostor_outputs()
        df = df.copy()

        def technique_plot_label(row: pd.Series) -> str:
            technique = row["impostor_generation_technique"]
            index = row.get("index")
            if technique == "on_the_fly" and pd.notna(index):
                if isinstance(index, float) and index.is_integer():
                    index = int(index)
                return f"{technique} ({index})"
            return technique

        df["technique_plot"] = df.apply(technique_plot_label, axis=1)

        required_cols = {
            "dataset_name",
            "impostor_generation_technique",
            "confusion_left",
            "confusion_right",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}, only have {df.columns}")

        confusion_order = ["TP", "FP", "FN", "TN"]

        sns.set(style="whitegrid")

        for dataset_name in df["dataset_name"].dropna().unique():
            ds_df = df[df["dataset_name"] == dataset_name]

            if ds_df.empty:
                continue

            hue_order = sorted(ds_df["technique_plot"].dropna().unique())
            on_the_fly_variants = [h for h in hue_order if h.startswith("on_the_fly (")]
            palette = {
                h: CONFIG.LABEL_COLORS.get(h, "#4c4c4c")
                for h in hue_order
            }
            if on_the_fly_variants:
                base_color = CONFIG.LABEL_COLORS.get("on_the_fly", "#9467bd")
                variant_colors = sns.light_palette(
                    base_color, n_colors=len(on_the_fly_variants) + 2
                )[1:-1]
                for label, color in zip(on_the_fly_variants, variant_colors):
                    palette[label] = color

            fig, axes = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(12, 8),
                sharex=True,
            )

            for ax, side in zip(axes, ["left", "right"]):
                counts = (
                    ds_df
                    .groupby(
                        ["technique_plot", f"confusion_{side}"]
                    )
                    .size()
                    .reset_index(name="count")
                )

                # Normalize per impostor generation approach
                counts["percent"] = (
                    counts
                    .groupby("technique_plot")["count"]
                    .transform(lambda x: x / x.sum())
                )

                sns.barplot(
                    data=counts,
                    x=f"confusion_{side}",
                    y="percent",
                    hue="technique_plot",
                    hue_order=hue_order,
                    order=confusion_order,
                    palette=palette,
                    ax=ax,
                )

                ax.set_title(f"{side.capitalize()} side")
                ax.set_ylabel("Percentage of instances")
                ax.set_ylim(0, 1)

                # Remove duplicate legends
                ax.legend_.remove()

            axes[-1].set_xlabel("Confusion category")

            fig.suptitle(
                f"Confusion Matrix Counts per Impostor Generation Approach\n{dataset_name.capitalize().replace('_', ' ')} Dataset",
                y=1.02,
            )

            # Get handles and labels from the first axis
            handles, labels = axes[0].get_legend_handles_labels()

            # Compute total number of instances per impostor generation approach
            # using the full dataset (sum over all confusion categories, any side)
            total_counts = (
                ds_df
                .melt(
                    id_vars=["technique_plot"],
                    value_vars=["confusion_left", "confusion_right"],
                    var_name="side",
                    value_name="confusion"
                )
                .groupby("technique_plot")
                .size()
                .to_dict()
            )

            def translated_label(label: str) -> str:
                if label.startswith("on_the_fly (") and label.endswith(")"):
                    base = "on_the_fly"
                    index = label[len("on_the_fly ("):-1]
                    return f"{CONFIG.LABEL_TRANSLATIONS.get(base, base)} ({index})"
                return CONFIG.LABEL_TRANSLATIONS.get(label, label)

            # Translate labels + append counts
            translated_labels = [
                f"{translated_label(label)} ({total_counts.get(label, 0)//2})"
                for label in labels
            ]

            # Create shared legend
            fig.legend(
                handles,
                translated_labels,
                title="Impostor generation (# Pairs)",
                loc="upper right",
                bbox_to_anchor=(0.99, 1.02),
            )

            plt.tight_layout()

            # --------------------------------------------------
            # Save
            # --------------------------------------------------
            safe_dataset = dataset_name.replace(" ", "_")
            self._save_fig(
                fname=f"histogram_confusion_{safe_dataset}",
                fig=fig,
            )
            plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    statistical_analysis = StatisticalAnalysis()
    # statistical_analysis.display_p_val_per_side()
    statistical_analysis.histogram_per_approach()
