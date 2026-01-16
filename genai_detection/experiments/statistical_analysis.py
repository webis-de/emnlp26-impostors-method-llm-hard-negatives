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

# Fixed colors per label (matplotlib-compatible)
LABEL_COLORS = {
    "in_domain": "#1f77b4",                 # blue
    "on_the_fly": "#9467bd",                # purple
    "one_step_llm": "#8c564b",              # brown
    "two_step_llm": "#e377c2",              # pink
    "unsupervised_baseline_min-max": "#ff7f0e", # orange
    "unsupervised_baseline_cosine": "#2ca02c",  # green
    "supervised_baseline": "#d62728",       # red
    "unmasking": "#7f7f7f",                 # gray
    "ppmd": "#bcbd22",                      # olive
    "translation": "#17becf",               # cyan
}

LEGEND_TRANSLATIONS = {
    "in_domain": "In-Domain",
    "on_the_fly": "Retrieval-Based",
    "one_step_llm": "One-Step Paraphraser (LLM)",
    "two_step_llm": "Two-Step Paraphraser (LLM)",
    "unsupervised_baseline_min-max": "Unsup. Min-Max (B)",
    "unsupervised_baseline_cosine": "Unsup. Cosine (B)",
    "supervised_baseline": "Sup. SVM (B)",
    "unmasking": "Unmasking",
    "ppmd": "PPMd",
    "translation": "Translation",
}

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
        logger.info("Obtained impostor outputs")

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

        for col in ["left_id", "right_id"]:
            impostor_outputs[col] = impostor_outputs[col] = impostor_outputs[col].apply(
                lambda x: bson.ObjectId(x) if pd.notna(x) else x
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
        logger.info(f"Obtained ground truth data.")
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


                # tech_df = tech_df.copy()
                # tech_df["log_p_value"] = np.log10(tech_df["p_value"].clip(lower=1e-300))

                summary = (
                    tech_df
                    .groupby(["confusion", "side"])["p_value"]
                    .agg(["count", "nunique", "min", "max"])
                )

                print("Summary:\n",summary)

                if tech_df.empty:
                    continue
                fig, ax = plt.subplots(figsize=(12, 6))

                # for side in ["left", "right"]:
                sns.violinplot(
                    data=tech_df,
                    x="confusion",
                    y="p_value",
                    hue="side",
                    split=True,
                    inner="quart",
                    bw_adjust=.5,
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

                plt.legend(title="Side", loc="upper right")
                plt.tight_layout()

                # --------------------------------------------------
                # Save
                # --------------------------------------------------
                safe_dataset = dataset_name.replace(" ", "_")
                safe_tech = technique.replace(" ", "_")

                self._save_fig(fname=f"p_vals_{safe_dataset}_{safe_tech}", fig=fig)

                plt.close(fig)

    def histogram_per_approach(self):
        """
        Plot histograms showing the number of instances per TP, FP, TN, FN
        for each impostor generation approach, per dataset,
        with one row per side (left / right).
        """
        df = self._get_impostor_outputs()

        required_cols = {
            "dataset_name",
            "impostor_generation_technique",
            "confusion_left",
            "confusion_right",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        confusion_order = ["TP", "FP", "FN", "TN"]

        sns.set(style="whitegrid")

        for dataset_name in df["dataset_name"].dropna().unique():
            ds_df = df[df["dataset_name"] == dataset_name]

            if ds_df.empty:
                continue

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
                        ["impostor_generation_technique", f"confusion_{side}"]
                    )
                    .size()
                    .reset_index(name="count")
                )

                # Normalize per impostor generation approach
                counts["percent"] = (
                    counts
                    .groupby("impostor_generation_technique")["count"]
                    .transform(lambda x: x / x.sum())
                )

                sns.barplot(
                    data=counts,
                    x=f"confusion_{side}",
                    y="percent",
                    hue="impostor_generation_technique",
                    order=confusion_order,
                    palette=LABEL_COLORS,
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
                    id_vars=["impostor_generation_technique"],
                    value_vars=["confusion_left", "confusion_right"],
                    var_name="side",
                    value_name="confusion"
                )
                .groupby("impostor_generation_technique")
                .size()
                .to_dict()
            )

            # Translate labels + append counts
            translated_labels = [
                f"{LEGEND_TRANSLATIONS.get(label, label)} ({total_counts.get(label, 0)//2})"
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
    statistical_analysis.display_p_val_per_side()
    statistical_analysis.histogram_per_approach()
