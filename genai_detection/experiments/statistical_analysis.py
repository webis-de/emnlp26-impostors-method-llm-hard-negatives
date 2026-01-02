import logging
from pathlib import Path

import bson
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
    Path(__file__).resolve().parents[3]
    / CONFIG.SAVE_PATH
    / "statistical_analysis"
)
LOCAL_SAVE_PATH.mkdir(parents=True, exist_ok=True)


class StatisticalAnalysis:

    def __init__(self):
        self.mongodb = ParaphraseMongoDB()

    def display_p_val_per_side(self):
        """
        Display p-value distributions per side per dataset per impostor
        generation technique per TP, FP, FN, TN.
        """
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
        assert isinstance(gt_data["left_id"][0], bson.objectid.ObjectId) and isinstance(gt_data["right_id"][0], bson.objectid.ObjectId), f"Columns of gt_data have incorrect data type: {type(gt_data['left_id'][0])}, {type(gt_data['right_id'][0])}"
        assert isinstance(impostor_outputs["left_id"][0], bson.objectid.ObjectId) and isinstance(impostor_outputs["right_id"][0],
                                                                  bson.objectid.ObjectId), f"Columns of impostor_outputs have incorrect data type: {type(impostor_outputs['left_id'][0])}, {type(impostor_outputs['right_id'][0])}"
        df = impostor_outputs.merge(
            gt_data,
            on=["left_id", "right_id"],
            how="left",
            validate="many_to_one",
        )

        if df["same"].isna().any():
            logger.warning(f"{len(df['same'].isna())}/{len(df)} impostor outputs have no ground-truth match")

        # --------------------------------------------------------------
        # Confusion matrix label
        # --------------------------------------------------------------
        def confusion_type(row, pred_col:str):
            if row["same"] and row[pred_col]:
                return "TP"
            if not row["same"] and row[pred_col]:
                return "FP"
            if row["same"] and not row[pred_col]:
                return "FN"
            return "TN"

        df["confusion_left"] = df.apply(lambda x: confusion_type(row=x, pred_col="left_disputed_left_candidate_uncorrected_p_value_pred"), axis=1)
        df["confusion_right"] = df.apply(lambda x: confusion_type(row=x, pred_col="left_disputed_right_candidate_uncorrected_p_value_pred"), axis=1)

        # --------------------------------------------------------------
        # Long-format p-values (left / right)
        # --------------------------------------------------------------
        long_df = pd.concat(
            [
                df.assign(
                    side="left",
                    p_value=df[
                        "left_disputed_left_candidate_uncorrected_p_value"
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

        long_df = long_df.dropna(subset=["p_value"])

        # --------------------------------------------------------------
        # Plotting
        # --------------------------------------------------------------
        sns.set(style="whitegrid")

        for dataset_name in long_df["dataset_name"].dropna().unique():
            ds_df = long_df[long_df["dataset_name"] == dataset_name]

            for technique in ds_df["impostor_generation_technique"].unique():
                tech_df = ds_df[
                    ds_df["impostor_generation_technique"] == technique
                    ]

                if tech_df.empty:
                    continue

                for side in ["left", "right"]:

                    fig, ax = plt.subplots(figsize=(12, 6))

                    sns.violinplot(
                        data=tech_df,
                        x=f"confusion_{side}",
                        y="p_value",
                        hue="side",
                        split=True,
                        inner="quartile",
                        ax=ax,
                    )

                    ax.set_title(
                        f"P-value distribution – {dataset_name}\n"
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

                    for fmt in ["pdf", "svg"]:
                        fname = f"{side}_p_vals_{safe_dataset}_{safe_tech}.{fmt}"
                        out_path = LOCAL_SAVE_PATH / fname
                        fig.savefig(out_path, bbox_inches="tight")
                        logger.info("Saved %s", out_path)

                    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    statistical_analysis = StatisticalAnalysis()
    statistical_analysis.display_p_val_per_side()
