import logging
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from genai_detection.config import CONFIG
from genai_detection.mongo_db.mongo_utils import ParaphraseMongoDB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", )

class ParaphraseScoresVisualizer():
    def __init__(self):
        self.mongoDB = ParaphraseMongoDB()
        self.original_collection = self.mongoDB.original_collection
        self.paraphrase_collection = self.mongoDB.non_naive_paraphrase_collection
        self.paraphrase_score_collection = self.mongoDB.paraphrase_score_collection
        self.save_dir_path = (Path(__file__).parent.parent.parent / CONFIG.SAVE_PATH / "openai_paraphrases" /
                           "visualization")
        self.save_dir_path.mkdir(parents=True, exist_ok=True)

    def scores_per_original_documents(self, text_ids: list[str]):
        """
        Fetch paraphrase score documents for multiple original document IDs.
        Create histograms per feature and a combined scatter plot with colors by text_id.
        """

        all_frames = []  # collect each group separately
        color_map = {}  # store assigned colors for legend

        # --- Fetch and store data for each text_id ---
        for idx, text_id in enumerate(text_ids):
            cursor = self.mongoDB.find_document_by_non_id_field(
                collection=self.paraphrase_score_collection,
                document_field_name="text_id",
                document_value=text_id,
            )
            docs = list(cursor)
            if not docs:
                logging.info(f"No paraphrases found for text_id: {text_id}")
                continue

            df = pd.DataFrame(docs)
            df["text_id"] = text_id  # tag it
            all_frames.append(df)

        if not all_frames:
            logging.info("No paraphrase data found.")
            return None

        # --- Combined DataFrame ---
        df_all = pd.concat(all_frames, ignore_index=True)
        numeric_df = df_all.select_dtypes(include="number")

        if numeric_df.empty:
            logging.info("No numeric features found across all documents.")
            return df_all

        # ---------- HISTOGRAMS ----------
        for col in numeric_df.columns:
            plt.figure(figsize=(8, 5))
            for text_id in text_ids:
                df_sub = df_all[df_all["text_id"] == text_id]
                if df_sub.empty:
                    continue
                plt.hist(
                    df_sub[col],
                    bins=20,
                    alpha=0.5,
                    label=f"{text_id}",
                    edgecolor="black",
                )

            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.legend()

            safe_col = col.replace(" ", "_").lower()
            filename = self.save_dir_path / f"multi_{safe_col}.svg"
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        # ---------- SCATTER PLOT (sem vs syn) ----------
        if "sem_sim_avg" in numeric_df.columns and "syn_sim_avg" in numeric_df.columns:

            fig = plt.figure(figsize=(12, 6))

            # Main plot placement (thinner to make space for inset on right)
            main_ax = fig.add_axes([0.08, 0.1, 0.62, 0.8])

            # Assign unique colors automatically
            cmap = plt.get_cmap("tab20b", len(text_ids))

            for idx, text_id in enumerate(text_ids):
                df_sub = df_all[df_all["text_id"] == text_id]
                if df_sub.empty:
                    continue

                color = cmap(idx % cmap.N)
                color_map[text_id] = color

                main_ax.scatter(
                    df_sub["sem_sim_avg"],
                    df_sub["syn_sim_avg"],
                    c=[color_map[text_id]],
                    alpha=0.7,
                    edgecolor="k",
                    label=f"{text_id}",
                )

            main_ax.set_title("Semantic vs. Syntactic Similarity")
            main_ax.set_xlabel(r"Average Semantic Similarity")
            main_ax.set_ylabel(r"Average Syntactic Similarity")
            main_ax.legend(title="text_id")

            # ---- Inset plot outside main plot ----
            inset_size = 0.25
            inset_left = 0.73
            inset_bottom = 0.1
            inset_ax = fig.add_axes([inset_left, inset_bottom, inset_size, inset_size])

            for idx, text_id in enumerate(text_ids):
                df_sub = df_all[df_all["text_id"] == text_id]
                if df_sub.empty:
                    continue

                inset_ax.scatter(
                    df_sub["sem_sim_avg"],
                    df_sub["syn_sim_avg"],
                    c=[color_map[text_id]],
                    alpha=0.7,
                    edgecolor="k",
                    s=10,
                )

            inset_ax.set_xlim(0, 1)
            inset_ax.set_ylim(0, 1)
            inset_ax.set_title("Full range", fontsize=8)
            inset_ax.grid(True)
            inset_ax.set_xticks([0, 0.5, 1])
            inset_ax.set_yticks([0, 0.5, 1])
            inset_ax.tick_params(axis="both", labelsize=8)

            for spine in inset_ax.spines.values():
                spine.set_edgecolor("gray")

            filename = self.save_dir_path / "multi_sem_vs_syn_scatter.svg"
            plt.savefig(filename)
            plt.close()

        logging.info(f"Plots saved to: {self.save_dir_path}")
        return df_all


if __name__ == "__main__":
    # run collection_score.py file before running this to compute all paraphrases scores
    visualizer = ParaphraseScoresVisualizer()
    visualizer.scores_per_original_documents(text_ids=["68f50029edacdf3d5c0279dd", "68f50029edacdf3d5c027d4f"])
